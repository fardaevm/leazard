# agent.py
import os
import json
import logging
import math
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TypedDict, Dict, Any, List

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage

from rag.retriever import LawRetriever

log = logging.getLogger("leazard.agent")

# ── Config (all from .env, no hardcoded values) ───────────────────────────────

# ZIP prefixes that are in scope. Override per deployment.
# e.g. SUPPORTED_ZIP_PREFIXES="94,95" or "100,101" for NYC
SUPPORTED_ZIP_PREFIXES: tuple[str, ...] = tuple(
    p.strip()
    for p in os.getenv("SUPPORTED_ZIP_PREFIXES", "94,95").split(",")
    if p.strip()
)
SUPPORTED_REGION = os.getenv("SUPPORTED_REGION", "San Francisco / California Bay Area")

# Severity → score weight. Change without touching code.
# e.g. SEVERITY_WEIGHTS="Low:1,Medium:3,High:5"
def _parse_weights() -> dict[str, float]:
    raw = os.getenv("SEVERITY_WEIGHTS", "Low:1,Medium:3,High:5")
    out: dict[str, float] = {}
    for pair in raw.split(","):
        k, _, v = pair.partition(":")
        if k.strip() and v.strip():
            out[k.strip()] = float(v.strip())
    return out

SEVERITY_WEIGHTS = _parse_weights()
# Sort order: highest weight = first (most severe first)
SEVERITY_ORDER = {k: i for i, k in enumerate(
    sorted(SEVERITY_WEIGHTS, key=lambda k: -SEVERITY_WEIGHTS[k])
)}

MAX_CATEGORIES    = int(os.getenv("MAX_CATEGORIES", "12"))
LEASE_CHARS       = int(os.getenv("LEASE_EXTRACT_CHARS", "100000"))
LEASE_CHUNK_CHARS = int(os.getenv("LEASE_CHUNK_CHARS", "24000"))
LEASE_CHUNK_OVERLAP = int(os.getenv("LEASE_CHUNK_OVERLAP", "500"))
EXCERPT_WINDOW    = int(os.getenv("EXCERPT_WINDOW_CHARS", "900"))
EXCERPT_MAX       = int(os.getenv("EXCERPT_MAX_CHARS", "1400"))
MAX_RECOMMENDATIONS = int(os.getenv("MAX_RECOMMENDATIONS", "20"))
LEASE_QUOTE_MAX   = int(os.getenv("LEASE_QUOTE_MAX_CHARS", "300"))

# Weighted severity at which the score reaches ~6.3/10 (1 - 1/e). Larger = more lenient.
RISK_SCORE_SCALE  = float(os.getenv("RISK_SCORE_SCALE", "10"))
RISK_LABEL_MODERATE_AT = float(os.getenv("RISK_LABEL_MODERATE_AT", "3.5"))
RISK_LABEL_HIGH_AT     = float(os.getenv("RISK_LABEL_HIGH_AT", "7"))
LETTER_MAX_ISSUES = int(os.getenv("LETTER_MAX_ISSUES", "5"))

_LEASE_JSON_LIST_KEYS = ("other_key_terms",)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_json(raw: str) -> Any:
    """Strip markdown code fences that LLMs often wrap JSON in, then parse."""
    clean = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
    return json.loads(clean)


def _relevant_excerpt(text: str, terms: list[str]) -> str:
    """Slide a window over the lease and return the slice with the most keyword hits."""
    if len(text) <= EXCERPT_MAX:
        return text
    low = text.lower()
    step = max(EXCERPT_WINDOW // 3, 1)
    best_pos, best_count = 0, 0
    for pos in range(0, len(text) - EXCERPT_WINDOW + 1, step):
        chunk = low[pos : pos + EXCERPT_WINDOW]
        count = sum(chunk.count(t.lower()) for t in terms)
        if count > best_count:
            best_count, best_pos = count, pos
    return text[best_pos : best_pos + EXCERPT_MAX]


def _chunk_text(text: str, size: int | None = None, overlap: int | None = None) -> list[str]:
    """Split text into overlapping chunks of at most `size` chars."""
    size = LEASE_CHUNK_CHARS if size is None else size
    overlap = LEASE_CHUNK_OVERLAP if overlap is None else overlap
    if len(text) <= size:
        return [text]
    step = max(size - overlap, 1)
    chunks = []
    for start in range(0, len(text), step):
        chunks.append(text[start : start + size])
        if start + size >= len(text):
            break
    return chunks


def _merge_lease_json(parts: list[dict]) -> dict:
    """Merge per-chunk extractions: first non-empty value wins; list keys are unioned."""
    merged: dict[str, Any] = {}
    for part in parts:
        for key, val in part.items():
            if key in _LEASE_JSON_LIST_KEYS:
                items = merged.setdefault(key, [])
                for v in (val if isinstance(val, list) else []):
                    if v not in items:
                        items.append(v)
            elif key == "lease_type":
                if merged.get(key) in (None, "", "unknown"):
                    merged[key] = val
            elif merged.get(key) in (None, "") and val not in (None, ""):
                merged[key] = val
            else:
                merged.setdefault(key, val)
    return merged


def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def _verify_quote(quote: Any, normalized_lease: str) -> str:
    """Return the quote (whitespace-normalized, capped) only if it appears verbatim in the lease."""
    if not isinstance(quote, str):
        return ""
    q = _normalize_ws(quote)[:LEASE_QUOTE_MAX].strip()
    return q if q and q in normalized_lease else ""


def _score(flags: list[dict]) -> float:
    """Saturating 0-10 score: 10 * (1 - exp(-weighted_severity / RISK_SCORE_SCALE))."""
    total = sum(SEVERITY_WEIGHTS.get((f.get("severity") or "").strip(), 0.0) for f in flags)
    if total <= 0 or RISK_SCORE_SCALE <= 0:
        return 0.0
    return round(10.0 * (1.0 - math.exp(-total / RISK_SCORE_SCALE)), 1)


def _risk_label(score: float) -> str:
    if score >= RISK_LABEL_HIGH_AT:
        return "High"
    if score >= RISK_LABEL_MODERATE_AT:
        return "Moderate"
    return "Low"


def _resolve_citations(support: Any, chunks: list[dict]) -> list[dict]:
    """Map "[n]" ids in `support` to retrieved chunks (1-indexed). Unknown ids are dropped."""
    if not isinstance(support, str):
        return []
    out: list[dict] = []
    seen: set[int] = set()
    for m in re.findall(r"\[(\d+)\]", support):
        n = int(m)
        if n in seen or not (1 <= n <= len(chunks)):
            continue
        seen.add(n)
        c = chunks[n - 1]
        out.append({
            "id":      n,
            "source":  c["source_name"],
            "page":    c.get("page"),
            "section": c.get("section"),
            "url":     c.get("url"),
        })
    return out


def _fallback_email(issues: list[dict]) -> str:
    lines = [
        "Subject: Requested changes to the lease for [RENTAL_ADDRESS]",
        "",
        "Dear [LANDLORD_NAME],",
        "",
        "Thank you for sending the lease for [RENTAL_ADDRESS]. Before signing, "
        "I would like to discuss the following points:",
        "",
    ]
    if issues:
        for i, f in enumerate(issues, start=1):
            lines.append(f"{i}. {f.get('category', 'Lease term')}: {f.get('finding', '')}".rstrip())
    else:
        lines.append("1. Please confirm the lease terms are final and send a copy for my records.")
    lines += [
        "",
        "Could we discuss these by [RESPONSE_DATE]?",
        "",
        "Best regards,",
        "[TENANT_NAME]",
        "[TENANT_PHONE] | [TENANT_EMAIL]",
        "[DATE]",
    ]
    return "\n".join(lines)


# ── State ─────────────────────────────────────────────────────────────────────

class LeaseState(TypedDict, total=False):
    zip_code:   str
    lease_text: str
    status:     str
    message:    str
    lease_json: Dict[str, Any]
    categories: List[Dict[str, Any]]   # discovered at runtime
    risk_json:  Dict[str, Any]
    letter_text: str


# ── Graph ─────────────────────────────────────────────────────────────────────

def build_app():
    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL") or "gpt-4o-mini", temperature=0)
    law = LawRetriever(top_k=int(os.getenv("RAG_TOP_K", "6")))

    # ── Node 1: validate_zip ──────────────────────────────────────────────────
    def validate_zip(state: LeaseState) -> LeaseState:
        z = (state.get("zip_code") or "").strip()
        ok = z.isdigit() and len(z) == 5 and z.startswith(SUPPORTED_ZIP_PREFIXES)
        if not ok:
            return {
                **state,
                "status": "out_of_scope",
                "message": f"This tool currently supports {SUPPORTED_REGION} ZIP codes.",
            }
        return {**state, "status": "ok"}

    # ── Node 2: extract_structured ────────────────────────────────────────────
    def extract_structured(state: LeaseState) -> LeaseState:
        lease_text = state.get("lease_text", "")
        if len(lease_text) > LEASE_CHARS:
            log.warning("Lease text truncated for extraction: %d of %d chars used",
                        LEASE_CHARS, len(lease_text))
        chunks = _chunk_text(lease_text[:LEASE_CHARS])
        sys_msg = SystemMessage(content=(
            f"You are a lease data extractor for {SUPPORTED_REGION} rental agreements. "
            "Return ONLY a valid JSON object — no markdown, no explanation."
        ))

        def _extract_one(idx: int, chunk: str) -> dict | None:
            part_note = (
                f"This is part {idx + 1} of {len(chunks)} of the lease. "
                "Use null (or [] for arrays) for fields not present in this part.\n\n"
                if len(chunks) > 1 else ""
            )
            prompt = _extract_prompt(part_note, chunk)
            for attempt in range(2):
                try:
                    user_msg = HumanMessage(content=prompt if attempt == 0
                        else prompt + "\n\nReturn ONLY the JSON object starting with { and ending with }.")
                    parsed = _parse_json(llm.invoke([sys_msg, user_msg]).content)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    pass
            return None

        with ThreadPoolExecutor(max_workers=min(len(chunks), 4)) as pool:
            parts = list(pool.map(lambda a: _extract_one(*a), enumerate(chunks)))
        ok_parts = [p for p in parts if p is not None]
        if not ok_parts:
            return {**state, "status": "error",
                    "message": "Could not extract structured lease data after retrying."}
        if len(ok_parts) < len(parts):
            log.warning("Lease extraction failed for %d of %d chunks",
                        len(parts) - len(ok_parts), len(parts))
        return {**state, "lease_json": _merge_lease_json(ok_parts)}

    def _extract_prompt(part_note: str, chunk: str) -> str:
        return (
            part_note +
            "Extract key fields from this lease into JSON with exactly these keys:\n"
            "lease_type: one of [\"fixed\",\"month_to_month\",\"unknown\"]\n"
            "start_date: string or null\n"
            "end_date: string or null\n"
            "rent_amount: string or null\n"
            "deposit_amount: string or null\n"
            "notice_period_days: number or null\n"
            "late_fee_policy: string or null\n"
            "rent_increase_clause: string or null\n"
            "subletting_clause: string or null\n"
            "maintenance_responsibilities: string or null\n"
            "address_or_city_if_present: string or null\n"
            "other_key_terms: array of strings\n\n"
            f"LEASE TEXT:\n{chunk}"
        )

    # ── Node 3: discover_categories ───────────────────────────────────────────
    def discover_categories(state: LeaseState) -> LeaseState:
        """
        Ask the LLM to read the lease and decide which risk areas deserve analysis.
        Output is a list of {title, terms} dicts — fully dynamic, nothing predefined.
        """
        lease_text = state.get("lease_text", "")
        lease_json = state.get("lease_json", {})

        sys_msg = SystemMessage(content=(
            f"You are a tenant-protection expert for {SUPPORTED_REGION}. "
            "Your job is to identify every legally significant clause or risk area "
            "in a lease that a tenant should understand. "
            "Return ONLY valid JSON — no markdown."
        ))
        user_msg = HumanMessage(content=(
            "Read the lease summary and full text below. "
            f"Identify up to {MAX_CATEGORIES} distinct risk categories that exist in THIS lease. "
            "For each category produce:\n"
            "  title: short descriptive name (e.g. \"Automatic renewal\", \"Pest control liability\")\n"
            "  terms: array of 3-8 key words or phrases from the lease relevant to this category\n\n"
            "Return a JSON array of objects with keys 'title' and 'terms'. "
            "Only include categories that are actually present or relevant to this specific lease.\n\n"
            f"LEASE SUMMARY (structured):\n{json.dumps(lease_json, ensure_ascii=False)}\n\n"
            f"LEASE TEXT (first {LEASE_CHARS} chars):\n{lease_text[:LEASE_CHARS]}"
        ))
        try:
            resp = llm.invoke([sys_msg, user_msg]).content
            cats = _parse_json(resp)
            if not isinstance(cats, list) or not cats:
                raise ValueError("empty or invalid category list")
            # Normalise: keep only dicts with title + terms
            cats = [
                c for c in cats
                if isinstance(c, dict) and c.get("title") and isinstance(c.get("terms"), list)
            ][:MAX_CATEGORIES]
            return {**state, "categories": cats}
        except Exception:
            return {**state, "status": "error",
                    "message": "Could not discover risk categories from lease."}

    # ── Node 4: analyze_risk ──────────────────────────────────────────────────
    def analyze_risk(state: LeaseState) -> LeaseState:
        zip_code       = state.get("zip_code", "")
        lease_text     = state.get("lease_text", "")
        lease_json_str = json.dumps(state.get("lease_json", {}), ensure_ascii=False)
        categories     = state.get("categories") or []
        normalized_lease = _normalize_ws(lease_text)

        def _analyze_one(cat: dict) -> tuple[list, list]:
            title: str       = cat["title"]
            terms: list[str] = cat.get("terms") or []
            excerpt = _relevant_excerpt(lease_text, terms)
            q = (
                f"{SUPPORTED_REGION} tenant law regarding: {title}. "
                f"Key terms: {', '.join(terms)}. ZIP {zip_code}. "
                f"Lease excerpt: {excerpt}"
            )
            law_ctx, law_refs = law.context_with_chunks(
                q,
                top_k=int(os.getenv("RAG_TOP_K", "6")),
                max_chars=int(os.getenv("RAG_CONTEXT_MAX_CHARS", "5000")),
            )
            severities = list(SEVERITY_WEIGHTS.keys())
            sys_msg = SystemMessage(content=(
                f"You are a {SUPPORTED_REGION} lease risk analyzer. "
                "Informational assistance only — not legal advice. "
                "Return ONLY valid JSON, no markdown."
            ))
            user_msg = HumanMessage(content=(
                "Produce JSON with exactly:\n"
                "flags: array of {category, severity, finding, why_it_matters, support, lease_quote}\n"
                "recommendations: array of strings\n\n"
                f"severity must be one of: {severities}\n"
                "support must cite LAW_CONTEXT chunk ids like \"[1][3]\" or be \"\" if none applies.\n"
                "lease_quote must be the exact sentence or clause from LEASE_EXCERPT that the flag "
                f"is about, copied character-for-character (no paraphrasing, max {LEASE_QUOTE_MAX} "
                "chars), or \"\" if no single clause applies.\n"
                "If a claim lacks LAW_CONTEXT support, prefix why_it_matters with \"Uncertain:\" "
                f"and use the lowest severity.\n"
                "Never fabricate ordinance or statute citations.\n\n"
                f"CATEGORY: {title}\n\n"
                f"LEASE_JSON:\n{lease_json_str}\n\n"
                f"LEASE_EXCERPT:\n{excerpt}\n\n"
                f"LAW_CONTEXT:\n{law_ctx}"
            ))
            try:
                resp = llm.invoke([sys_msg, user_msg]).content
                part = _parse_json(resp)
                if not isinstance(part, dict):
                    return [], []
            except Exception:
                return [], []
            flags = [f for f in (part.get("flags") or []) if isinstance(f, dict)]
            for f in flags:
                f["lease_quote"] = _verify_quote(f.get("lease_quote"), normalized_lease)
                f["citations"] = _resolve_citations(f.get("support"), law_refs)
            recs  = [r.strip() for r in (part.get("recommendations") or [])
                     if isinstance(r, str) and r.strip()]
            return flags, recs

        all_flags: list[dict] = []
        all_recs:  list[str]  = []

        with ThreadPoolExecutor(max_workers=max(len(categories), 1)) as pool:
            futures = [pool.submit(_analyze_one, cat) for cat in categories]
            for f in futures:
                flags, recs = f.result()
                all_flags.extend(flags)
                all_recs.extend(recs)

        all_flags.sort(key=lambda f: SEVERITY_ORDER.get((f.get("severity") or "").strip(), 99))

        seen: set[str] = set()
        dedup_recs: list[str] = []
        for r in all_recs:
            key = r.lower()
            if key not in seen:
                seen.add(key)
                dedup_recs.append(r)

        score = _score(all_flags)
        return {
            **state,
            "risk_json": {
                "risk_score": score,
                "risk_label": _risk_label(score),
                "flags": all_flags,
                "recommendations": dedup_recs[:MAX_RECOMMENDATIONS],
            },
        }

    # ── Node 5: draft_letter ──────────────────────────────────────────────────
    def draft_letter(state: LeaseState) -> LeaseState:
        zip_code  = state.get("zip_code", "")
        lease     = state.get("lease_json", {})
        risk      = state.get("risk_json", {})

        # Only medium/high flags (already sorted most severe first)
        top_flags = [
            f for f in (risk.get("flags") or [])
            if (f.get("severity") or "") in SEVERITY_WEIGHTS
            and SEVERITY_WEIGHTS.get(f.get("severity", ""), 0) > min(SEVERITY_WEIGHTS.values())
        ][:LETTER_MAX_ISSUES]
        if not top_flags:
            return {**state, "letter_text": _fallback_email([])}

        def _issue_line(f: dict) -> str:
            line = f"- [{f.get('severity')}] {f.get('category','')}: {f.get('finding','')}"
            if f.get("lease_quote"):
                line += f"\n  Lease clause: \"{f['lease_quote']}\""
            sources = "; ".join(c["source"] for c in f.get("citations") or [])
            if sources:
                line += f"\n  Sources: {sources}"
            return line

        issues_summary = "\n".join(_issue_line(f) for f in top_flags)

        sys_msg = SystemMessage(content=(
            "Draft a concise, polite, professional email from a prospective or current tenant "
            "to their landlord, negotiating changes to the lease before signing. "
            "Start with a 'Subject:' line. For EACH issue provided, write one numbered, "
            "specific request (what to change, remove, or clarify in the lease). "
            "Do not add issues that are not provided. Only mention laws or sources listed "
            "under 'Sources'; never invent statutes. Do not threaten legal action. "
            "Use these placeholders: [DATE], [LANDLORD_NAME], [TENANT_NAME], [RENTAL_ADDRESS], "
            "[RESPONSE_DATE], [TENANT_PHONE], [TENANT_EMAIL]. "
            "Return plain text only — no markdown."
        ))
        user_msg = HumanMessage(content=(
            f"Region: {SUPPORTED_REGION}\n"
            f"ZIP: {zip_code}\n"
            f"Lease type: {lease.get('lease_type', 'unknown')}\n"
            f"Monthly rent: {lease.get('rent_amount', '?')}\n\n"
            f"Issues to raise as requests:\n{issues_summary}\n\n"
            "Write the negotiation email now."
        ))
        try:
            letter = llm.invoke([sys_msg, user_msg]).content.strip()
        except Exception:
            log.warning("Negotiation email generation failed; using fallback template")
            letter = ""
        return {**state, "letter_text": letter or _fallback_email(top_flags)}

    # ── Assemble graph ────────────────────────────────────────────────────────
    g = StateGraph(LeaseState)
    g.add_node("validate_zip",        validate_zip)
    g.add_node("extract_structured",  extract_structured)
    g.add_node("discover_categories", discover_categories)
    g.add_node("analyze_risk",        analyze_risk)
    g.add_node("draft_letter",        draft_letter)

    g.set_entry_point("validate_zip")

    def _route(stop_on_error: bool):
        def _fn(state: LeaseState) -> str:
            if state.get("status") in ("out_of_scope", "error"):
                return END
            return "ok"
        return _fn

    g.add_conditional_edges(
        "validate_zip",
        lambda s: END if s.get("status") in ("out_of_scope", "error") else "extract_structured",
    )
    g.add_conditional_edges(
        "extract_structured",
        lambda s: END if s.get("status") == "error" else "discover_categories",
    )
    g.add_conditional_edges(
        "discover_categories",
        lambda s: END if s.get("status") == "error" else "analyze_risk",
    )
    g.add_edge("analyze_risk", "draft_letter")
    g.add_edge("draft_letter", END)

    return g.compile()
