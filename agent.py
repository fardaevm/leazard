# agent.py
import os
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TypedDict, Dict, Any, List

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage

from rag.retriever import LawRetriever

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
LEASE_CHARS       = int(os.getenv("LEASE_EXTRACT_CHARS", "12000"))
EXCERPT_WINDOW    = int(os.getenv("EXCERPT_WINDOW_CHARS", "900"))
EXCERPT_MAX       = int(os.getenv("EXCERPT_MAX_CHARS", "1400"))
MAX_RECOMMENDATIONS = int(os.getenv("MAX_RECOMMENDATIONS", "20"))


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
        sys_msg = SystemMessage(content=(
            f"You are a lease data extractor for {SUPPORTED_REGION} rental agreements. "
            "Return ONLY a valid JSON object — no markdown, no explanation."
        ))
        prompt = (
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
            f"LEASE TEXT:\n{lease_text[:LEASE_CHARS]}"
        )
        for attempt in range(2):
            try:
                user_msg = HumanMessage(content=prompt if attempt == 0
                    else prompt + "\n\nReturn ONLY the JSON object starting with { and ending with }.")
                resp = llm.invoke([sys_msg, user_msg]).content
                return {**state, "lease_json": _parse_json(resp)}
            except Exception:
                if attempt == 1:
                    return {**state, "status": "error",
                            "message": "Could not extract structured lease data after retrying."}
        return state  # unreachable

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

        def _score(flags: list[dict]) -> float:
            total = 0.0
            for f in flags:
                sev = (f.get("severity") or "").strip()
                total += SEVERITY_WEIGHTS.get(sev, 0.0)
                if sev in SEVERITY_WEIGHTS and sev != min(SEVERITY_WEIGHTS, key=SEVERITY_WEIGHTS.get):
                    if not (f.get("support") or "").strip():
                        total += 1.0
            return min(10.0, round(total / 2.0, 1))

        def _analyze_one(cat: dict) -> tuple[list, list]:
            title: str       = cat["title"]
            terms: list[str] = cat.get("terms") or []
            excerpt = _relevant_excerpt(lease_text, terms)
            q = (
                f"{SUPPORTED_REGION} tenant law regarding: {title}. "
                f"Key terms: {', '.join(terms)}. ZIP {zip_code}. "
                f"Lease excerpt: {excerpt}"
            )
            law_ctx = law.context(
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
                "flags: array of {category, severity, finding, why_it_matters, support}\n"
                "recommendations: array of strings\n\n"
                f"severity must be one of: {severities}\n"
                "support must cite LAW_CONTEXT chunk ids like \"[1][3]\" or be \"\" if none applies.\n"
                "If a claim lacks LAW_CONTEXT support, prefix why_it_matters with \"Uncertain:\" "
                f"and use the lowest severity.\n"
                "Never fabricate ordinance or statute citations.\n\n"
                f"CATEGORY: {title}\n\n"
                f"LEASE_JSON:\n{lease_json_str}\n\n"
                f"LAW_CONTEXT:\n{law_ctx}"
            ))
            try:
                resp = llm.invoke([sys_msg, user_msg]).content
                part = _parse_json(resp)
            except Exception:
                return [], []
            flags = [f for f in (part.get("flags") or []) if isinstance(f, dict)]
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

        return {
            **state,
            "risk_json": {
                "risk_score": _score(all_flags),
                "flags": all_flags,
                "recommendations": dedup_recs[:MAX_RECOMMENDATIONS],
            },
        }

    # ── Node 5: draft_letter ──────────────────────────────────────────────────
    def draft_letter(state: LeaseState) -> LeaseState:
        zip_code  = state.get("zip_code", "")
        lease     = state.get("lease_json", {})
        risk      = state.get("risk_json", {})

        # Only pass high/medium flags so the prompt stays focused
        top_flags = [
            f for f in (risk.get("flags") or [])
            if (f.get("severity") or "") in SEVERITY_WEIGHTS
            and SEVERITY_WEIGHTS.get(f.get("severity", ""), 0) > min(SEVERITY_WEIGHTS.values())
        ]
        issues_summary = "\n".join(
            f"- [{f['severity']}] {f.get('category','')}: {f.get('finding','')}"
            for f in top_flags[:6]
        ) or "No major issues flagged."

        sys_msg = SystemMessage(content=(
            "Draft a concise, professional tenant move-out notice letter template. "
            "Use these placeholders: [DATE], [LANDLORD_NAME], [LANDLORD_ADDRESS], "
            "[TENANT_NAME], [RENTAL_ADDRESS], [MOVE_OUT_DATE]. "
            "Return plain text only — no markdown."
        ))
        user_msg = HumanMessage(content=(
            f"Region: {SUPPORTED_REGION}\n"
            f"ZIP: {zip_code}\n"
            f"Lease type: {lease.get('lease_type', 'unknown')}\n"
            f"Notice period: {lease.get('notice_period_days', '?')} days\n"
            f"Monthly rent: {lease.get('rent_amount', '?')}\n\n"
            f"Key risk issues to reference:\n{issues_summary}\n\n"
            "Write the move-out notice template now."
        ))
        letter = llm.invoke([sys_msg, user_msg]).content.strip()
        return {**state, "letter_text": letter}

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
