# agent.py
import os
import json
import logging
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from pathlib import Path
from typing import TypedDict, Dict, Any, List, Callable

from langchain_openai import ChatOpenAI
from langgraph.config import get_config, get_stream_writer
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
# e.g. SEVERITY_WEIGHTS="OK:0,Low:0.5,Medium:1.5,High:3"
def _parse_weights() -> dict[str, float]:
    raw = os.getenv("SEVERITY_WEIGHTS", "OK:0,Low:0.5,Medium:1.5,High:3")
    out: dict[str, float] = {}
    for pair in raw.split(","):
        k, _, v = pair.partition(":")
        if k.strip() and v.strip():
            out[k.strip()] = float(v.strip())
    out.setdefault("OK", 0.0)
    return out

SEVERITY_WEIGHTS = _parse_weights()
# Sort order: highest weight = first (most severe first)
SEVERITY_ORDER = {k: i for i, k in enumerate(
    sorted(SEVERITY_WEIGHTS, key=lambda k: -SEVERITY_WEIGHTS[k])
)}

SEVERITY_RUBRIC = (
    "Severity rubric:\n"
    "  OK: standard, lawful, market-typical. No action needed.\n"
    "  Low: minor or common, but worth knowing.\n"
    "  Medium: unusual, one-sided, or ambiguous; worth asking about or negotiating.\n"
    "  High: likely violates California or local law (must be supported by LAW_CONTEXT), "
    "or is severely one-sided or costly.\n"
    "A clause being present is not a risk. Use OK when a clause is standard and lawful. "
    "Only use Medium or High if you can state the specific problem in one sentence. "
    "If you cannot support a claim with LAW_CONTEXT, severity may not exceed Low.\n"
)

MARKET_NORMS_PATH = Path(os.getenv("MARKET_NORMS_PATH",
                                   str(Path(__file__).resolve().parent / "rag" / "market_norms.md")))
CALIBRATION_PASS = os.getenv("CALIBRATION_PASS", "false").strip().lower() in ("1", "true", "yes")
QUOTE_MATCH_RATIO = float(os.getenv("QUOTE_MATCH_RATIO", "0.9"))
CITATION_MIN_OVERLAP = int(os.getenv("CITATION_MIN_OVERLAP", "2"))
# FAISS L2 distance at or below which a chunk counts as semantically relevant. Unset = keywords only.
RAG_MAX_DISTANCE = float(os.getenv("RAG_MAX_DISTANCE")) if os.getenv("RAG_MAX_DISTANCE") else None

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
# Each additional flag (most severe first) counts RISK_DECAY times the previous one.
RISK_DECAY        = float(os.getenv("RISK_DECAY", "0.85"))
RISK_LABEL_MODERATE_AT = float(os.getenv("RISK_LABEL_MODERATE_AT", "3"))
RISK_LABEL_HIGH_AT     = float(os.getenv("RISK_LABEL_HIGH_AT", "6"))
LETTER_MAX_ISSUES = int(os.getenv("LETTER_MAX_ISSUES", "5"))

LLM_TIMEOUT_S     = float(os.getenv("LLM_TIMEOUT_S", "60"))
LLM_MAX_RETRIES   = int(os.getenv("LLM_MAX_RETRIES", "2"))
CATEGORY_WORKERS  = max(int(os.getenv("CATEGORY_WORKERS", "6")), 1)

_LEASE_JSON_LIST_KEYS = ("other_key_terms",)


# ── Job progress / errors (shared with the API and UI) ────────────────────────

# Graph nodes in execution order, then terminal pseudo-steps. `progress` is the
# percentage reached when that step completes; `label` is shown while it runs.
JOB_STEPS: dict[str, dict[str, Any]] = {
    "queued":              {"label": "Waiting to start",               "progress": 0},
    "validate_zip":        {"label": "Checking your ZIP",              "progress": 5},
    "extract_structured":  {"label": "Reading your lease",             "progress": 25},
    "discover_categories": {"label": "Finding risk areas",             "progress": 40},
    "analyze_risk":        {"label": "Checking California law",        "progress": 90},
    "draft_letter":        {"label": "Writing your negotiation email", "progress": 95},
    "done":                {"label": "Done",                           "progress": 100},
}
GRAPH_NODES = ("validate_zip", "extract_structured", "discover_categories", "analyze_risk", "draft_letter")

ERROR_MESSAGES: dict[str, str] = {
    "out_of_scope":      f"This tool currently supports {SUPPORTED_REGION} ZIP codes.",
    "scanned_pdf":       "This looks like a scanned PDF. Scanned leases aren't supported yet.",
    "extraction_failed": "We couldn't read this lease. Please check the PDF and try again.",
    "analysis_failed":   "Lease analysis failed. Please try again.",
    "timeout":           "Analysis took too long. Please try again.",
    "interrupted":       "The server restarted while your lease was being analyzed. Please try again.",
}


class PipelineError(Exception):
    """User-safe failure: `message` must never contain lease text or internals."""

    def __init__(self, code: str, message: str | None = None, http_status: int = 502):
        self.code = code
        self.message = message or ERROR_MESSAGES.get(code, ERROR_MESSAGES["analysis_failed"])
        self.http_status = http_status
        super().__init__(code)


ProgressFn = Callable[[str, int], None]


def run_graph(graph, state: dict, on_progress: ProgressFn | None = None,
              deadline: float | None = None) -> dict:
    """Stream the graph, reporting (step_label, progress) as nodes and categories complete.

    Raises PipelineError for out-of-scope ZIPs, node errors, and when `deadline`
    (time.monotonic()) passes. Returns the final state.
    """
    report = on_progress or (lambda label, pct: None)
    lo = JOB_STEPS["discover_categories"]["progress"]
    hi = JOB_STEPS["analyze_risk"]["progress"]

    def _check_deadline() -> None:
        if deadline is not None and time.monotonic() > deadline:
            raise PipelineError("timeout", http_status=504)

    final = dict(state)
    report(JOB_STEPS[GRAPH_NODES[0]]["label"], JOB_STEPS["queued"]["progress"])
    config = {"configurable": {"deadline": deadline}}
    for mode, chunk in graph.stream(state, config=config, stream_mode=["updates", "custom"]):
        _check_deadline()
        if mode == "custom":
            done, total = chunk.get("categories_done"), chunk.get("categories_total")
            if total:
                report(JOB_STEPS["analyze_risk"]["label"], lo + (hi - lo) * done // total)
            continue
        for node, update in chunk.items():
            if update:
                final.update(update)
            status = final.get("status")
            if status == "out_of_scope":
                raise PipelineError("out_of_scope", final.get("message"), http_status=200)
            if status == "error":
                code = "extraction_failed" if node == "extract_structured" else "analysis_failed"
                raise PipelineError(code, final.get("message"))
            if node in GRAPH_NODES:
                idx = GRAPH_NODES.index(node)
                running = GRAPH_NODES[min(idx + 1, len(GRAPH_NODES) - 1)]
                report(JOB_STEPS[running]["label"], JOB_STEPS[node]["progress"])
    _check_deadline()
    return final


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


def _canon_severity(sev: Any) -> str:
    """Map a model-provided severity onto a configured level, case-insensitively."""
    s = str(sev or "").strip()
    for k in SEVERITY_WEIGHTS:
        if k.lower() == s.lower():
            return k
    return s


def _weight(flag: dict) -> float:
    return SEVERITY_WEIGHTS.get(_canon_severity(flag.get("severity")), 0.0)


def _risk_label(score: float) -> str:
    if score >= RISK_LABEL_HIGH_AT:
        return "High"
    if score >= RISK_LABEL_MODERATE_AT:
        return "Moderate"
    return "Low"


def _just_below(threshold: float) -> float:
    """Largest one-decimal score strictly below `threshold`."""
    return math.ceil(threshold * 10 - 1e-9) / 10 - 0.1


def score_flags(flags: list[dict]) -> tuple[float, str, dict[str, int]]:
    """Deterministic 0-10 score, label and per-severity counts.

    Non-OK weights are taken most severe first, each discounted by RISK_DECAY**i,
    then squashed with 10 * (1 - exp(-total / RISK_SCORE_SCALE)). Hard rules on top:
    no Medium/High → Low; no High → not High; one High → at least Moderate;
    two or more High → High.
    """
    counts = {"high": 0, "medium": 0, "low": 0, "ok": 0}
    weights: list[float] = []
    for f in flags:
        sev = _canon_severity(f.get("severity")).lower()
        if sev in counts:
            counts[sev] += 1
        w = _weight(f)
        if sev != "ok" and w > 0:
            weights.append(w)
    weights.sort(reverse=True)
    total = sum(w * RISK_DECAY ** i for i, w in enumerate(weights))
    score = 10.0 * (1.0 - math.exp(-total / RISK_SCORE_SCALE)) if total > 0 and RISK_SCORE_SCALE > 0 else 0.0

    if counts["high"] == 0:
        score = min(score, _just_below(RISK_LABEL_HIGH_AT))
        if counts["medium"] == 0:
            score = min(score, _just_below(RISK_LABEL_MODERATE_AT))
    elif counts["high"] == 1:
        score = max(score, RISK_LABEL_MODERATE_AT)
    else:
        score = max(score, RISK_LABEL_HIGH_AT)
    score = round(min(max(score, 0.0), 10.0), 1)
    return score, _risk_label(score), counts


def _quote_key(quote: Any) -> str:
    return _normalize_ws(re.sub(r"[^a-z0-9]+", " ", str(quote or "").lower()))


def _same_clause(a: str, b: str) -> bool:
    """Normalized quotes refer to the same clause: equal, contained, or near-identical."""
    if not a or not b:
        return False
    if a == b:
        return True
    if min(len(a), len(b)) >= 20 and (a in b or b in a):
        return True
    return SequenceMatcher(None, a, b).ratio() >= QUOTE_MATCH_RATIO


def _merge_duplicate_flags(flags: list[dict]) -> list[dict]:
    """One flag per clause: highest severity wins; categories and citations are merged.

    Flags without a lease quote are only merged with exact (category, finding) duplicates.
    """
    groups: list[list[dict]] = []
    keys: list[str] = []
    for f in flags:
        qk = _quote_key(f.get("lease_quote"))
        key = qk or "\0" + _quote_key(f.get("category")) + "|" + _quote_key(f.get("finding"))
        for i, k in enumerate(keys):
            if (qk and _same_clause(qk, k)) or k == key:
                groups[i].append(f)
                break
        else:
            groups.append([f])
            keys.append(key)

    merged: list[dict] = []
    for group in groups:
        # Ties go to the most complete quote; max() keeps the first among equals.
        base = dict(max(group, key=lambda f: (_weight(f), len(f.get("lease_quote") or ""))))
        cats: list[str] = []
        cites: list[dict] = []
        seen_cites: set[tuple] = set()
        for f in group:
            for c in str(f.get("category") or "").split(" / "):
                if c.strip() and c.strip().lower() not in (x.lower() for x in cats):
                    cats.append(c.strip())
            for c in f.get("citations") or []:
                ck = (c.get("source"), c.get("page"), c.get("section"), c.get("url"))
                if ck not in seen_cites:
                    seen_cites.add(ck)
                    cites.append(c)
        base["category"] = " / ".join(cats) or base.get("category", "")
        base["citations"] = cites
        merged.append(base)
    return merged


_STOPWORDS = frozenset("""
a an and are as at be been being by can could do does for from had has have if in into is it its
may might must no nor not of on or our shall should so such than that the their them then there these
they this those to under upon was were what when where which who will with within without would you your
any all each other more less only also per day days
tenant tenants landlord landlords lease leases leased premises agreement rent rental rents unit units
property section law laws applicable california san francisco city county
""".split())


def _content_tokens(text: str) -> set[str]:
    out: set[str] = set()
    for t in re.findall(r"[a-z0-9]+", (text or "").lower()):
        if len(t) < 3 or t in _STOPWORDS:
            continue
        out.add(t[:-1] if len(t) > 4 and t.endswith("s") else t)
    return out


def _citation_relevant(clause_text: str, ref: dict) -> bool:
    """A retrieved chunk is relevant if it shares enough content words with the clause,
    or (when RAG_MAX_DISTANCE is set) it is semantically close enough."""
    if RAG_MAX_DISTANCE is not None and ref.get("score") is not None and ref["score"] <= RAG_MAX_DISTANCE:
        return True
    shared = _content_tokens(clause_text) & _content_tokens(ref.get("text") or "")
    return len(shared) >= max(CITATION_MIN_OVERLAP, 1)


def _filter_citations(flag: dict, refs: list[dict]) -> list[dict]:
    """Resolve the flag's "[n]" ids and keep only chunks relevant to its clause."""
    clause = " ".join(str(flag.get(k) or "") for k in ("lease_quote", "category", "finding"))
    return [c for c in _resolve_citations(flag.get("support"), refs)
            if _citation_relevant(clause, refs[c["id"] - 1])]


def _cap_unsupported(flag: dict) -> dict:
    """Severity above Low requires at least one relevant citation and no self-declared
    "Uncertain:"; otherwise cap at Low."""
    low = SEVERITY_WEIGHTS.get("Low", 0.0)
    why = str(flag.get("why_it_matters") or "")
    if _weight(flag) > low and (not flag.get("citations") or why.startswith("Uncertain:")):
        flag["severity"] = "Low"
        if not why.startswith("Uncertain:"):
            flag["why_it_matters"] = f"Uncertain: {why}".strip()
    return flag


def _load_market_norms(path: Path, region: str) -> str:
    """Return the `## <region>` section of the market-norms file ("" if missing)."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        log.warning("Market norms file not found; analysis runs without MARKET_CONTEXT")
        return ""
    for section in re.split(r"(?m)^## ", text)[1:]:
        head, _, body = section.partition("\n")
        if head.strip().lower() == region.strip().lower():
            return body.strip()
    log.warning("No market norms section for the configured region")
    return ""


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


def _fallback_questions_email() -> str:
    return "\n".join([
        "Subject: A few questions about the lease for [RENTAL_ADDRESS]",
        "",
        "Dear [LANDLORD_NAME],",
        "",
        "Thank you for sending the lease for [RENTAL_ADDRESS]. It looks clear to me, "
        "and I have a few quick questions before signing:",
        "",
        "1. Which utilities and services are included in the rent, and which should I set up myself?",
        "2. How should I submit maintenance or repair requests, and who is the contact for emergencies?",
        "3. Could you confirm the move-in date and how keys will be handed over?",
        "",
        "Could you let me know by [RESPONSE_DATE]?",
        "",
        "Best regards,",
        "[TENANT_NAME]",
        "[TENANT_PHONE] | [TENANT_EMAIL]",
        "[DATE]",
    ])


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
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL") or "gpt-4o-mini",
        temperature=0,
        timeout=LLM_TIMEOUT_S,
        max_retries=LLM_MAX_RETRIES,
    )
    law = LawRetriever(top_k=int(os.getenv("RAG_TOP_K", "6")))
    market_ctx = _load_market_norms(MARKET_NORMS_PATH, SUPPORTED_REGION)

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
            "Your job is to list the topics in a lease that should be checked against "
            "law and market norms. A topic being present does not mean it is a risk. "
            "Return ONLY valid JSON — no markdown."
        ))
        user_msg = HumanMessage(content=(
            "Read the lease summary and full text below. "
            f"Identify up to {MAX_CATEGORIES} distinct topics to check in THIS lease "
            "(standard and unusual alike). Cover every section of the lease, including the last "
            "ones. If there are more sections than the limit, combine minor ones into one topic "
            "(e.g. \"Disclosures, governing law and other terms\") so nothing is left out; the "
            "final topic must cover any remaining sections at the end of the lease. Each "
            "clause belongs to exactly one topic; do not create overlapping topics for the same clause. "
            "For each topic produce:\n"
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

        def _analyze_one(cat: dict) -> tuple[list, list] | None:
            """Returns None when this category couldn't be analyzed."""
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
                + SEVERITY_RUBRIC +
                "For non-OK flags, finding must name the concrete problem in one sentence. "
                "For OK flags, finding is a neutral one-line description of the clause and "
                "why_it_matters may be \"\".\n"
                "Report one flag per lease clause (a section or paragraph, not each sentence), "
                "at most once, and only clauses about CATEGORY. finding must mention every key "
                "term of the clause (amounts, deadlines, notice periods).\n"
                "MARKET_CONTEXT describes what is typical; use it to judge whether a clause is "
                "standard. LAW_CONTEXT is the only source for legal citations.\n"
                "support must cite LAW_CONTEXT chunk ids like \"[1][3]\" that directly address "
                "this clause, or be \"\" if none applies.\n"
                "lease_quote must be the exact text of the clause from LEASE_EXCERPT that the flag "
                "is about (all of its sentences, not just the first), copied character-for-character "
                f"(no paraphrasing, max {LEASE_QUOTE_MAX} chars), or \"\" if no single clause applies.\n"
                "If a claim lacks LAW_CONTEXT support, prefix why_it_matters with \"Uncertain:\".\n"
                "Recommendations only for non-OK flags.\n"
                "Never fabricate ordinance or statute citations.\n\n"
                f"CATEGORY: {title}\n\n"
                f"LEASE_JSON:\n{lease_json_str}\n\n"
                f"LEASE_EXCERPT:\n{excerpt}\n\n"
                f"MARKET_CONTEXT:\n{market_ctx or '(none)'}\n\n"
                f"LAW_CONTEXT:\n{law_ctx}"
            ))
            try:
                resp = llm.invoke([sys_msg, user_msg]).content
                part = _parse_json(resp)
                if not isinstance(part, dict):
                    return None
            except Exception:
                return None
            flags = [f for f in (part.get("flags") or []) if isinstance(f, dict)]
            for f in flags:
                f["severity"] = _canon_severity(f.get("severity"))
                f["lease_quote"] = _verify_quote(f.get("lease_quote"), normalized_lease)
                f["citations"] = _filter_citations(f, law_refs)
            recs  = [r.strip() for r in (part.get("recommendations") or [])
                     if isinstance(r, str) and r.strip()]
            return flags, recs

        all_flags: list[dict] = []
        all_recs:  list[str]  = []
        skipped = 0

        try:
            write_progress = get_stream_writer()
            deadline = (get_config().get("configurable") or {}).get("deadline")
        except Exception:
            write_progress, deadline = (lambda _chunk: None), None

        # Collect in category order so output is deterministic regardless of completion order.
        results: list[tuple[list, list] | None] = [None] * len(categories)
        workers = max(min(len(categories), CATEGORY_WORKERS), 1)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_analyze_one, cat): i for i, cat in enumerate(categories)}
            for done, fut in enumerate(as_completed(futures), start=1):
                try:
                    results[futures[fut]] = fut.result()
                except Exception as e:
                    log.warning("Category analysis raised: %s", type(e).__name__)
                write_progress({"categories_done": done, "categories_total": len(categories)})
                if deadline is not None and time.monotonic() > deadline:
                    pool.shutdown(wait=False, cancel_futures=True)
                    return {**state, "status": "error", "message": ERROR_MESSAGES["timeout"]}
        for res in results:
            if res is None:
                skipped += 1
                continue
            flags, recs = res
            all_flags.extend(flags)
            all_recs.extend(recs)

        if skipped:
            log.warning("Skipped %d of %d risk categories", skipped, len(categories))
        if categories and skipped == len(categories):
            return {**state, "status": "error", "message": "Could not analyze any risk areas in this lease."}

        all_flags = _merge_duplicate_flags(all_flags)
        if CALIBRATION_PASS:
            _calibrate(all_flags)
        all_flags = [_cap_unsupported(f) for f in all_flags]
        all_flags.sort(key=lambda f: SEVERITY_ORDER.get(f.get("severity") or "", 99))

        seen: set[str] = set()
        dedup_recs: list[str] = []
        for r in all_recs:
            key = r.lower()
            if key not in seen:
                seen.add(key)
                dedup_recs.append(r)

        score, label, counts = score_flags(all_flags)
        ok_flags = [f for f in all_flags if f.get("severity") == "OK"]
        return {
            **state,
            "risk_json": {
                "risk_score": score,
                "risk_label": label,
                "flags": [f for f in all_flags if f.get("severity") != "OK"],
                "recommendations": dedup_recs[:MAX_RECOMMENDATIONS],
                "skipped_categories": skipped,
                "counts": counts,
                "standard_clauses": [
                    {"category": f.get("category", ""), "finding": f.get("finding", ""),
                     "lease_quote": f.get("lease_quote", "")}
                    for f in ok_flags
                ],
            },
        }

    def _calibrate(flags: list[dict]) -> None:
        """Optional second opinion on severities. Mutates flags; leaves them unchanged on failure."""
        if not flags:
            return
        items = [{"id": i, "category": f.get("category", ""), "finding": f.get("finding", ""),
                  "lease_quote": f.get("lease_quote", ""), "severity": f.get("severity", ""),
                  "has_law_citation": bool(f.get("citations"))}
                 for i, f in enumerate(flags)]
        sys_msg = SystemMessage(content=(
            f"You are a {SUPPORTED_REGION} lease severity calibrator. "
            "Return ONLY valid JSON, no markdown."
        ))
        user_msg = HumanMessage(content=(
            "Review these lease flags and correct any severity that does not follow the rubric.\n"
            + SEVERITY_RUBRIC +
            "A flag with has_law_citation=false may not exceed Low.\n"
            "Return JSON: {\"adjustments\": [{\"id\": int, \"severity\": one of "
            f"{list(SEVERITY_WEIGHTS)}, \"reason\": one sentence}}]}}. "
            "Include only flags whose severity should change.\n\n"
            f"MARKET_CONTEXT:\n{market_ctx or '(none)'}\n\n"
            f"FLAGS:\n{json.dumps(items, ensure_ascii=False)}"
        ))
        try:
            adjustments = _parse_json(llm.invoke([sys_msg, user_msg]).content).get("adjustments") or []
        except Exception as e:
            log.warning("Calibration pass failed: %s", type(e).__name__)
            return
        changed = 0
        for a in adjustments:
            if not isinstance(a, dict) or not isinstance(a.get("id"), int):
                continue
            sev = _canon_severity(a.get("severity"))
            if not (0 <= a["id"] < len(flags)) or sev not in SEVERITY_WEIGHTS:
                continue
            f = flags[a["id"]]
            if sev != f.get("severity"):
                f["severity"] = sev
                f["calibration_reason"] = str(a.get("reason") or "")[:300]
                changed += 1
        log.info("Calibration adjusted %d of %d flags", changed, len(flags))

    # ── Node 5: draft_letter ──────────────────────────────────────────────────
    def draft_letter(state: LeaseState) -> LeaseState:
        zip_code  = state.get("zip_code", "")
        lease     = state.get("lease_json", {})
        risk      = state.get("risk_json", {})

        # Only medium/high flags (already sorted most severe first)
        medium = SEVERITY_WEIGHTS.get("Medium", 0.0)
        top_flags = [f for f in (risk.get("flags") or []) if _weight(f) >= medium > 0][:LETTER_MAX_ISSUES]
        if not top_flags:
            return {**state, "letter_text": _questions_email(lease, risk)}

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

    def _questions_email(lease: dict, risk: dict) -> str:
        """No negotiable issues: a short, polite email with a few clarifying questions."""
        topics = [f"- {f.get('category', '')}: {f.get('finding', '')}"
                  for f in (risk.get("flags") or []) + (risk.get("standard_clauses") or [])][:12]
        sys_msg = SystemMessage(content=(
            "Draft a short, polite email from a prospective or current tenant to their landlord "
            "before signing a lease that looks standard. Start with a 'Subject:' line. Ask two or "
            "three specific clarifying questions about the lease; make no demands and do not ask "
            "for changes. Do not cite laws. "
            "Use these placeholders: [DATE], [LANDLORD_NAME], [TENANT_NAME], [RENTAL_ADDRESS], "
            "[RESPONSE_DATE], [TENANT_PHONE], [TENANT_EMAIL]. "
            "Return plain text only — no markdown."
        ))
        user_msg = HumanMessage(content=(
            f"Region: {SUPPORTED_REGION}\n"
            f"Lease type: {lease.get('lease_type', 'unknown')}\n"
            f"Lease topics:\n" + ("\n".join(topics) or "- (none)") + "\n\n"
            "Write the email now."
        ))
        try:
            letter = llm.invoke([sys_msg, user_msg]).content.strip()
        except Exception:
            log.warning("Questions email generation failed; using fallback template")
            letter = ""
        return letter or _fallback_questions_email()

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
    g.add_conditional_edges(
        "analyze_risk",
        lambda s: END if s.get("status") == "error" else "draft_letter",
    )
    g.add_edge("draft_letter", END)

    return g.compile()
