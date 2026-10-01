import json
import logging
import re
import threading
import time
from types import SimpleNamespace

import pytest

import agent
from rag.retriever import LawRetriever, RetrievedChunk, display_source_name


# ── Score normalization ───────────────────────────────────────────────────────

def _flags(*sevs):
    return [{"severity": s} for s in sevs]


def _score(flags):
    return agent.score_flags(flags)[0]


def _label(flags):
    return agent.score_flags(flags)[1]


def test_default_weights_include_ok():
    assert agent.SEVERITY_WEIGHTS == {"OK": 0.0, "Low": 0.5, "Medium": 1.5, "High": 3.0}


@pytest.mark.parametrize("sevs,score,label", [
    ([], 0.0, "Low"),
    (["OK"] * 12, 0.0, "Low"),
    (["Bogus", ""], 0.0, "Low"),
])
def test_score_zero_cases(sevs, score, label):
    assert agent.score_flags(_flags(*sevs))[:2] == (score, label)


def test_score_required_cases():
    assert _label(_flags(*["Low"] * 12)) != "High"
    assert _label(_flags(*["Medium"] * 4)) == "Moderate"
    assert _label(_flags("High")) in ("Moderate", "High")
    assert _label(_flags("High", "High")) == "High"


def test_score_hard_rules_cap_and_floor():
    assert _score(_flags(*["Low"] * 200)) < agent.RISK_LABEL_MODERATE_AT
    assert _label(_flags(*["Medium"] * 200)) == "Moderate"
    assert _label(_flags("High", *["OK"] * 20)) == "Moderate"
    assert _label(_flags("High", "High", *["OK"] * 20)) == "High"
    assert _score(_flags(*["High"] * 100)) <= 10.0


def test_score_ok_flags_do_not_contribute_and_case_is_ignored():
    assert _score(_flags("Medium", "OK", "ok")) == _score(_flags("medium"))


def test_score_is_monotonic_with_diminishing_returns():
    assert 0 < _score(_flags("Low")) < _score(_flags("Medium")) < _score(_flags("High"))
    s = [_score(_flags(*["Medium"] * n)) for n in range(1, 5)]
    assert s == sorted(s)
    assert s[1] - s[0] > s[3] - s[2]


def test_score_is_deterministic_and_order_independent():
    a = _flags("High", "Low", "Medium", "Medium")
    assert agent.score_flags(a) == agent.score_flags(list(reversed(a))) == agent.score_flags(a)


def test_score_counts():
    assert agent.score_flags(_flags("High", "medium", "Low", "OK", "OK", "?"))[2] == \
        {"high": 1, "medium": 1, "low": 1, "ok": 2}


@pytest.mark.parametrize("score,label", [(0, "Low"), (2.9, "Low"), (3.0, "Moderate"),
                                         (5.9, "Moderate"), (6.0, "High"), (10, "High")])
def test_risk_label(score, label):
    assert agent._risk_label(score) == label


# ── Duplicate merging ─────────────────────────────────────────────────────────

def _flag(cat, sev, quote, cites=(), finding="f"):
    return {"category": cat, "severity": sev, "finding": finding, "why_it_matters": "",
            "lease_quote": quote, "citations": [{"id": i, "source": s, "page": 1, "section": None,
                                                 "url": None} for i, s in cites]}


DEPOSIT_Q = "Tenant shall pay a security deposit of $3,450.00 upon signing."


def test_merge_keeps_highest_severity_and_unions_categories_and_citations():
    merged = agent._merge_duplicate_flags([
        _flag("Security deposit", "Low", DEPOSIT_Q, [(1, "a")]),
        _flag("Move-in costs", "Medium", DEPOSIT_Q.lower() + "  ", [(1, "b"), (2, "a")], finding="m"),
        _flag("Deposit return", "OK", "Tenant shall pay a security deposit of $3,450 upon signing"),
        _flag("Late fees", "Low", "Late fee of $75.00."),
    ])
    assert len(merged) == 2
    m = merged[0]
    assert m["severity"] == "Medium" and m["finding"] == "m"
    assert m["category"] == "Security deposit / Move-in costs / Deposit return"
    assert [c["source"] for c in m["citations"]] == ["a", "b"]
    assert merged[1]["category"] == "Late fees"
    assert agent.score_flags(merged)[2]["medium"] == 1


def test_merge_contained_quote_counts_once():
    long_q = "The deposit will be held and returned within 21 days after Tenant vacates."
    merged = agent._merge_duplicate_flags([
        _flag("Deposit", "OK", long_q), _flag("Deposit return", "Low", "returned within 21 days after Tenant vacates"),
    ])
    assert len(merged) == 1 and merged[0]["severity"] == "Low"


def test_merge_does_not_join_different_or_unquoted_clauses():
    merged = agent._merge_duplicate_flags([
        _flag("Pets", "OK", "No pets are permitted without prior written consent."),
        _flag("Smoking", "OK", "Smoking and vaping of any substance are prohibited."),
        _flag("Disclosures", "Low", "", finding="Missing A"),
        _flag("Disclosures", "Low", "", finding="Missing B"),
        _flag("Disclosures", "Low", "", finding="Missing A"),
    ])
    assert [f["category"] for f in merged] == ["Pets", "Smoking", "Disclosures", "Disclosures"]


# ── Citation relevance and severity cap ───────────────────────────────────────

ENTRY_REFS = [
    {"source_name": "sf ordinance", "page": 3, "section": None, "url": None,
     "text": "A landlord may enter the unit only after giving written notice at least 24 hours "
             "in advance, except in an emergency, for repairs or inspection."},
    {"source_name": "sf rent board rules", "page": 9, "section": None, "url": None,
     "text": "Capital improvement passthrough petitions must be filed with the Board."},
    {"source_name": "sf ordinance", "page": 4, "section": None, "url": None},
]


def test_citations_without_overlap_are_dropped():
    f = {"category": "Entry by landlord", "finding": "24-hour written notice is standard",
         "lease_quote": "Landlord may enter the Premises upon at least 24 hours' written notice "
                        "for inspection, repairs, or showing, except in an emergency.",
         "support": "[1][2][3][9]"}
    assert [c["id"] for c in agent._filter_citations(f, ENTRY_REFS)] == [1]


def test_citation_kept_by_semantic_distance_when_enabled(monkeypatch):
    f = {"category": "Rent", "finding": "x", "lease_quote": "", "support": "[2]"}
    refs = [dict(ENTRY_REFS[0]), dict(ENTRY_REFS[1], score=0.3)]
    assert agent._filter_citations(f, refs) == []
    monkeypatch.setattr(agent, "RAG_MAX_DISTANCE", 0.5)
    assert [c["id"] for c in agent._filter_citations(f, refs)] == [2]


def test_unsupported_claims_are_capped_at_low():
    high = agent._cap_unsupported({"severity": "High", "why_it_matters": "Illegal.", "citations": []})
    assert high["severity"] == "Low" and high["why_it_matters"] == "Uncertain: Illegal."
    med = agent._cap_unsupported({"severity": "Medium", "why_it_matters": "Uncertain: x", "citations": []})
    assert med["why_it_matters"] == "Uncertain: x"
    kept = agent._cap_unsupported({"severity": "High", "why_it_matters": "y", "citations": [{"id": 1}]})
    assert kept["severity"] == "High" and kept["why_it_matters"] == "y"
    for sev in ("OK", "Low"):
        f = agent._cap_unsupported({"severity": sev, "why_it_matters": "", "citations": []})
        assert f["severity"] == sev and f["why_it_matters"] == ""


def test_self_declared_uncertain_claim_is_capped_even_with_citation():
    f = agent._cap_unsupported({"severity": "Medium", "why_it_matters": "Uncertain: maybe high.",
                                "citations": [{"id": 1}]})
    assert f["severity"] == "Low" and f["why_it_matters"] == "Uncertain: maybe high."


def test_merge_tie_keeps_most_complete_quote():
    short = "Tenant shall pay a security deposit of $3,450.00 upon signing."
    full = short + " The deposit will be returned within 21 days after Tenant vacates."
    [m] = agent._merge_duplicate_flags([_flag("Rent", "OK", short, finding="rent"),
                                        _flag("Security deposit", "OK", full, finding="deposit")])
    assert m["lease_quote"] == full and m["finding"] == "deposit"
    assert m["category"] == "Rent / Security deposit"


# ── Market norms ──────────────────────────────────────────────────────────────

def test_market_norms_section_selected_by_region(tmp_path):
    p = tmp_path / "norms.md"
    p.write_text("# Title\nintro\n## Region A\n- a norm\n## Region B\n- b norm\n")
    assert agent._load_market_norms(p, "region b") == "- b norm"
    assert agent._load_market_norms(p, "Region C") == ""
    assert agent._load_market_norms(tmp_path / "missing.md", "Region A") == ""


def test_seeded_market_norms_cover_default_region():
    ctx = agent._load_market_norms(agent.MARKET_NORMS_PATH, agent.SUPPORTED_REGION)
    assert "verify against current law" in ctx.lower()
    assert "21 days" in ctx and "24 hours" in ctx and "AB 12" in ctx


# ── Quote substring check ─────────────────────────────────────────────────────

LEASE = "Section 4.  Tenant shall pay a\nlate fee of $200 if rent is late.\n\nSection 5. No pets."
NORM = agent._normalize_ws(LEASE)


def test_quote_kept_when_verbatim_modulo_whitespace():
    assert agent._verify_quote("Tenant shall pay a late   fee of $200", NORM) == \
        "Tenant shall pay a late fee of $200"
    assert agent._verify_quote("  No pets. ", NORM) == "No pets."


def test_quote_blanked_when_paraphrased_or_invalid():
    assert agent._verify_quote("Tenant must pay a late fee of $200", NORM) == ""
    assert agent._verify_quote("no pets.", NORM) == ""  # case-sensitive
    assert agent._verify_quote(None, NORM) == ""
    assert agent._verify_quote(123, NORM) == ""
    assert agent._verify_quote("   ", NORM) == ""


def test_quote_capped_at_max_length():
    long_lease = "word " * 200
    q = agent._verify_quote(long_lease, agent._normalize_ws(long_lease))
    assert 0 < len(q) <= agent.LEASE_QUOTE_MAX


# ── Chunking / truncation ─────────────────────────────────────────────────────

def test_chunk_text_short_text_single_chunk():
    assert agent._chunk_text("abc", size=10, overlap=2) == ["abc"]


def test_chunk_text_covers_everything_with_overlap():
    text = "".join(chr(65 + i % 26) for i in range(1000))
    chunks = agent._chunk_text(text, size=300, overlap=50)
    assert all(len(c) <= 300 for c in chunks)
    assert chunks[0] == text[:300]
    assert chunks[-1] == text[-len(chunks[-1]):]
    for a, b in zip(chunks, chunks[1:]):
        assert a[-50:] == b[:50]
    rebuilt = chunks[0] + "".join(c[50:] for c in chunks[1:])
    assert rebuilt == text


def test_merge_lease_json_first_value_wins_and_lists_union():
    merged = agent._merge_lease_json([
        {"lease_type": "unknown", "rent_amount": None, "deposit_amount": "$6000", "other_key_terms": ["a"]},
        {"lease_type": "fixed", "rent_amount": "$3000", "deposit_amount": "$1", "other_key_terms": ["a", "b"]},
        {"lease_type": "month_to_month", "rent_amount": "$9", "late_fee_policy": None},
    ])
    assert merged == {
        "lease_type": "fixed",
        "rent_amount": "$3000",
        "deposit_amount": "$6000",
        "other_key_terms": ["a", "b"],
        "late_fee_policy": None,
    }


# ── Citation resolution ───────────────────────────────────────────────────────

REFS = [
    {"chunk_id": 10, "source": "ca_civil_code_1950.5.pdf", "source_name": "ca civil code 1950.5",
     "page": 2, "section": None, "url": None},
    {"chunk_id": 11, "source": "sf_rent_ordinance.pdf", "source_name": "sf rent ordinance",
     "page": 7, "section": "37.9", "url": "https://example.org/37.9"},
]


def test_resolve_citations_maps_and_drops_unknown_ids():
    cites = agent._resolve_citations("[2][1][9][0][2]", REFS)
    assert [c["id"] for c in cites] == [2, 1]
    assert cites[0] == {"id": 2, "source": "sf rent ordinance", "page": 7,
                        "section": "37.9", "url": "https://example.org/37.9"}
    assert agent._resolve_citations("", REFS) == []
    assert agent._resolve_citations(None, REFS) == []
    assert agent._resolve_citations("[1]", []) == []


def test_display_source_name():
    assert display_source_name("ca_civil-code_1950.5.pdf") == "ca civil code 1950.5"
    assert display_source_name("") == "unknown"


def test_context_with_chunks_only_refs_labels_that_fit():
    r = object.__new__(LawRetriever)
    r.top_k = 3
    r.search = lambda q, top_k=None: [
        RetrievedChunk(chunk_id=i, source=f"law_{i}.pdf", page=i, score=0.1, text="x" * 100)
        for i in range(3)
    ]
    full, refs = r.context_with_chunks("q", max_chars=100_000)
    assert [x["chunk_id"] for x in refs] == [0, 1, 2]
    assert r.context("q", max_chars=100_000) == full

    second_label = full.index("[2]")
    ctx, refs = r.context_with_chunks("q", max_chars=second_label + 3)
    assert [x["chunk_id"] for x in refs] == [0]
    assert refs[0]["source_name"] == "law 0"


# ── Full graph with mocked LLM + retriever ────────────────────────────────────

class FakeLLM:
    def __init__(self):
        self.lock = threading.Lock()
        self.extract_prompts: list[str] = []
        self.analyze_prompts: list[str] = []
        self.email_prompts: list[str] = []

    def invoke(self, messages):
        sys_txt, user_txt = messages[0].content, messages[1].content
        if "lease data extractor" in sys_txt:
            with self.lock:
                self.extract_prompts.append(user_txt)
            m = re.search(r"part (\d+) of", user_txt)
            part = int(m.group(1)) if m else 1
            data = {"lease_type": "fixed" if part == 2 else "unknown",
                    "rent_amount": "$3000" if part == 1 else None,
                    "deposit_amount": "$6000" if part == 2 else None,
                    "other_key_terms": [f"term-{part}"]}
            return SimpleNamespace(content=json.dumps(data))
        if "tenant-protection expert" in sys_txt:
            return SimpleNamespace(content=json.dumps([{"title": "Deposit", "terms": ["deposit"]}]))
        if "lease risk analyzer" in sys_txt:
            with self.lock:
                self.analyze_prompts.append(user_txt)
            return SimpleNamespace(content="```json\n" + json.dumps({
                "flags": [
                    {"category": "Deposit", "severity": "High", "finding": "Non-refundable deposit",
                     "why_it_matters": "x", "support": "[1][7]",
                     "lease_quote": "Security deposit is   non-refundable."},
                    {"category": "Deposit", "severity": "Medium", "finding": "Paraphrased",
                     "why_it_matters": "y", "support": "",
                     "lease_quote": "The deposit cannot be refunded."},
                ],
                "recommendations": ["Ask for refundable deposit"],
            }) + "\n```")
        if "email from a prospective or current tenant" in sys_txt:
            with self.lock:
                self.email_prompts.append(user_txt)
            return SimpleNamespace(content="Subject: Lease changes\n\nDear [LANDLORD_NAME], ...")
        raise AssertionError("unexpected prompt")


class FakeRetriever:
    def __init__(self, top_k=6):
        pass

    def context_with_chunks(self, q, top_k=None, max_chars=9000):
        return "[1] chunk_id=3 source=ca_civil_code.pdf page=4\nDeposits are refundable.", [
            {"chunk_id": 3, "source": "ca_civil_code.pdf", "source_name": "ca civil code",
             "page": 4, "section": None, "url": None, "score": 0.4, "text": "Deposits are refundable."}]


@pytest.fixture
def graph(monkeypatch):
    llm = FakeLLM()
    monkeypatch.setattr(agent, "ChatOpenAI", lambda **kw: llm)
    monkeypatch.setattr(agent, "LawRetriever", FakeRetriever)
    monkeypatch.setattr(agent, "LEASE_CHUNK_CHARS", 1000)
    monkeypatch.setattr(agent, "LEASE_CHUNK_OVERLAP", 100)
    return agent.build_app(), llm


def _lease(n_chars: int) -> str:
    head = "Security deposit is non-refundable. "
    return (head + "filler text. " * n_chars)[:n_chars]


def test_graph_chunks_long_lease_merges_and_annotates_flags(graph, caplog):
    app, llm = graph
    with caplog.at_level(logging.WARNING, logger="leazard.agent"):
        state = app.invoke({"zip_code": "94110", "lease_text": _lease(2500)})

    assert state["status"] == "ok"
    assert len(llm.extract_prompts) == 3  # 2500 chars, size 1000, overlap 100
    assert all("part " in p for p in llm.extract_prompts)
    assert not any("truncated" in r.message for r in caplog.records)
    assert state["lease_json"]["rent_amount"] == "$3000"
    assert state["lease_json"]["lease_type"] == "fixed"
    assert state["lease_json"]["other_key_terms"] == ["term-1", "term-2", "term-3"]

    risk = state["risk_json"]
    high, low = risk["flags"]
    assert (high["severity"], low["severity"]) == ("High", "Low")  # uncited Medium capped
    assert (risk["risk_score"], risk["risk_label"], risk["counts"]) == agent.score_flags(_flags("High", "Low"))
    assert risk["standard_clauses"] == []
    assert high["lease_quote"] == "Security deposit is non-refundable."
    assert low["lease_quote"] == ""  # paraphrase dropped, flag kept
    assert high["citations"] == [{"id": 1, "source": "ca civil code", "page": 4,
                                  "section": None, "url": None}]
    assert low["citations"] == []
    assert low["why_it_matters"] == "Uncertain: y"
    assert "LEASE_EXCERPT:" in llm.analyze_prompts[0]
    assert "MARKET_CONTEXT:" in llm.analyze_prompts[0] and "21 days" in llm.analyze_prompts[0]
    assert "A clause being present is not a risk." in llm.analyze_prompts[0]

    assert state["letter_text"].startswith("Subject:")
    assert "Non-refundable deposit" in llm.email_prompts[0]
    assert "ca civil code" in llm.email_prompts[0]
    assert "Paraphrased" not in llm.email_prompts[0]  # Low flags are not negotiated


def test_graph_warns_when_lease_exceeds_cap_without_logging_text(graph, caplog, monkeypatch):
    app, llm = graph
    monkeypatch.setattr(agent, "LEASE_CHARS", 1500)
    secret = "SSN 123-45-6789 "
    lease = secret + _lease(4000)
    with caplog.at_level(logging.WARNING, logger="leazard.agent"):
        app.invoke({"zip_code": "94110", "lease_text": lease})

    msgs = [r.getMessage() for r in caplog.records]
    assert any("truncated" in m and "1500" in m for m in msgs)
    assert not any(secret.strip() in m for m in msgs)
    assert sum(len(p) for p in llm.extract_prompts) < 1500 + 2 * 2000  # only capped text sent


def test_short_lease_single_extraction_call(graph):
    app, llm = graph
    app.invoke({"zip_code": "94110", "lease_text": _lease(500)})
    assert len(llm.extract_prompts) == 1
    assert "part " not in llm.extract_prompts[0]


class MultiCategoryLLM(FakeLLM):
    """Discovers `n` categories; analysis raises for titles in `broken`."""

    def __init__(self, n=5, broken=(), delay=0.0):
        super().__init__()
        self.n, self.broken, self.delay = n, set(broken), delay
        self.active = self.peak = 0

    def invoke(self, messages):
        sys_txt, user_txt = messages[0].content, messages[1].content
        if "tenant-protection expert" in sys_txt:
            return SimpleNamespace(content=json.dumps(
                [{"title": f"Cat{i}", "terms": ["deposit"]} for i in range(self.n)]))
        if "lease risk analyzer" in sys_txt:
            title = re.search(r"CATEGORY: (\S+)", user_txt).group(1)
            with self.lock:
                self.active += 1
                self.peak = max(self.peak, self.active)
            try:
                if self.delay:
                    time.sleep(self.delay)
                if title in self.broken:
                    raise TimeoutError("llm timed out")
            finally:
                with self.lock:
                    self.active -= 1
        return super().invoke(messages)


def _graph_with(monkeypatch, llm):
    monkeypatch.setattr(agent, "ChatOpenAI", lambda **kw: llm)
    monkeypatch.setattr(agent, "LawRetriever", FakeRetriever)
    return agent.build_app()


def test_failed_categories_are_skipped_and_counted(monkeypatch):
    app = _graph_with(monkeypatch, MultiCategoryLLM(n=4, broken={"Cat1", "Cat3"}))
    state = app.invoke({"zip_code": "94110", "lease_text": _lease(500)})
    assert state["status"] == "ok"
    assert state["risk_json"]["skipped_categories"] == 2
    # Two surviving categories report the same two clauses; duplicates are merged.
    assert len(state["risk_json"]["flags"]) == 2
    assert state["letter_text"]


def test_all_categories_failing_is_an_error_not_a_low_score(monkeypatch):
    app = _graph_with(monkeypatch, MultiCategoryLLM(n=3, broken={"Cat0", "Cat1", "Cat2"}))
    state = app.invoke({"zip_code": "94110", "lease_text": _lease(500)})
    assert state["status"] == "error"
    assert "risk_json" not in state
    assert "letter_text" not in state


def test_category_pool_is_capped(monkeypatch):
    monkeypatch.setattr(agent, "CATEGORY_WORKERS", 2)
    llm = MultiCategoryLLM(n=6, delay=0.05)
    app = _graph_with(monkeypatch, llm)
    app.invoke({"zip_code": "94110", "lease_text": _lease(500)})
    assert llm.peak == 2


def test_llm_client_has_timeout_and_retries(monkeypatch):
    seen = {}
    monkeypatch.setattr(agent, "ChatOpenAI", lambda **kw: seen.update(kw) or FakeLLM())
    monkeypatch.setattr(agent, "LawRetriever", FakeRetriever)
    agent.build_app()
    assert seen["timeout"] == agent.LLM_TIMEOUT_S == 60
    assert seen["max_retries"] == agent.LLM_MAX_RETRIES == 2


def test_run_graph_reports_real_node_progress(monkeypatch):
    app = _graph_with(monkeypatch, MultiCategoryLLM(n=2))
    seen = []
    state = agent.run_graph(app, {"zip_code": "94110", "lease_text": _lease(500)},
                            lambda label, pct: seen.append((label, pct)))
    assert state["risk_json"]["skipped_categories"] == 0
    assert state["letter_text"].startswith("Subject:")
    assert "lease_text" in state
    assert seen == [
        ("Checking your ZIP", 0),
        ("Reading your lease", 5),
        ("Finding risk areas", 25),
        ("Checking California law", 40),
        ("Checking California law", 65),
        ("Checking California law", 90),
        ("Writing your negotiation email", 90),
        ("Writing your negotiation email", 95),
    ]


def test_run_graph_out_of_scope_and_deadline(monkeypatch):
    monkeypatch.setattr(agent, "CATEGORY_WORKERS", 1)
    llm = MultiCategoryLLM(n=6, delay=0.1)
    app = _graph_with(monkeypatch, llm)
    with pytest.raises(agent.PipelineError) as exc:
        agent.run_graph(app, {"zip_code": "10001", "lease_text": "x"})
    assert exc.value.code == "out_of_scope"

    t0 = time.monotonic()
    with pytest.raises(agent.PipelineError) as exc:
        agent.run_graph(app, {"zip_code": "94110", "lease_text": _lease(500)},
                        deadline=t0 + 0.15)
    assert exc.value.code == "timeout"
    assert time.monotonic() - t0 < 0.5  # remaining categories were cancelled, not run
    assert len(llm.analyze_prompts) < 6
    assert not llm.email_prompts


def test_email_falls_back_to_template_when_no_medium_or_high(monkeypatch):
    for out in (agent._fallback_email([]), agent._fallback_questions_email()):
        for ph in ("[LANDLORD_NAME]", "[TENANT_NAME]", "[RENTAL_ADDRESS]", "[DATE]"):
            assert ph in out


STANDARD_LEASE = ("Tenant shall pay a security deposit of $3,450.00 upon signing. "
                  "Smoking and vaping of any substance are prohibited inside the Premises. "
                  "Tenant shall maintain renter's insurance.")


class StandardLeaseLLM(FakeLLM):
    """Two topics that report the same deposit clause as OK, plus a High claim without support."""

    def __init__(self, calibration=None):
        super().__init__()
        self.calibration = calibration
        self.calibration_prompts: list[str] = []

    def invoke(self, messages):
        sys_txt, user_txt = messages[0].content, messages[1].content
        if "tenant-protection expert" in sys_txt:
            return SimpleNamespace(content=json.dumps(
                [{"title": "Deposit", "terms": ["deposit"]}, {"title": "Move-in", "terms": ["deposit"]}]))
        if "lease risk analyzer" in sys_txt:
            with self.lock:
                self.analyze_prompts.append(user_txt)
            title = re.search(r"CATEGORY: (\S+)", user_txt).group(1)
            flags = [{"category": title, "severity": "ok", "finding": "One month's rent deposit.",
                      "why_it_matters": "", "support": "",
                      "lease_quote": "Tenant shall pay a security deposit of $3,450.00 upon signing."}]
            if title == "Deposit":
                flags += [
                    {"category": "Smoking", "severity": "High", "finding": "Smoking ban",
                     "why_it_matters": "Not allowed.", "support": "[1]",
                     "lease_quote": "Smoking and vaping of any substance are prohibited inside the Premises."},
                    {"category": "Insurance", "severity": "OK", "finding": "Renter's insurance required.",
                     "why_it_matters": "", "support": "", "lease_quote": "Tenant shall maintain renter's insurance."},
                ]
            return SimpleNamespace(content=json.dumps({"flags": flags, "recommendations": []}))
        if "severity calibrator" in sys_txt:
            with self.lock:
                self.calibration_prompts.append(user_txt)
            return SimpleNamespace(content=json.dumps(self.calibration))
        if "email from a prospective or current tenant" in sys_txt:
            with self.lock:
                self.email_prompts.append((sys_txt, user_txt))
            return SimpleNamespace(content="Subject: Questions\n\nDear [LANDLORD_NAME], ...")
        return super().invoke(messages)


def test_standard_lease_scores_low_with_standard_clauses_and_questions_email(monkeypatch):
    llm = StandardLeaseLLM()
    state = _graph_with(monkeypatch, llm).invoke({"zip_code": "94110", "lease_text": STANDARD_LEASE})
    risk = state["risk_json"]
    assert (risk["risk_score"], risk["risk_label"]) == (agent.score_flags(_flags("Low"))[0], "Low")
    assert risk["counts"] == {"high": 0, "medium": 0, "low": 1, "ok": 2}
    # Irrelevant citation dropped, so the unsupported High is capped.
    [smoking] = risk["flags"]
    assert smoking["severity"] == "Low" and smoking["citations"] == []
    assert smoking["why_it_matters"].startswith("Uncertain:")
    assert [c["category"] for c in risk["standard_clauses"]] == ["Deposit / Move-in", "Insurance"]
    assert set(risk["standard_clauses"][0]) == {"category", "finding", "lease_quote"}
    [(sys_txt, user_txt)] = llm.email_prompts
    assert "clarifying questions" in sys_txt and "Smoking" in user_txt
    assert state["letter_text"].startswith("Subject:")
    assert not llm.calibration_prompts


def test_calibration_pass_adjusts_severity_but_cap_still_applies(monkeypatch):
    monkeypatch.setattr(agent, "CALIBRATION_PASS", True)
    llm = StandardLeaseLLM(calibration={"adjustments": [
        {"id": 0, "severity": "High", "reason": "upgrade without support"},
        {"id": 2, "severity": "Low", "reason": "worth knowing"},
        {"id": 99, "severity": "High"}, {"id": 1, "severity": "Bogus"},
    ]})
    state = _graph_with(monkeypatch, llm).invoke({"zip_code": "94110", "lease_text": STANDARD_LEASE})
    risk = state["risk_json"]
    assert len(llm.calibration_prompts) == 1
    assert "A clause being present is not a risk." in llm.calibration_prompts[0]
    assert risk["counts"] == {"high": 0, "medium": 0, "low": 3, "ok": 0}
    assert risk["risk_label"] == "Low"
    assert risk["standard_clauses"] == []
