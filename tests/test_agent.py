import json
import logging
import re
import threading
from types import SimpleNamespace

import pytest

import agent
from rag.retriever import LawRetriever, RetrievedChunk, display_source_name


# ── Score normalization ───────────────────────────────────────────────────────

def _flags(*sevs):
    return [{"severity": s} for s in sevs]


def test_score_zero_for_no_or_unknown_flags():
    assert agent._score([]) == 0.0
    assert agent._score(_flags("Bogus", "")) == 0.0


def test_score_is_bounded_monotonic_and_does_not_saturate_early():
    many_medium = agent._score(_flags(*["Medium"] * 8))
    many_high = agent._score(_flags(*["High"] * 8))
    assert 0 < agent._score(_flags("Low")) < agent._score(_flags("Medium")) < agent._score(_flags("High"))
    assert many_medium < many_high < 10.0
    # The old formula (sum/2, capped at 10) gave 10.0 for both of these.
    assert agent._score(_flags(*["High"] * 4)) < agent._score(_flags(*["High"] * 6))
    assert agent._score(_flags(*["High"] * 100)) <= 10.0


def test_score_is_deterministic_and_order_independent():
    a = _flags("High", "Low", "Medium", "Medium")
    assert agent._score(a) == agent._score(list(reversed(a))) == agent._score(a)


def test_score_expected_values_with_default_scale():
    assert agent.RISK_SCORE_SCALE == 10
    assert agent._score(_flags("High")) == 3.9
    assert agent._score(_flags("High", "High")) == 6.3


@pytest.mark.parametrize("score,label", [(0, "Low"), (3.4, "Low"), (3.5, "Moderate"),
                                         (6.9, "Moderate"), (7.0, "High"), (10, "High")])
def test_risk_label(score, label):
    assert agent._risk_label(score) == label


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
             "page": 4, "section": None, "url": None}]


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
    assert risk["risk_score"] == agent._score(_flags("High", "Medium"))
    assert risk["risk_label"] == agent._risk_label(risk["risk_score"])
    high, medium = risk["flags"]
    assert high["lease_quote"] == "Security deposit is non-refundable."
    assert medium["lease_quote"] == ""  # paraphrase dropped, flag kept
    assert high["citations"] == [{"id": 1, "source": "ca civil code", "page": 4,
                                  "section": None, "url": None}]
    assert medium["citations"] == []
    assert "LEASE_EXCERPT:" in llm.analyze_prompts[0]

    assert state["letter_text"].startswith("Subject:")
    assert "Non-refundable deposit" in llm.email_prompts[0]
    assert "ca civil code" in llm.email_prompts[0]


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


def test_email_falls_back_to_template_when_no_medium_or_high(monkeypatch):
    out = agent._fallback_email([])
    for ph in ("[LANDLORD_NAME]", "[TENANT_NAME]", "[RENTAL_ADDRESS]", "[DATE]"):
        assert ph in out
