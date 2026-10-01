"""Golden test against the real LLM and law index. Run with: pytest -m live -s tests/test_golden.py"""
import os
import statistics
from pathlib import Path

import pytest
from dotenv import dotenv_values

import agent
from agent import build_app, run_graph
from utils.extract_pdf import extract_text_from_pdf

ROOT = Path(__file__).resolve().parent.parent
GOLDEN_PDF = ROOT / "tests" / "golden" / "sample_sf_lease_agreement.pdf"
RUNS = int(os.getenv("GOLDEN_RUNS", "3"))

# Topic → substrings (lowercase) that identify it in category, finding or lease_quote.
STANDARD_TOPICS = {
    "deposit":          ("security deposit",),
    "21-day return":    ("21 days", "21-day", "1950.5"),
    "entry notice":     ("24 hours",),
    "governing law":    ("governed by", "governing law"),
    "smoking":          ("smoking",),
    "renter's insurance": ("insurance",),
}


def _text(item: dict) -> str:
    return " ".join(str(item.get(k) or "") for k in ("category", "finding", "lease_quote")).lower()


def _topic_problems(risk: dict) -> list[str]:
    problems = []
    standard = [_text(s) for s in risk.get("standard_clauses") or []]
    flags = [(f.get("severity"), _text(f)) for f in risk.get("flags") or []]
    for topic, needles in STANDARD_TOPICS.items():
        def hit(t: str, needles=needles) -> bool:
            return any(n in t for n in needles)
        too_severe = [sev for sev, t in flags if hit(t) and sev not in ("OK", "Low")]
        if too_severe:
            problems.append(f"{topic}: flagged {too_severe}")
        elif not any(hit(t) for t in standard) and not any(hit(t) for _, t in flags):
            problems.append(f"{topic}: not reported")
    return problems


@pytest.mark.live
def test_golden_sf_lease_scores_low(monkeypatch):
    if not GOLDEN_PDF.exists():
        pytest.skip("golden PDF missing")
    key = dotenv_values(ROOT / ".env").get("OPENAI_API_KEY")
    if key:
        monkeypatch.setenv("OPENAI_API_KEY", key)
    if os.getenv("OPENAI_API_KEY", "").startswith("sk-test"):
        pytest.skip("no real OPENAI_API_KEY")

    text = extract_text_from_pdf(GOLDEN_PDF)
    app = build_app()
    results = []
    for _ in range(RUNS):
        risk = run_graph(app, {"zip_code": "94105", "lease_text": text})["risk_json"]
        results.append(risk)
        print(f"\nscore={risk['risk_score']} label={risk['risk_label']} counts={risk['counts']}")
        for f in risk["flags"]:
            print(f"  [{f['severity']}] {f['category']}: {f['finding']}")
        for s in risk["standard_clauses"]:
            print(f"  [OK] {s['category']}: {s['finding']}")
        print("  topic problems:", _topic_problems(risk))

    scores = [r["risk_score"] for r in results]
    print(f"\nscores={scores} mean={statistics.mean(scores):.2f} "
          f"stdev={statistics.pstdev(scores):.2f} range={max(scores) - min(scores):.1f}")

    for risk in results:
        assert risk["risk_label"] in ("Low", "Moderate")
        assert risk["counts"]["high"] == 0
        assert not [f for f in risk["flags"] if f["severity"] == "High"]
        assert _topic_problems(risk) == []
        assert agent.score_flags(risk["flags"])[:2] == (risk["risk_score"], risk["risk_label"])
