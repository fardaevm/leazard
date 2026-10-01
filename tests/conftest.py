import os
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_TMP = Path(tempfile.mkdtemp(prefix="leazard-tests-"))
os.environ["SECRET_KEY"] = "test-secret-key-not-for-production"
os.environ["DATABASE_URL"] = f"sqlite:///{_TMP / 'test.db'}"
os.environ["UPLOAD_DIR"] = str(_TMP / "uploads")
os.environ["MAX_UPLOAD_MB"] = "1"
os.environ.setdefault("OPENAI_API_KEY", "sk-test-not-used")
os.environ["MAX_CONCURRENT_JOBS"] = "3"
os.environ["JOB_TIMEOUT_S"] = "30"


def make_pdf(text_lines: list[str]) -> bytes:
    """Build a minimal one-page PDF; empty text_lines gives a page with no text (scan-like)."""
    content = "BT /F1 11 Tf 50 750 Td 14 TL\n"
    for line in text_lines:
        safe = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        content += f"({safe}) Tj T*\n"
    content += "ET"
    stream = content.encode("latin-1") if text_lines else b""
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    return bytes(out)


class FakeGraph:
    """Mimics graph_app.stream(stream_mode=["updates", "custom"]) without any LLM calls.

    status: "ok" | "error" (fails at `fail_at`) | "out_of_scope".
    gate: optional threading.Event the run waits on before starting (to hold a job "running").
    """

    def __init__(self):
        self.calls = 0
        self.raise_exc: Exception | None = None
        self.status = "ok"
        self.fail_at = "discover_categories"
        self.categories = 3
        self.category_delay = 0.0
        self.gate: threading.Event | None = None

    def stream(self, state, config=None, stream_mode=None):
        self.calls += 1
        if self.gate is not None:
            self.gate.wait(10)
        if self.raise_exc:
            raise self.raise_exc
        if self.status == "out_of_scope":
            yield "updates", {"validate_zip": {"status": "out_of_scope",
                                               "message": "Out of scope region."}}
            return
        updates = [
            ("validate_zip",        {"status": "ok"}),
            ("extract_structured",  {"lease_json": {"address_or_city_if_present": "SF"}}),
            ("discover_categories", {"categories": [{"title": f"c{i}", "terms": []}
                                                    for i in range(self.categories)]}),
            ("analyze_risk",        {"risk_json": {"risk_score": 1.0, "risk_label": "Low", "flags": [],
                                                   "recommendations": [], "skipped_categories": 0}}),
            ("draft_letter",        {"letter_text": "Dear [LANDLORD_NAME]"}),
        ]
        for node, update in updates:
            if self.status == "error" and node == self.fail_at:
                yield "updates", {node: {"status": "error", "message": "boom"}}
                return
            if node == "analyze_risk":
                for i in range(self.categories):
                    time.sleep(self.category_delay)
                    yield "custom", {"categories_done": i + 1, "categories_total": self.categories}
            yield "updates", {node: update}


@pytest.fixture(scope="session")
def main_module():
    import rag.indexer
    import agent
    rag.indexer.ensure_index = lambda: None
    agent.build_app = lambda: FakeGraph()
    import main
    return main


@pytest.fixture
def client(main_module, monkeypatch):
    from fastapi.testclient import TestClient
    graph = FakeGraph()
    monkeypatch.setattr(main_module, "graph_app", graph)
    c = TestClient(main_module.app)
    c.graph = graph
    return c


@pytest.fixture
def upload_dir(main_module) -> Path:
    return main_module.UPLOAD_DIR
