import threading
import time
import uuid

import pytest

from conftest import make_pdf
from test_api import LEASE_LINES, _auth, _files

import agent
import db as dbmod


def _post_job(client, headers, data: bytes, zip_code="94110", filename="lease.pdf"):
    return client.post(
        "/jobs",
        headers=headers,
        data={"zip_code": zip_code},
        files={"lease": (filename, data, "application/pdf")},
    )


def _wait(client, headers, job_id, timeout=10.0) -> dict:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        body = client.get(f"/jobs/{job_id}", headers=headers).json()
        if body["status"] in ("done", "error"):
            return body
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not finish: {body}")


@pytest.fixture(autouse=True)
def _isolate_jobs(client):
    """Each test starts with no active jobs and never leaves a gated job behind."""
    dbmod.mark_interrupted_jobs()
    yield
    if client.graph.gate is not None:
        client.graph.gate.set()
    end = time.monotonic() + 10
    with dbmod.SessionLocal() as s:
        while s.query(dbmod.Job).filter(dbmod.Job.status.in_(dbmod.JOB_ACTIVE_STATUSES)).count():
            assert time.monotonic() < end, "jobs still active after test"
            time.sleep(0.02)
            s.expire_all()


# ── Happy path ────────────────────────────────────────────────────────────────

def test_create_job_returns_202_quickly_and_completes(client):
    headers = _auth(client)
    client.graph.gate = threading.Event()  # graph won't even start until released

    t0 = time.monotonic()
    r = _post_job(client, headers, make_pdf(LEASE_LINES))
    assert r.status_code == 202, r.text
    assert time.monotonic() - t0 < 1.0
    job_id = r.json()["job_id"]
    assert set(r.json()) == {"job_id"}

    mid = client.get(f"/jobs/{job_id}", headers=headers).json()
    assert mid["status"] in ("queued", "running")
    assert mid["lease_id"] is None

    client.graph.gate.set()
    done = _wait(client, headers, job_id)
    assert done == {"status": "done", "step": "Done", "progress": 100,
                    "error_code": None, "error_message": None, "lease_id": done["lease_id"]}

    lease = client.get(f"/history/{done['lease_id']}", headers=headers)
    assert lease.status_code == 200
    assert lease.json()["risk_json"]["skipped_categories"] == 0


def test_progress_advances_through_expected_steps(client, main_module, monkeypatch):
    seen: list[tuple[str, int]] = []
    real = main_module._update_job

    def spy(job_id, **fields):
        if "step" in fields:
            seen.append((fields["step"], fields["progress"]))
        real(job_id, **fields)

    monkeypatch.setattr(main_module, "_update_job", spy)
    client.graph.categories = 4
    headers = _auth(client)
    job_id = _post_job(client, headers, make_pdf(LEASE_LINES)).json()["job_id"]
    assert _wait(client, headers, job_id)["status"] == "done"

    labels = [s for s, _ in seen]
    progress = [p for _, p in seen]
    assert progress == sorted(progress), "progress must never go backwards"
    assert seen[0] == ("Checking your ZIP", 0)
    assert ("Reading your lease", 5) in seen
    assert ("Finding risk areas", 25) in seen
    # discover done → 40, then one tick per category up to 90
    law = [p for s, p in seen if s == "Checking California law"]
    assert law == [40, 52, 65, 77, 90]
    assert ("Writing your negotiation email", 95) in seen
    assert seen[-1] == ("Done", 100)
    ordered = list(dict.fromkeys(labels))
    assert ordered == ["Checking your ZIP", "Reading your lease", "Finding risk areas",
                       "Checking California law", "Writing your negotiation email", "Done"]


def test_step_labels_live_in_one_constants_dict():
    assert agent.JOB_STEPS["validate_zip"] == {"label": "Checking your ZIP", "progress": 5}
    assert agent.JOB_STEPS["extract_structured"] == {"label": "Reading your lease", "progress": 25}
    assert agent.JOB_STEPS["discover_categories"] == {"label": "Finding risk areas", "progress": 40}
    assert agent.JOB_STEPS["analyze_risk"] == {"label": "Checking California law", "progress": 90}
    assert agent.JOB_STEPS["draft_letter"] == {"label": "Writing your negotiation email", "progress": 95}
    assert agent.JOB_STEPS["done"]["progress"] == 100


# ── Ownership ─────────────────────────────────────────────────────────────────

def test_job_is_owner_only(client):
    owner, other = _auth(client), _auth(client)
    job_id = _post_job(client, owner, make_pdf(LEASE_LINES)).json()["job_id"]
    assert client.get(f"/jobs/{job_id}", headers=other).status_code == 404
    assert client.get(f"/jobs/{uuid.uuid4().hex}", headers=owner).status_code == 404
    assert client.get(f"/jobs/{job_id}").status_code in (401, 403)
    assert _wait(client, owner, job_id)["status"] == "done"


def test_job_validation_matches_analyze(client, upload_dir):
    headers = _auth(client)
    before = _files(upload_dir)
    assert _post_job(client, headers, make_pdf(LEASE_LINES), zip_code="9411").status_code == 400
    assert _post_job(client, headers, make_pdf(LEASE_LINES), filename="x.txt").status_code == 400
    assert _post_job(client, headers, b"MZ not a pdf" * 10).status_code == 400
    assert _post_job(client, headers, b"%PDF-" + b"0" * (2 * 1024 * 1024)).status_code == 413
    assert _files(upload_dir) == before
    assert client.graph.calls == 0


# ── Failure paths ─────────────────────────────────────────────────────────────

def _assert_failed(client, headers, job_id, code, upload_dir, before):
    body = _wait(client, headers, job_id)
    assert body["status"] == "error"
    assert body["error_code"] == code
    assert body["error_message"]
    assert "Traceback" not in body["error_message"]
    assert "Tenant shall pay" not in body["error_message"]
    assert body["lease_id"] is None
    assert _files(upload_dir) == before
    return body


def test_out_of_scope_job(client, upload_dir):
    headers = _auth(client)
    before = _files(upload_dir)
    client.graph.status = "out_of_scope"
    job_id = _post_job(client, headers, make_pdf(LEASE_LINES), zip_code="10001").json()["job_id"]
    body = _assert_failed(client, headers, job_id, "out_of_scope", upload_dir, before)
    assert body["error_message"] == "Out of scope region."  # message comes from the graph unchanged
    assert client.get("/history", headers=headers).json() == []


def test_out_of_scope_real_message_from_graph():
    with pytest.raises(agent.PipelineError) as exc:
        class G:
            def stream(self, state, config=None, stream_mode=None):
                yield "updates", {"validate_zip": {"status": "out_of_scope",
                                                   "message": agent.ERROR_MESSAGES["out_of_scope"]}}
        agent.run_graph(G(), {"zip_code": "10001"})
    assert exc.value.code == "out_of_scope"
    assert exc.value.message == agent.ERROR_MESSAGES["out_of_scope"]


def test_scanned_pdf_job(client, upload_dir):
    headers = _auth(client)
    before = _files(upload_dir)
    job_id = _post_job(client, headers, make_pdf([])).json()["job_id"]
    body = _assert_failed(client, headers, job_id, "scanned_pdf", upload_dir, before)
    assert body["error_message"] == "This looks like a scanned PDF. Scanned leases aren't supported yet."
    assert client.graph.calls == 0


def test_unreadable_pdf_job_is_extraction_failed(client, upload_dir):
    headers = _auth(client)
    before = _files(upload_dir)
    job_id = _post_job(client, headers, b"%PDF-1.4\nthis is garbage, not a pdf body").json()["job_id"]
    _assert_failed(client, headers, job_id, "extraction_failed", upload_dir, before)


def test_llm_extraction_error_is_extraction_failed(client, upload_dir):
    headers = _auth(client)
    before = _files(upload_dir)
    client.graph.status, client.graph.fail_at = "error", "extract_structured"
    job_id = _post_job(client, headers, make_pdf(LEASE_LINES)).json()["job_id"]
    _assert_failed(client, headers, job_id, "extraction_failed", upload_dir, before)


def test_graph_exception_is_analysis_failed_and_not_leaked(client, upload_dir, caplog):
    headers = _auth(client)
    before = _files(upload_dir)
    client.graph.raise_exc = RuntimeError("SECRET LEASE TEXT Tenant shall pay")
    job_id = _post_job(client, headers, make_pdf(LEASE_LINES)).json()["job_id"]
    body = _assert_failed(client, headers, job_id, "analysis_failed", upload_dir, before)
    assert "SECRET" not in body["error_message"]
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "RuntimeError" in logged
    assert "SECRET" not in logged


def test_node_error_is_analysis_failed(client, upload_dir):
    headers = _auth(client)
    before = _files(upload_dir)
    client.graph.status, client.graph.fail_at = "error", "discover_categories"
    job_id = _post_job(client, headers, make_pdf(LEASE_LINES)).json()["job_id"]
    _assert_failed(client, headers, job_id, "analysis_failed", upload_dir, before)


def test_job_timeout(client, main_module, monkeypatch, upload_dir):
    headers = _auth(client)
    before = _files(upload_dir)
    monkeypatch.setattr(main_module, "JOB_TIMEOUT_S", 0.2)
    client.graph.categories, client.graph.category_delay = 5, 0.1
    job_id = _post_job(client, headers, make_pdf(LEASE_LINES)).json()["job_id"]
    body = _assert_failed(client, headers, job_id, "timeout", upload_dir, before)
    assert body["error_message"] == agent.ERROR_MESSAGES["timeout"]


def test_failed_job_does_not_block_next_job(client):
    headers = _auth(client)
    client.graph.raise_exc = RuntimeError("x")
    first = _post_job(client, headers, make_pdf(LEASE_LINES)).json()["job_id"]
    assert _wait(client, headers, first)["status"] == "error"
    client.graph.raise_exc = None
    second = _post_job(client, headers, make_pdf(LEASE_LINES))
    assert second.status_code == 202
    assert _wait(client, headers, second.json()["job_id"])["status"] == "done"


# ── Concurrency limits ────────────────────────────────────────────────────────

def test_one_active_job_per_user(client, upload_dir):
    headers = _auth(client)
    client.graph.gate = threading.Event()
    first = _post_job(client, headers, make_pdf(LEASE_LINES))
    assert first.status_code == 202
    before = _files(upload_dir)

    second = _post_job(client, headers, make_pdf(LEASE_LINES))
    assert second.status_code == 429
    assert "already have a lease analysis in progress" in second.json()["detail"]
    assert _files(upload_dir) == before

    other = _auth(client)
    assert _post_job(client, other, make_pdf(LEASE_LINES)).status_code == 202

    client.graph.gate.set()
    assert _wait(client, headers, first.json()["job_id"])["status"] == "done"
    assert _post_job(client, headers, make_pdf(LEASE_LINES)).status_code == 202


def test_global_concurrency_cap(client, main_module, monkeypatch):
    monkeypatch.setattr(main_module, "MAX_CONCURRENT_JOBS", 2)
    client.graph.gate = threading.Event()
    users = [_auth(client) for _ in range(3)]
    assert _post_job(client, users[0], make_pdf(LEASE_LINES)).status_code == 202
    assert _post_job(client, users[1], make_pdf(LEASE_LINES)).status_code == 202
    third = _post_job(client, users[2], make_pdf(LEASE_LINES))
    assert third.status_code == 429
    assert "busy" in third.json()["detail"]


def test_concurrent_submissions_from_same_user_admit_only_one(client, main_module):
    headers = _auth(client)
    client.graph.gate = threading.Event()
    pdf = make_pdf(LEASE_LINES)
    codes: list[int] = []
    threads = [threading.Thread(target=lambda: codes.append(_post_job(client, headers, pdf).status_code))
               for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(codes) == [202, 429, 429, 429, 429]


# ── Startup cleanup / migration ───────────────────────────────────────────────

def test_startup_marks_stuck_jobs_interrupted(client):
    headers = _auth(client)
    with dbmod.SessionLocal() as s:
        uid = s.query(dbmod.User).order_by(dbmod.User.id.desc()).first().id
        stuck = [dbmod.Job(user_id=uid, status=st, step="x", progress=30) for st in ("queued", "running")]
        finished = dbmod.Job(user_id=uid, status="done", step="Done", progress=100)
        s.add_all([*stuck, finished])
        s.commit()
        ids = [j.id for j in stuck]
        done_id = finished.id

    assert dbmod.mark_interrupted_jobs() == 2

    for job_id in ids:
        body = client.get(f"/jobs/{job_id}", headers=headers).json()
        assert body["status"] == "error"
        assert body["error_code"] == "interrupted"
        assert body["error_message"]
    assert client.get(f"/jobs/{done_id}", headers=headers).json()["status"] == "done"


def test_create_tables_migrates_existing_db_additively(tmp_path, monkeypatch):
    from sqlalchemy import create_engine, inspect, text

    eng = create_engine(f"sqlite:///{tmp_path / 'old.db'}")
    with eng.begin() as c:
        c.execute(text("CREATE TABLE users (id INTEGER PRIMARY KEY, username VARCHAR(64) NOT NULL, "
                       "hashed_password VARCHAR(256) NOT NULL, created_at DATETIME)"))
        c.execute(text("CREATE TABLE leases (id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, "
                       "file_id VARCHAR(64) NOT NULL, risk_score FLOAT)"))
        c.execute(text("INSERT INTO users (id, username, hashed_password) VALUES (1, 'old', 'h')"))
        c.execute(text("INSERT INTO leases (id, user_id, file_id, risk_score) VALUES (7, 1, 'abc', 4.2)"))

    monkeypatch.setattr(dbmod, "engine", eng)
    dbmod.create_tables()
    dbmod.create_tables()  # idempotent

    insp = inspect(eng)
    assert insp.has_table("jobs")
    assert {"letter_text", "risk_json", "zip_code"} <= {c["name"] for c in insp.get_columns("leases")}
    with eng.connect() as c:
        assert c.execute(text("SELECT id, file_id, risk_score FROM leases")).all() == [(7, "abc", 4.2)]
        assert c.execute(text("SELECT username FROM users")).scalar() == "old"


def test_sqlite_uses_wal(main_module):
    with dbmod.engine.connect() as c:
        assert c.exec_driver_sql("PRAGMA journal_mode").scalar().lower() == "wal"


# ── /analyze stays backward compatible ────────────────────────────────────────

def test_analyze_out_of_scope_keeps_200_shape(client, upload_dir):
    headers = _auth(client)
    before = _files(upload_dir)
    client.graph.status = "out_of_scope"
    r = client.post("/analyze", headers=headers, data={"zip_code": "10001"},
                    files={"lease": ("lease.pdf", make_pdf(LEASE_LINES), "application/pdf")})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "out_of_scope"
    assert body["message"]
    for key in ("id", "file_id", "zip_code", "lease_json", "risk_json", "letter_text"):
        assert key in body
    assert _files(upload_dir) == before
