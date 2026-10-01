# main.py
import os
import sys

from dotenv import load_dotenv
load_dotenv()

# Fail fast, before the (slow, paid) RAG index build. Never print the values.
_missing = [k for k in ("OPENAI_API_KEY", "SECRET_KEY") if not os.getenv(k, "").strip()]
if _missing:
    sys.exit(f"Missing required environment variable(s): {', '.join(_missing)}. See .env.example.")
if len(os.environ["SECRET_KEY"].strip()) < 32:
    sys.exit("SECRET_KEY must be at least 32 characters (e.g. `openssl rand -hex 32`).")

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import asyncio
import json
import logging
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Depends, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy.orm import Session

from utils.extract_pdf import extract_text_from_pdf, looks_scanned, PDF_MAGIC
from rag.indexer import ensure_index
from agent import build_app, run_graph, PipelineError, ProgressFn, JOB_STEPS
from db import (
    create_tables, get_db, mark_interrupted_jobs, SessionLocal,
    User, LeaseRecord, Job, JOB_ACTIVE_STATUSES,
)
from auth import hash_password, verify_password, create_token, decode_token

log = logging.getLogger("leazard.api")

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads"))
BASE_DIR   = Path(__file__).resolve().parent
MAX_UPLOAD_MB     = float(os.getenv("MAX_UPLOAD_MB", "10"))
MAX_UPLOAD_BYTES  = int(MAX_UPLOAD_MB * 1024 * 1024)
UPLOAD_READ_CHUNK = 1024 * 1024
MAX_CONCURRENT_JOBS = max(int(os.getenv("MAX_CONCURRENT_JOBS", "3")), 1)
JOB_TIMEOUT_S       = float(os.getenv("JOB_TIMEOUT_S", "240"))

# Owned by the app (not per-request), so jobs outlive the POST /jobs request.
_job_executor   = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_JOBS, thread_name_prefix="leazard-job")
_job_admit_lock = threading.Lock()


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    yield
    # Cancelled queued jobs stay "queued" in the DB and are failed on next startup.
    _job_executor.shutdown(wait=False, cancel_futures=True)


app = FastAPI(title="Leaze", lifespan=_lifespan)
app.mount("/ui", StaticFiles(directory=str(BASE_DIR / "ui")), name="ui")

# Startup
ensure_index()
create_tables()
if (_stuck := mark_interrupted_jobs()):
    log.warning("Marked %d interrupted job(s) as failed", _stuck)
graph_app = build_app()

security = HTTPBearer()


# ── Auth dependency ───────────────────────────────────────────────────────────

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
) -> User:
    payload = decode_token(credentials.credentials)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")
    user = db.query(User).filter(User.id == int(payload["sub"])).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found.")
    return user


# ── Request schemas ───────────────────────────────────────────────────────────

class AuthBody(BaseModel):
    username: str
    password: str


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def home():
    return FileResponse(BASE_DIR / "ui" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/auth/register", status_code=201)
def register(body: AuthBody, db: Session = Depends(get_db)):
    if len(body.username) < 3:
        raise HTTPException(400, "Username must be at least 3 characters.")
    if len(body.password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters.")
    if db.query(User).filter(User.username == body.username).first():
        raise HTTPException(400, "Username already taken.")
    db.add(User(username=body.username, hashed_password=hash_password(body.password)))
    db.commit()
    return {"ok": True}


@app.post("/auth/login")
def login(body: AuthBody, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == body.username).first()
    if not user or not verify_password(body.password, user.hashed_password):
        raise HTTPException(401, "Invalid username or password.")
    return {"access_token": create_token(user.id, user.username), "token_type": "bearer"}


@app.get("/history")
def get_history(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    records = (
        db.query(LeaseRecord)
        .filter(LeaseRecord.user_id == user.id)
        .order_by(LeaseRecord.created_at.desc())
        .all()
    )
    return [
        {
            "id":                r.id,
            "file_id":           r.file_id,
            "address":           r.address,
            "zip_code":          r.zip_code,
            "risk_score":        r.risk_score,
            "original_filename": r.original_filename,
            "created_at":        r.created_at.isoformat(),
        }
        for r in records
    ]


@app.get("/history/{lease_id}")
def get_lease_record(
    lease_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    record = db.query(LeaseRecord).filter(
        LeaseRecord.id == lease_id, LeaseRecord.user_id == user.id
    ).first()
    if not record:
        raise HTTPException(404, "Lease not found.")
    return {
        "id":                record.id,
        "file_id":           record.file_id,
        "address":           record.address,
        "zip_code":          record.zip_code,
        "risk_score":        record.risk_score,
        "original_filename": record.original_filename,
        "created_at":        record.created_at.isoformat(),
        "lease_json":        json.loads(record.lease_json  or "{}"),
        "risk_json":         json.loads(record.risk_json   or "{}"),
        "letter_text":       record.letter_text,
    }


@app.get("/uploads/{file_id}")
def get_pdf(
    file_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # Prevent path traversal — file_id must be hex UUID
    if not file_id or not all(c in "0123456789abcdefABCDEF" for c in file_id):
        raise HTTPException(400, "Invalid file ID.")
    owned = db.query(LeaseRecord.id).filter(
        LeaseRecord.file_id == file_id, LeaseRecord.user_id == user.id
    ).first()
    path = UPLOAD_DIR / f"{file_id}.pdf"
    if not owned or not path.exists():
        raise HTTPException(404, "PDF not found.")
    return FileResponse(path, media_type="application/pdf")


async def _save_upload(lease: UploadFile, pdf_path: Path) -> None:
    """Stream the upload to disk, enforcing MAX_UPLOAD_BYTES and the %PDF- header."""
    size = 0
    with pdf_path.open("wb") as out:
        while chunk := await lease.read(UPLOAD_READ_CHUNK):
            if size == 0 and not chunk.startswith(PDF_MAGIC):
                raise HTTPException(400, "File is not a valid PDF.")
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                raise HTTPException(413, f"File exceeds the {MAX_UPLOAD_MB:g} MB limit.")
            out.write(chunk)
    if size == 0:
        raise HTTPException(400, "Uploaded file is empty.")


async def _validate_and_save(request: Request, lease: UploadFile, zip_code: str) -> str:
    """Validate the form + upload and store it. Returns the new file_id; nothing is left on failure."""
    if not (zip_code.isdigit() and len(zip_code) == 5):
        raise HTTPException(400, "ZIP must be 5 digits.")
    if not (lease.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_UPLOAD_BYTES + 64 * 1024:
        raise HTTPException(413, f"File exceeds the {MAX_UPLOAD_MB:g} MB limit.")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    file_id  = uuid.uuid4().hex
    pdf_path = UPLOAD_DIR / f"{file_id}.pdf"
    try:
        await _save_upload(lease, pdf_path)
    except BaseException:
        pdf_path.unlink(missing_ok=True)
        raise
    return file_id


def _log_failure(what: str, e: BaseException) -> None:
    # Type + stack frames only: exception messages may contain lease text.
    log.error("%s: %s\n%s", what, type(e).__name__, "".join(traceback.format_tb(e.__traceback__)))


def run_pipeline(
    file_id: str,
    zip_code: str,
    user_id: int,
    filename: str | None,
    on_progress: ProgressFn | None = None,
    deadline: float | None = None,
) -> tuple[LeaseRecord, dict]:
    """Shared by /analyze and jobs: PDF text → graph → persisted LeaseRecord.

    Raises PipelineError with a user-safe message; the uploaded PDF is deleted on any failure.
    Uses its own DB session so it is safe to call from worker threads.
    """
    pdf_path = UPLOAD_DIR / f"{file_id}.pdf"
    try:
        try:
            text = extract_text_from_pdf(pdf_path)
        except Exception:
            raise PipelineError("extraction_failed", "Could not read this PDF.", http_status=422)
        if looks_scanned(text):
            raise PipelineError("scanned_pdf", http_status=422)

        try:
            state = run_graph(graph_app, {"zip_code": zip_code, "lease_text": text}, on_progress, deadline)
        except PipelineError:
            raise
        except Exception as e:
            _log_failure("Lease analysis failed", e)
            raise PipelineError("analysis_failed")

        lease_json = state.get("lease_json") or {}
        risk_json  = state.get("risk_json")  or {}
        address    = (lease_json.get("address_or_city_if_present") or "").strip() or f"ZIP {zip_code}"

        record = LeaseRecord(
            user_id           = user_id,
            file_id           = file_id,
            original_filename = filename,
            address           = address,
            zip_code          = zip_code,
            risk_score        = risk_json.get("risk_score"),
            lease_json        = json.dumps(lease_json),
            risk_json         = json.dumps(risk_json),
            letter_text       = state.get("letter_text"),
        )
        with SessionLocal(expire_on_commit=False) as db:
            db.add(record)
            db.commit()
    except BaseException:
        pdf_path.unlink(missing_ok=True)
        raise
    return record, state


@app.post("/analyze")
async def analyze(
    request:  Request,
    lease:    UploadFile = File(...),
    zip_code: str        = Form(...),
    user: User           = Depends(get_current_user),
):
    file_id = await _validate_and_save(request, lease, zip_code)
    loop = asyncio.get_running_loop()
    try:
        record, state = await loop.run_in_executor(
            None, lambda: run_pipeline(file_id, zip_code, user.id, lease.filename)
        )
    except PipelineError as e:
        if e.code == "out_of_scope":
            return JSONResponse({
                "id": None, "file_id": None, "zip_code": zip_code,
                "status": "out_of_scope", "message": e.message,
                "lease_json": {}, "risk_json": {}, "letter_text": None,
            })
        raise HTTPException(e.http_status, e.message)

    # Exclude raw lease_text from the response (can be megabytes)
    return JSONResponse({
        "id":          record.id,
        "file_id":     file_id,
        "zip_code":    zip_code,
        "status":      state.get("status"),
        "message":     state.get("message"),
        "lease_json":  json.loads(record.lease_json),
        "risk_json":   json.loads(record.risk_json),
        "letter_text": record.letter_text,
    })


# ── Background jobs ───────────────────────────────────────────────────────────

def _update_job(job_id: str, **fields) -> None:
    with SessionLocal() as db:
        db.query(Job).filter(Job.id == job_id).update(
            {**fields, "updated_at": datetime.utcnow()}, synchronize_session=False
        )
        db.commit()


def _job_limit_error(db: Session, user_id: int) -> str | None:
    active = db.query(Job).filter(Job.status.in_(JOB_ACTIVE_STATUSES))
    if active.filter(Job.user_id == user_id).first():
        return "You already have a lease analysis in progress. Please wait for it to finish."
    if active.count() >= MAX_CONCURRENT_JOBS:
        return "The server is busy analyzing other leases. Please try again in a minute."
    return None


def _run_job(job_id: str, file_id: str, zip_code: str, user_id: int, filename: str | None) -> None:
    deadline = time.monotonic() + JOB_TIMEOUT_S
    best = 0

    def on_progress(label: str, pct: int) -> None:
        nonlocal best
        best = max(best, min(int(pct), 99))
        _update_job(job_id, step=label, progress=best)

    try:
        _update_job(job_id, status="running")
        record, _ = run_pipeline(file_id, zip_code, user_id, filename, on_progress, deadline)
        _update_job(job_id, status="done", step=JOB_STEPS["done"]["label"],
                    progress=JOB_STEPS["done"]["progress"], lease_record_id=record.id)
    except PipelineError as e:
        log.warning("Job %s failed: %s", job_id, e.code)
        _update_job(job_id, status="error", error_code=e.code, error_message=e.message)
    except Exception as e:
        _log_failure(f"Job {job_id} crashed", e)
        _update_job(job_id, status="error", error_code="analysis_failed",
                    error_message=PipelineError("analysis_failed").message)


@app.post("/jobs", status_code=202)
async def create_job(
    request:  Request,
    lease:    UploadFile = File(...),
    zip_code: str        = Form(...),
    user: User           = Depends(get_current_user),
    db:   Session        = Depends(get_db),
):
    if (msg := _job_limit_error(db, user.id)):
        raise HTTPException(429, msg)
    file_id = await _validate_and_save(request, lease, zip_code)

    with _job_admit_lock:
        with SessionLocal() as jdb:
            if (msg := _job_limit_error(jdb, user.id)):
                (UPLOAD_DIR / f"{file_id}.pdf").unlink(missing_ok=True)
                raise HTTPException(429, msg)
            job = Job(user_id=user.id, status="queued",
                      step=JOB_STEPS["queued"]["label"], progress=JOB_STEPS["queued"]["progress"])
            jdb.add(job)
            jdb.commit()
            job_id = job.id

    _job_executor.submit(_run_job, job_id, file_id, zip_code, user.id, lease.filename)
    return JSONResponse({"job_id": job_id}, status_code=202)


@app.get("/jobs/{job_id}")
def get_job(job_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id, Job.user_id == user.id).first()
    if not job:
        raise HTTPException(404, "Job not found.")
    return {
        "status":        job.status,
        "step":          job.step,
        "progress":      job.progress,
        "error_code":    job.error_code,
        "error_message": job.error_message,
        "lease_id":      job.lease_record_id,
    }
