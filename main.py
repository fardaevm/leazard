# main.py
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import asyncio
import json
import logging
import uuid
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Depends, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy.orm import Session

from utils.extract_pdf import extract_text_from_pdf, looks_scanned, PDF_MAGIC
from rag.indexer import ensure_index
from agent import build_app
from db import create_tables, get_db, User, LeaseRecord
from auth import hash_password, verify_password, create_token, decode_token

log = logging.getLogger("leazard.api")

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads"))
BASE_DIR   = Path(__file__).resolve().parent
MAX_UPLOAD_MB     = float(os.getenv("MAX_UPLOAD_MB", "10"))
MAX_UPLOAD_BYTES  = int(MAX_UPLOAD_MB * 1024 * 1024)
UPLOAD_READ_CHUNK = 1024 * 1024

app = FastAPI(title="Leaze")
app.mount("/ui", StaticFiles(directory=str(BASE_DIR / "ui")), name="ui")

# Startup
ensure_index()
create_tables()
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


@app.post("/analyze")
async def analyze(
    request:  Request,
    lease:    UploadFile = File(...),
    zip_code: str        = Form(...),
    user: User           = Depends(get_current_user),
    db:   Session        = Depends(get_db),
):
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
        try:
            text = extract_text_from_pdf(pdf_path)
        except Exception:
            raise HTTPException(422, "Could not read this PDF.")
        if looks_scanned(text):
            raise HTTPException(
                422, "This looks like a scanned PDF. Scanned leases aren't supported yet."
            )

        loop  = asyncio.get_running_loop()
        try:
            state = await loop.run_in_executor(
                None, lambda: graph_app.invoke({"zip_code": zip_code, "lease_text": text})
            )
        except Exception as e:
            # Type only: exception messages/tracebacks may contain lease text.
            log.error("Lease analysis failed: %s", type(e).__name__)
            raise HTTPException(502, "Lease analysis failed. Please try again.")
        if state.get("status") == "error":
            raise HTTPException(502, state.get("message") or "Lease analysis failed.")

        lease_json = state.get("lease_json") or {}
        risk_json  = state.get("risk_json")  or {}
        address    = (lease_json.get("address_or_city_if_present") or "").strip() or f"ZIP {zip_code}"

        record = LeaseRecord(
            user_id           = user.id,
            file_id           = file_id,
            original_filename = lease.filename,
            address           = address,
            zip_code          = zip_code,
            risk_score        = risk_json.get("risk_score"),
            lease_json        = json.dumps(lease_json),
            risk_json         = json.dumps(risk_json),
            letter_text       = state.get("letter_text"),
        )
        db.add(record)
        db.commit()
        db.refresh(record)
    except BaseException:
        pdf_path.unlink(missing_ok=True)
        raise

    # Exclude raw lease_text from the response (can be megabytes)
    return JSONResponse({
        "id":          record.id,
        "file_id":     file_id,
        "zip_code":    zip_code,
        "status":      state.get("status"),
        "message":     state.get("message"),
        "lease_json":  lease_json,
        "risk_json":   risk_json,
        "letter_text": state.get("letter_text"),
    })
