# db.py
import os
import json
import uuid
from datetime import datetime
from pathlib import Path

from sqlalchemy import (
    create_engine, event, inspect, text, Column, Integer, String, Text, Float, DateTime, ForeignKey,
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

_DB_PATH = Path(__file__).resolve().parent / "leaze.db"
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{_DB_PATH}")
_IS_SQLITE = DATABASE_URL.startswith("sqlite")

engine = create_engine(
    DATABASE_URL,
    # Sessions are never shared across threads; each worker opens its own.
    connect_args={"check_same_thread": False, "timeout": 30} if _IS_SQLITE else {},
)

if _IS_SQLITE:
    @event.listens_for(engine, "connect")
    def _sqlite_pragmas(dbapi_conn, _record):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA busy_timeout=30000")
        cur.close()

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id              = Column(Integer, primary_key=True, index=True)
    username        = Column(String(64), unique=True, nullable=False, index=True)
    hashed_password = Column(String(256), nullable=False)
    created_at      = Column(DateTime, default=datetime.utcnow)

    leases = relationship("LeaseRecord", back_populates="user", cascade="all, delete-orphan")


class LeaseRecord(Base):
    __tablename__ = "leases"

    id                = Column(Integer, primary_key=True, index=True)
    user_id           = Column(Integer, ForeignKey("users.id"), nullable=False)
    file_id           = Column(String(64), nullable=False)
    original_filename = Column(String(256))
    address           = Column(String(256))   # from lease_json.address_or_city_if_present
    zip_code          = Column(String(10))
    risk_score        = Column(Float)
    lease_json        = Column(Text)          # JSON-serialized dict
    risk_json         = Column(Text)          # JSON-serialized dict
    letter_text       = Column(Text)
    created_at        = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="leases")


JOB_ACTIVE_STATUSES = ("queued", "running")


class Job(Base):
    __tablename__ = "jobs"

    id              = Column(String(32), primary_key=True, default=lambda: uuid.uuid4().hex)
    user_id         = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    status          = Column(String(16), nullable=False, default="queued", index=True)  # queued|running|done|error
    step            = Column(String(64), nullable=False, default="")
    progress        = Column(Integer, nullable=False, default=0)                       # 0-100
    error_code      = Column(String(32), nullable=True)
    error_message   = Column(String(512), nullable=True)
    lease_record_id = Column(Integer, ForeignKey("leases.id"), nullable=True)
    created_at      = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at      = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


def _add_missing_columns() -> None:
    """Additive-only migration: ALTER TABLE ADD COLUMN for model columns absent from the DB."""
    insp = inspect(engine)
    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            if not insp.has_table(table.name):
                continue
            existing = {c["name"] for c in insp.get_columns(table.name)}
            for col in table.columns:
                if col.name in existing:
                    continue
                ddl = col.type.compile(dialect=engine.dialect)
                conn.execute(text(f'ALTER TABLE "{table.name}" ADD COLUMN "{col.name}" {ddl}'))


def create_tables() -> None:
    Base.metadata.create_all(bind=engine)
    _add_missing_columns()


def mark_interrupted_jobs() -> int:
    """Jobs left queued/running by a previous process can never finish; fail them."""
    with SessionLocal() as db:
        n = (
            db.query(Job)
            .filter(Job.status.in_(JOB_ACTIVE_STATUSES))
            .update(
                {
                    Job.status: "error",
                    Job.error_code: "interrupted",
                    Job.error_message: "The server restarted while your lease was being analyzed. Please try again.",
                    Job.updated_at: datetime.utcnow(),
                },
                synchronize_session=False,
            )
        )
        db.commit()
    return n


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
