# db.py
import os
import json
from datetime import datetime
from pathlib import Path

from sqlalchemy import (
    create_engine, Column, Integer, String, Text, Float, DateTime, ForeignKey,
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

_DB_PATH = Path(__file__).resolve().parent / "leaze.db"
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{_DB_PATH}")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)
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


def create_tables() -> None:
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
