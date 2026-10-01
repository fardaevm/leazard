# auth.py
import os
from datetime import datetime, timedelta, timezone

from jose import JWTError, jwt
from passlib.context import CryptContext

SECRET_KEY            = os.getenv("SECRET_KEY", "").strip()
ALGORITHM             = "HS256"
TOKEN_EXPIRE_MINUTES  = int(os.getenv("TOKEN_EXPIRE_MINUTES", "1440"))  # 24 h

if not SECRET_KEY:
    raise RuntimeError("SECRET_KEY must be set in the environment (no default is provided).")
if TOKEN_EXPIRE_MINUTES <= 0:
    raise RuntimeError("TOKEN_EXPIRE_MINUTES must be a positive integer.")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_token(user_id: int, username: str) -> str:
    payload = {
        "sub": str(user_id),
        "username": username,
        "exp": datetime.now(timezone.utc) + timedelta(minutes=TOKEN_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict | None:
    try:
        payload = jwt.decode(
            token, SECRET_KEY, algorithms=[ALGORITHM],
            options={"require_exp": True, "require_sub": True},
        )
    except JWTError:
        return None
    return payload if str(payload.get("sub", "")).isdigit() else None
