"""Session-cookie auth for the web app."""

from __future__ import annotations

from typing import Optional

import bcrypt
from fastapi import HTTPException, Request, status
from sqlmodel import Session, select

from extraction.storage import User

# bcrypt truncates inputs longer than 72 bytes; pre-truncate to avoid the error.
_MAX_BYTES = 72


def hash_password(plain: str) -> str:
    pw = plain.encode("utf-8")[:_MAX_BYTES]
    return bcrypt.hashpw(pw, bcrypt.gensalt()).decode("ascii")


def verify_password(plain: str, hashed: str) -> bool:
    try:
        pw = plain.encode("utf-8")[:_MAX_BYTES]
        return bcrypt.checkpw(pw, hashed.encode("ascii"))
    except (ValueError, TypeError):
        return False


def authenticate(session: Session, username: str, password: str) -> Optional[User]:
    user = session.exec(select(User).where(User.username == username)).first()
    if user and verify_password(password, user.password_hash):
        return user
    return None


def current_username(request: Request) -> str:
    """Return the logged-in username or raise 401."""
    username = request.session.get("username")
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="not logged in")
    return username


def current_user_optional(request: Request) -> Optional[str]:
    return request.session.get("username")
