"""
Email-only sign-in. No password.

Reviewers identify themselves by entering an email that's in the User table.
This is **not real authentication** — anyone who knows a reviewer's email
can sign in as them. It exists purely so the audit log records who did what
during a review session.

Why this is OK here:
- The deployment is internal (a small review team).
- The Sheet sync is admin-gated; rogue edits get caught at publish time.
- Real auth (passwords or OAuth) can be layered in later if needed.
"""

from __future__ import annotations

import re
from typing import Optional

from fastapi import HTTPException, Request, status
from sqlmodel import Session, select

from extraction.storage import User

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def normalize_email(email: str) -> str:
    return (email or "").strip().lower()


def authenticate(session: Session, email: str) -> Optional[User]:
    """Return the User if `email` is in the allowlist."""
    email = normalize_email(email)
    if not email or not EMAIL_RE.match(email):
        return None
    return session.exec(select(User).where(User.email == email)).first()


def current_email(request: Request) -> str:
    """Return the signed-in email or raise 401."""
    email = request.session.get("email")
    if not email:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="not signed in")
    return email


def current_email_optional(request: Request) -> Optional[str]:
    return request.session.get("email")


def require_admin(request: Request) -> str:
    """Return the email and 403 if the session is not an admin."""
    email = current_email(request)
    if not request.session.get("is_admin"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="admin only")
    return email
