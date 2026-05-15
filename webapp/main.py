"""
FastAPI web app for reviewing extracted effect sizes.

Run locally:

    uvicorn webapp.main:app --reload

Or via the entrypoint script:

    lancet-web
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Form, HTTPException, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlmodel import Session, select
from starlette.middleware.sessions import SessionMiddleware

from extraction.schema import effect_size_field_names, paper_field_names
from extraction.storage import (
    AuditLog,
    EffectSizeRow,
    PaperRow,
    ReviewStatus,
    User,
    get_engine,
)

from .auth import authenticate, current_user_optional, current_username

load_dotenv()

BASE_DIR = Path(__file__).parent
app = FastAPI(title="Lancet Meta Review")
app.add_middleware(
    SessionMiddleware,
    secret_key=os.environ.get("WEB_SECRET_KEY", "change-me"),
    same_site="lax",
)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

# One shared engine for the process; SQLite is fine for an internal review tool.
_engine = get_engine(os.environ.get("WEB_DB_PATH", "data/app.db"))


def get_session() -> Session:
    with Session(_engine) as session:
        yield session


def _log(session: Session, username: str, action: str, target: str, payload: dict | None = None) -> None:
    session.add(AuditLog(username=username, action=action, target=target, payload=payload or {}))


def _editable_paper_fields() -> list[str]:
    return [f for f in paper_field_names() if f not in {"unique_id"}]


def _editable_effect_fields() -> list[str]:
    return effect_size_field_names()


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


@app.get("/login", response_class=HTMLResponse)
def login_form(request: Request, error: Optional[str] = None) -> HTMLResponse:
    if current_user_optional(request):
        return RedirectResponse("/", status_code=status.HTTP_303_SEE_OTHER)
    return templates.TemplateResponse(
        request, "login.html", {"error": error}
    )


@app.post("/login")
def login_submit(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    session: Session = Depends(get_session),
):
    user = authenticate(session, username, password)
    if not user:
        return RedirectResponse(
            "/login?error=invalid", status_code=status.HTTP_303_SEE_OTHER
        )
    request.session["username"] = user.username
    request.session["is_admin"] = user.is_admin
    return RedirectResponse("/", status_code=status.HTTP_303_SEE_OTHER)


@app.post("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=status.HTTP_303_SEE_OTHER)


# ---------------------------------------------------------------------------
# Paper list
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
def index(
    request: Request,
    q: Optional[str] = None,
    status_filter: Optional[str] = None,
    session: Session = Depends(get_session),
):
    if not current_user_optional(request):
        return RedirectResponse("/login", status_code=status.HTTP_303_SEE_OTHER)

    stmt = select(PaperRow).where(PaperRow.status != ReviewStatus.deleted)
    if status_filter:
        try:
            stmt = stmt.where(PaperRow.status == ReviewStatus(status_filter))
        except ValueError:
            pass
    papers = list(session.exec(stmt.order_by(PaperRow.unique_id)))
    if q:
        ql = q.lower()
        papers = [
            p for p in papers
            if ql in (p.unique_id or "").lower()
            or ql in (p.authors or "").lower()
            or ql in (p.intervention_category or "").lower()
        ]

    # Effect size counts per paper
    es_counts: dict[str, int] = {}
    for row in session.exec(
        select(EffectSizeRow.paper_unique_id).where(EffectSizeRow.status != ReviewStatus.deleted)
    ):
        es_counts[row] = es_counts.get(row, 0) + 1

    return templates.TemplateResponse(
        request,
        "papers.html",
        {
            "papers": papers,
            "es_counts": es_counts,
            "q": q or "",
            "status_filter": status_filter or "",
            "statuses": [s.value for s in ReviewStatus],
            "username": request.session.get("username"),
        },
    )


# ---------------------------------------------------------------------------
# Paper detail (with effect sizes)
# ---------------------------------------------------------------------------


@app.get("/papers/{unique_id}", response_class=HTMLResponse)
def paper_detail(
    request: Request,
    unique_id: str,
    session: Session = Depends(get_session),
):
    current_username(request)
    paper = session.exec(select(PaperRow).where(PaperRow.unique_id == unique_id)).first()
    if not paper:
        raise HTTPException(404)
    effects = list(
        session.exec(
            select(EffectSizeRow)
            .where(EffectSizeRow.paper_unique_id == unique_id)
            .where(EffectSizeRow.status != ReviewStatus.deleted)
            .order_by(EffectSizeRow.id)
        )
    )
    return templates.TemplateResponse(
        request,
        "paper_detail.html",
        {
            "paper": paper,
            "effects": effects,
            "paper_fields": _editable_paper_fields(),
            "effect_fields": _editable_effect_fields(),
            "statuses": [s.value for s in ReviewStatus],
            "username": request.session.get("username"),
        },
    )


# ---------------------------------------------------------------------------
# Paper mutations
# ---------------------------------------------------------------------------


@app.post("/papers/{unique_id}/edit")
async def paper_edit(
    request: Request,
    unique_id: str,
    session: Session = Depends(get_session),
):
    username = current_username(request)
    form = await request.form()
    paper = session.exec(select(PaperRow).where(PaperRow.unique_id == unique_id)).first()
    if not paper:
        raise HTTPException(404)

    changes: dict[str, dict[str, str]] = {}
    for field in _editable_paper_fields():
        if field in form:
            new_val = str(form[field])
            old_val = getattr(paper, field, "")
            if new_val != old_val:
                changes[field] = {"from": old_val, "to": new_val}
                setattr(paper, field, new_val)

    reviewer_notes = form.get("reviewer_notes")
    if reviewer_notes is not None and reviewer_notes != paper.reviewer_notes:
        changes["reviewer_notes"] = {"from": paper.reviewer_notes, "to": str(reviewer_notes)}
        paper.reviewer_notes = str(reviewer_notes)

    needs_reextraction = bool(form.get("needs_reextraction"))
    if needs_reextraction != paper.needs_reextraction:
        changes["needs_reextraction"] = {"from": paper.needs_reextraction, "to": needs_reextraction}
        paper.needs_reextraction = needs_reextraction
        if needs_reextraction:
            paper.status = ReviewStatus.needs_reextraction

    if changes:
        paper.status = ReviewStatus.needs_reextraction if needs_reextraction else ReviewStatus.modified
        paper.last_modified_by = username
        paper.updated_at = datetime.utcnow()
        _log(session, username, "modify", f"paper:{paper.id}", {"changes": changes})

    session.add(paper)
    session.commit()
    return RedirectResponse(f"/papers/{unique_id}", status_code=status.HTTP_303_SEE_OTHER)


@app.post("/papers/{unique_id}/confirm")
def paper_confirm(
    request: Request,
    unique_id: str,
    session: Session = Depends(get_session),
):
    username = current_username(request)
    paper = session.exec(select(PaperRow).where(PaperRow.unique_id == unique_id)).first()
    if not paper:
        raise HTTPException(404)
    paper.status = ReviewStatus.confirmed
    paper.last_modified_by = username
    paper.updated_at = datetime.utcnow()
    _log(session, username, "confirm", f"paper:{paper.id}")
    session.add(paper)
    session.commit()
    return RedirectResponse(f"/papers/{unique_id}", status_code=status.HTTP_303_SEE_OTHER)


@app.post("/papers/{unique_id}/delete")
def paper_delete(
    request: Request,
    unique_id: str,
    session: Session = Depends(get_session),
):
    username = current_username(request)
    paper = session.exec(select(PaperRow).where(PaperRow.unique_id == unique_id)).first()
    if not paper:
        raise HTTPException(404)
    paper.status = ReviewStatus.deleted
    paper.last_modified_by = username
    paper.updated_at = datetime.utcnow()
    _log(session, username, "delete", f"paper:{paper.id}")
    session.add(paper)
    session.commit()
    return RedirectResponse("/", status_code=status.HTTP_303_SEE_OTHER)


# ---------------------------------------------------------------------------
# Effect-size mutations
# ---------------------------------------------------------------------------


@app.post("/effect-sizes/{es_id}/edit")
async def effect_edit(
    request: Request,
    es_id: int,
    session: Session = Depends(get_session),
):
    username = current_username(request)
    form = await request.form()
    es = session.get(EffectSizeRow, es_id)
    if not es:
        raise HTTPException(404)

    changes: dict[str, dict[str, str]] = {}
    for field in _editable_effect_fields():
        if field in form:
            new_val = str(form[field])
            old_val = getattr(es, field, "")
            if new_val != old_val:
                changes[field] = {"from": old_val, "to": new_val}
                setattr(es, field, new_val)

    reviewer_notes = form.get("reviewer_notes")
    if reviewer_notes is not None and reviewer_notes != es.reviewer_notes:
        changes["reviewer_notes"] = {"from": es.reviewer_notes, "to": str(reviewer_notes)}
        es.reviewer_notes = str(reviewer_notes)

    needs_reextraction = bool(form.get("needs_reextraction"))
    if needs_reextraction != es.needs_reextraction:
        changes["needs_reextraction"] = {"from": es.needs_reextraction, "to": needs_reextraction}
        es.needs_reextraction = needs_reextraction

    if changes:
        es.status = (
            ReviewStatus.needs_reextraction if needs_reextraction else ReviewStatus.modified
        )
        es.last_modified_by = username
        es.updated_at = datetime.utcnow()
        _log(session, username, "modify", f"effect_size:{es.id}", {"changes": changes})

    session.add(es)
    session.commit()
    return RedirectResponse(
        f"/papers/{es.paper_unique_id}#es-{es.id}", status_code=status.HTTP_303_SEE_OTHER
    )


@app.post("/effect-sizes/{es_id}/confirm")
def effect_confirm(
    request: Request,
    es_id: int,
    session: Session = Depends(get_session),
):
    username = current_username(request)
    es = session.get(EffectSizeRow, es_id)
    if not es:
        raise HTTPException(404)
    es.status = ReviewStatus.confirmed
    es.last_modified_by = username
    es.updated_at = datetime.utcnow()
    _log(session, username, "confirm", f"effect_size:{es.id}")
    session.add(es)
    session.commit()
    return RedirectResponse(
        f"/papers/{es.paper_unique_id}#es-{es.id}", status_code=status.HTTP_303_SEE_OTHER
    )


@app.post("/effect-sizes/{es_id}/delete")
def effect_delete(
    request: Request,
    es_id: int,
    session: Session = Depends(get_session),
):
    username = current_username(request)
    es = session.get(EffectSizeRow, es_id)
    if not es:
        raise HTTPException(404)
    es.status = ReviewStatus.deleted
    es.last_modified_by = username
    es.updated_at = datetime.utcnow()
    _log(session, username, "delete", f"effect_size:{es.id}")
    session.add(es)
    session.commit()
    return RedirectResponse(
        f"/papers/{es.paper_unique_id}", status_code=status.HTTP_303_SEE_OTHER
    )


@app.post("/papers/{unique_id}/effect-sizes/new")
async def effect_create(
    request: Request,
    unique_id: str,
    session: Session = Depends(get_session),
):
    username = current_username(request)
    form = await request.form()
    paper = session.exec(select(PaperRow).where(PaperRow.unique_id == unique_id)).first()
    if not paper:
        raise HTTPException(404)
    kwargs = {f: str(form.get(f, "")) for f in _editable_effect_fields()}
    es = EffectSizeRow(
        paper_unique_id=unique_id,
        status=ReviewStatus.modified,
        last_modified_by=username,
        **kwargs,
    )
    session.add(es)
    session.commit()
    session.refresh(es)
    _log(session, username, "add", f"effect_size:{es.id}", {"created": kwargs})
    session.commit()
    return RedirectResponse(f"/papers/{unique_id}", status_code=status.HTTP_303_SEE_OTHER)


@app.post("/effect-sizes/{es_id}/flag-reextract")
def effect_flag_reextract(
    request: Request,
    es_id: int,
    session: Session = Depends(get_session),
):
    username = current_username(request)
    es = session.get(EffectSizeRow, es_id)
    if not es:
        raise HTTPException(404)
    es.needs_reextraction = True
    es.status = ReviewStatus.needs_reextraction
    es.last_modified_by = username
    es.updated_at = datetime.utcnow()
    _log(session, username, "flag_reextract", f"effect_size:{es.id}")
    session.add(es)
    session.commit()
    return RedirectResponse(
        f"/papers/{es.paper_unique_id}#es-{es.id}", status_code=status.HTTP_303_SEE_OTHER
    )


def serve() -> None:
    """Console entrypoint: `lancet-web`."""
    import uvicorn

    host = os.environ.get("WEB_HOST", "127.0.0.1")
    port = int(os.environ.get("WEB_PORT", "8000"))
    uvicorn.run("webapp.main:app", host=host, port=port, reload=False)
