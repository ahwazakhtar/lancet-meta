"""
FastAPI web app for reviewing extracted effect sizes.

Sign-in is email-only and exists purely to attribute audit-log entries.
See `webapp/auth.py`.

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
from extraction.sheets import pull_from_sheets, push_to_sheets
from extraction.storage import (
    AuditLog,
    EffectSizeRow,
    PaperRow,
    ReviewStatus,
    User,
    get_engine,
    import_from_sheet_rows,
)

from .auth import authenticate, current_email, current_email_optional, require_admin

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


def _log(session: Session, email: str, action: str, target: str, payload: dict | None = None) -> None:
    session.add(AuditLog(email=email, action=action, target=target, payload=payload or {}))


def _editable_paper_fields() -> list[str]:
    return [f for f in paper_field_names() if f not in {"unique_id"}]


def _editable_effect_fields() -> list[str]:
    return effect_size_field_names()


# ---------------------------------------------------------------------------
# Sign-in (email only)
# ---------------------------------------------------------------------------


@app.get("/login", response_class=HTMLResponse)
def login_form(request: Request, error: Optional[str] = None) -> HTMLResponse:
    if current_email_optional(request):
        return RedirectResponse("/", status_code=status.HTTP_303_SEE_OTHER)
    return templates.TemplateResponse(request, "login.html", {"error": error})


@app.post("/login")
def login_submit(
    request: Request,
    email: str = Form(...),
    session: Session = Depends(get_session),
):
    user = authenticate(session, email)
    if not user:
        return RedirectResponse(
            "/login?error=unknown", status_code=status.HTTP_303_SEE_OTHER
        )
    request.session["email"] = user.email
    request.session["display_name"] = user.display_name or user.email
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
    if not current_email_optional(request):
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
            or ql in (p.title or "").lower()
        ]

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
            "display_name": request.session.get("display_name"),
        },
    )


# ---------------------------------------------------------------------------
# Paper detail
# ---------------------------------------------------------------------------


@app.get("/papers/{unique_id}", response_class=HTMLResponse)
def paper_detail(
    request: Request,
    unique_id: str,
    session: Session = Depends(get_session),
):
    current_email(request)
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
            "display_name": request.session.get("display_name"),
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
    email = current_email(request)
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
        paper.last_modified_by = email
        paper.updated_at = datetime.utcnow()
        _log(session, email, "modify", f"paper:{paper.id}", {"changes": changes})

    session.add(paper)
    session.commit()
    return RedirectResponse(f"/papers/{unique_id}", status_code=status.HTTP_303_SEE_OTHER)


@app.post("/papers/{unique_id}/confirm")
def paper_confirm(
    request: Request,
    unique_id: str,
    session: Session = Depends(get_session),
):
    email = current_email(request)
    paper = session.exec(select(PaperRow).where(PaperRow.unique_id == unique_id)).first()
    if not paper:
        raise HTTPException(404)
    paper.status = ReviewStatus.confirmed
    paper.last_modified_by = email
    paper.updated_at = datetime.utcnow()
    _log(session, email, "confirm", f"paper:{paper.id}")
    session.add(paper)
    session.commit()
    return RedirectResponse(f"/papers/{unique_id}", status_code=status.HTTP_303_SEE_OTHER)


@app.post("/papers/{unique_id}/delete")
def paper_delete(
    request: Request,
    unique_id: str,
    session: Session = Depends(get_session),
):
    email = current_email(request)
    paper = session.exec(select(PaperRow).where(PaperRow.unique_id == unique_id)).first()
    if not paper:
        raise HTTPException(404)
    paper.status = ReviewStatus.deleted
    paper.last_modified_by = email
    paper.updated_at = datetime.utcnow()
    _log(session, email, "delete", f"paper:{paper.id}")
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
    email = current_email(request)
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
        es.last_modified_by = email
        es.updated_at = datetime.utcnow()
        _log(session, email, "modify", f"effect_size:{es.id}", {"changes": changes})

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
    email = current_email(request)
    es = session.get(EffectSizeRow, es_id)
    if not es:
        raise HTTPException(404)
    es.status = ReviewStatus.confirmed
    es.last_modified_by = email
    es.updated_at = datetime.utcnow()
    _log(session, email, "confirm", f"effect_size:{es.id}")
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
    email = current_email(request)
    es = session.get(EffectSizeRow, es_id)
    if not es:
        raise HTTPException(404)
    es.status = ReviewStatus.deleted
    es.last_modified_by = email
    es.updated_at = datetime.utcnow()
    _log(session, email, "delete", f"effect_size:{es.id}")
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
    email = current_email(request)
    form = await request.form()
    paper = session.exec(select(PaperRow).where(PaperRow.unique_id == unique_id)).first()
    if not paper:
        raise HTTPException(404)
    kwargs = {f: str(form.get(f, "")) for f in _editable_effect_fields()}
    es = EffectSizeRow(
        paper_unique_id=unique_id,
        status=ReviewStatus.modified,
        last_modified_by=email,
        **kwargs,
    )
    session.add(es)
    session.commit()
    session.refresh(es)
    _log(session, email, "add", f"effect_size:{es.id}", {"created": kwargs})
    session.commit()
    return RedirectResponse(f"/papers/{unique_id}", status_code=status.HTTP_303_SEE_OTHER)


@app.post("/effect-sizes/{es_id}/flag-reextract")
def effect_flag_reextract(
    request: Request,
    es_id: int,
    session: Session = Depends(get_session),
):
    email = current_email(request)
    es = session.get(EffectSizeRow, es_id)
    if not es:
        raise HTTPException(404)
    es.needs_reextraction = True
    es.status = ReviewStatus.needs_reextraction
    es.last_modified_by = email
    es.updated_at = datetime.utcnow()
    _log(session, email, "flag_reextract", f"effect_size:{es.id}")
    session.add(es)
    session.commit()
    return RedirectResponse(
        f"/papers/{es.paper_unique_id}#es-{es.id}", status_code=status.HTTP_303_SEE_OTHER
    )


# ---------------------------------------------------------------------------
# Admin: Sheet sync + user management
# ---------------------------------------------------------------------------


@app.get("/admin", response_class=HTMLResponse)
def admin_panel(
    request: Request,
    msg: Optional[str] = None,
    err: Optional[str] = None,
    session: Session = Depends(get_session),
):
    require_admin(request)
    users = list(session.exec(select(User).order_by(User.email)))
    return templates.TemplateResponse(
        request,
        "admin.html",
        {
            "msg": msg,
            "err": err,
            "display_name": request.session.get("display_name"),
            "sheet_id": os.environ.get("GOOGLE_SHEET_ID", ""),
            "users": users,
        },
    )


@app.post("/admin/users/add")
def admin_add_user(
    request: Request,
    email: str = Form(...),
    display_name: str = Form(""),
    is_admin: bool = Form(False),
    session: Session = Depends(get_session),
):
    actor = require_admin(request)
    from .auth import EMAIL_RE, normalize_email

    norm = normalize_email(email)
    if not EMAIL_RE.match(norm):
        return RedirectResponse(
            f"/admin?err=Invalid+email:+{norm}", status_code=status.HTTP_303_SEE_OTHER
        )
    existing = session.exec(select(User).where(User.email == norm)).first()
    if existing:
        existing.display_name = display_name or existing.display_name
        existing.is_admin = is_admin
        session.add(existing)
        action = "update_user"
    else:
        session.add(User(email=norm, display_name=display_name, is_admin=is_admin))
        action = "add_user"
    _log(session, actor, action, f"user:{norm}", {"is_admin": is_admin, "display_name": display_name})
    session.commit()
    return RedirectResponse(
        f"/admin?msg=User+saved:+{norm}", status_code=status.HTTP_303_SEE_OTHER
    )


@app.post("/admin/users/{user_id}/delete")
def admin_delete_user(
    request: Request,
    user_id: int,
    session: Session = Depends(get_session),
):
    actor = require_admin(request)
    user = session.get(User, user_id)
    if not user:
        raise HTTPException(404)
    if user.email == actor:
        return RedirectResponse(
            "/admin?err=Cannot+remove+yourself", status_code=status.HTTP_303_SEE_OTHER
        )
    deleted_email = user.email
    session.delete(user)
    _log(session, actor, "delete_user", f"user:{deleted_email}")
    session.commit()
    return RedirectResponse(
        f"/admin?msg=Removed+{deleted_email}", status_code=status.HTTP_303_SEE_OTHER
    )


@app.post("/admin/import-from-sheet")
def admin_import(
    request: Request,
    session: Session = Depends(get_session),
):
    actor = require_admin(request)
    try:
        papers, effects = pull_from_sheets()
        n_p, n_e = import_from_sheet_rows(_engine, papers, effects, replace=True)
    except Exception as exc:  # noqa: BLE001
        return RedirectResponse(
            f"/admin?err={str(exc)[:300]}", status_code=status.HTTP_303_SEE_OTHER
        )
    _log(
        session, actor, "import_sheet", "sheet",
        {"papers": n_p, "effect_sizes": n_e},
    )
    session.commit()
    return RedirectResponse(
        f"/admin?msg=Imported+{n_p}+papers+and+{n_e}+effect+sizes",
        status_code=status.HTTP_303_SEE_OTHER,
    )


@app.post("/admin/publish-to-sheet")
def admin_publish(
    request: Request,
    session: Session = Depends(get_session),
):
    actor = require_admin(request)
    try:
        papers = list(
            session.exec(select(PaperRow).where(PaperRow.status != ReviewStatus.deleted))
        )
        effects = list(
            session.exec(select(EffectSizeRow).where(EffectSizeRow.status != ReviewStatus.deleted))
        )
        n_p, n_e = push_to_sheets(papers, effects)
    except Exception as exc:  # noqa: BLE001
        return RedirectResponse(
            f"/admin?err={str(exc)[:300]}", status_code=status.HTTP_303_SEE_OTHER
        )
    _log(
        session, actor, "publish_sheet", "sheet",
        {"papers": n_p, "effect_sizes": n_e},
    )
    session.commit()
    return RedirectResponse(
        f"/admin?msg=Published+{n_p}+papers+and+{n_e}+effect+sizes",
        status_code=status.HTTP_303_SEE_OTHER,
    )


# ---------------------------------------------------------------------------
# Bootstrap: auto-create the first admin from ADMIN_BOOTSTRAP_EMAIL on startup
# so a fresh Railway deploy is usable without shelling in.
# ---------------------------------------------------------------------------


@app.on_event("startup")
def _bootstrap_admin() -> None:
    bootstrap = os.environ.get("ADMIN_BOOTSTRAP_EMAIL", "").strip().lower()
    if not bootstrap:
        return
    with Session(_engine) as session:
        existing_users = session.exec(select(User)).first()
        if existing_users:
            return  # already populated; do nothing
        session.add(User(email=bootstrap, display_name="Bootstrap admin", is_admin=True))
        session.commit()


# ---------------------------------------------------------------------------
# Health check (Railway hits this)
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> dict:
    return {"ok": True}


def serve() -> None:
    """Console entrypoint: `lancet-web`."""
    import uvicorn

    host = os.environ.get("WEB_HOST", "0.0.0.0")
    port = int(os.environ.get("PORT") or os.environ.get("WEB_PORT", "8000"))
    uvicorn.run("webapp.main:app", host=host, port=port, reload=False)
