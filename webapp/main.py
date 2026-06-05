"""
FastAPI web app for reviewing extracted effect sizes.

Sign-in is email-only and exists purely to attribute audit-log entries.
See `webapp/auth.py`.

The review UI is a 4-step flow per paper:

  1. Tables     — declare which markdown tables contain effect sizes.
  2. Outcomes   — confirm the outcomes reported in each declared table.
  3. Timepoints — confirm baseline / endline / etc.
  4. Estimates  — confirm the actual effect-size estimates.

Paper-level fields are edited on a separate "Info" view.

Run locally:

    uvicorn webapp.main:app --reload
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
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

from extraction.schema import FIELD_OPTIONS, effect_size_field_names, paper_field_names
from extraction.sheets import pull_from_sheets, push_to_sheets
from extraction.storage import (
    AuditLog,
    EffectSizeRow,
    PaperReview,
    PaperRow,
    PaperTable,
    ReviewStatus,
    TableOutcome,
    TableTimepoint,
    User,
    get_engine,
    import_from_sheet_rows,
    import_paper_from_sheet,
)


MAX_REVIEWERS_PER_PAPER = 2


def _reviewers_for(session: Session, unique_id: str) -> list[str]:
    """Distinct reviewer emails who have marked this paper as reviewed."""
    rows = session.exec(
        select(PaperReview.reviewer_email)
        .where(PaperReview.paper_unique_id == unique_id)
    )
    return sorted({r for r in rows})


def _record_review(session: Session, unique_id: str, reviewer_email: str) -> bool:
    """Insert a PaperReview row if this reviewer hasn't completed before.

    Returns True if a new review was recorded, False if it was already there.
    """
    existing = session.exec(
        select(PaperReview)
        .where(PaperReview.paper_unique_id == unique_id)
        .where(PaperReview.reviewer_email == reviewer_email)
    ).first()
    if existing:
        return False
    session.add(PaperReview(
        paper_unique_id=unique_id,
        reviewer_email=reviewer_email,
    ))
    return True

from .auth import authenticate, current_email, current_email_optional, require_admin

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("webapp")

BASE_DIR = Path(__file__).parent

_DB_PATH = os.environ.get("WEB_DB_PATH", "data/app.db")
_engine = get_engine(_DB_PATH)


STEP_ORDER = ["tables", "outcomes", "timepoints", "estimates"]
STEP_LABELS = {
    "info": "Paper info",
    "tables": "1. Tables",
    "outcomes": "2. Outcomes",
    "timepoints": "3. Timepoints",
    "estimates": "4. Estimates",
}


def _ensure_bootstrap_admin() -> None:
    bootstrap = os.environ.get("ADMIN_BOOTSTRAP_EMAIL", "").strip().lower()
    if not bootstrap:
        log.warning("ADMIN_BOOTSTRAP_EMAIL not set; no bootstrap admin will be created.")
        return
    with Session(_engine) as session:
        existing = session.exec(select(User).where(User.email == bootstrap)).first()
        if existing:
            if not existing.is_admin:
                existing.is_admin = True
                session.add(existing)
                session.commit()
                log.info("Bootstrap: promoted existing user %s to admin.", bootstrap)
            else:
                log.info("Bootstrap: %s already an admin; nothing to do.", bootstrap)
            return
        session.add(User(email=bootstrap, display_name="Bootstrap admin", is_admin=True))
        session.commit()
        log.info("Bootstrap: created admin user %s.", bootstrap)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting webapp. db_path=%s", _DB_PATH)
    _ensure_bootstrap_admin()
    yield


app = FastAPI(title="Lancet Meta Review", lifespan=lifespan)
app.add_middleware(
    SessionMiddleware,
    secret_key=os.environ.get("WEB_SECRET_KEY", "change-me"),
    same_site="lax",
)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")
# Expose field-option dropdowns to every template so step_info and
# step_estimates can render <select> elements instead of free-text inputs
# for constrained fields.
templates.env.globals["field_options"] = FIELD_OPTIONS


def get_session() -> Session:
    with Session(_engine) as session:
        yield session


def _log(session: Session, email: str, action: str, target: str, payload: dict | None = None) -> None:
    session.add(AuditLog(email=email, action=action, target=target, payload=payload or {}))


def _editable_paper_fields() -> list[str]:
    return [f for f in paper_field_names() if f not in {"unique_id"}]


def _editable_effect_fields() -> list[str]:
    return effect_size_field_names()


def _is_admin(request: Request) -> bool:
    return bool(request.session.get("is_admin"))


def _require_checkout(request: Request, paper: PaperRow) -> None:
    """Block edits unless the paper is checked out by the caller (admins bypass)."""
    email = current_email(request)
    if _is_admin(request):
        return
    if paper.checked_out_by != email:
        holder = paper.checked_out_by or "nobody"
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Check out this paper before editing. Currently held by: {holder}.",
        )


def _require_draft_access(request: Request, paper: PaperRow) -> Optional[RedirectResponse]:
    """While a paper is checked out, the draft is private to the holder.

    Returns a redirect-to-dashboard if the caller can't access the draft;
    otherwise None. Admins always see; non-holders are bounced.
    """
    me = current_email(request)
    if paper.checked_out_by and paper.checked_out_by != me and not _is_admin(request):
        return RedirectResponse(
            "/?err=Paper+is+being+drafted+by+another+reviewer",
            status_code=status.HTTP_303_SEE_OTHER,
        )
    return None


def _get_paper_or_404(session: Session, unique_id: str) -> PaperRow:
    paper = session.exec(select(PaperRow).where(PaperRow.unique_id == unique_id)).first()
    if not paper:
        raise HTTPException(404)
    return paper


def _shell_context(request: Request, paper: PaperRow, step: str,
                    msg: Optional[str], err: Optional[str]) -> dict:
    me = current_email(request)
    held_by_me = paper.checked_out_by == me
    return {
        "paper": paper,
        "step": step,
        "step_labels": STEP_LABELS,
        "step_order": ["info"] + STEP_ORDER,
        "display_name": request.session.get("display_name"),
        "me": me,
        "held_by_me": held_by_me,
        "can_edit": held_by_me or _is_admin(request),
        "is_admin": _is_admin(request),
        "msg": msg,
        "err": err,
    }


def _redirect_step(unique_id: str, step: str, msg: Optional[str] = None, err: Optional[str] = None,
                   anchor: Optional[str] = None) -> RedirectResponse:
    target = f"/papers/{unique_id}/{step}"
    qs = []
    if msg:
        qs.append(f"msg={msg}")
    if err:
        qs.append(f"err={err}")
    if qs:
        target = f"{target}?{'&'.join(qs)}"
    if anchor:
        target = f"{target}#{anchor}"
    return RedirectResponse(target, status_code=status.HTTP_303_SEE_OTHER)


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
    from .auth import EMAIL_RE, normalize_email

    norm = normalize_email(email)
    if not norm or not EMAIL_RE.match(norm):
        return RedirectResponse(
            "/login?error=invalid", status_code=status.HTTP_303_SEE_OTHER
        )

    user = authenticate(session, email)
    if not user:
        bootstrap = os.environ.get("ADMIN_BOOTSTRAP_EMAIL", "").strip().lower()
        if bootstrap and norm == bootstrap:
            log.info("Bootstrap fallback at login: creating admin %s", bootstrap)
            _ensure_bootstrap_admin()
            user = authenticate(session, email)
        if not user:
            # Open sign-up: any well-formed email becomes a reviewer.
            # No password, attribution-only — matches the existing model.
            display = norm.split("@")[0]
            user = User(email=norm, display_name=display, is_admin=False)
            session.add(user)
            session.commit()
            session.refresh(user)
            log.info("Auto-created reviewer on first login: %s", norm)

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
    checkout_filter: Optional[str] = None,
    msg: Optional[str] = None,
    err: Optional[str] = None,
    session: Session = Depends(get_session),
):
    me = current_email_optional(request)
    if not me:
        return RedirectResponse("/login", status_code=status.HTTP_303_SEE_OTHER)

    stmt = select(PaperRow).where(PaperRow.status != ReviewStatus.deleted)
    if status_filter:
        try:
            stmt = stmt.where(PaperRow.status == ReviewStatus(status_filter))
        except ValueError:
            pass
    if checkout_filter == "mine":
        stmt = stmt.where(PaperRow.checked_out_by == me)
    elif checkout_filter == "held":
        stmt = stmt.where(PaperRow.checked_out_by.is_not(None))
    elif checkout_filter == "available":
        stmt = stmt.where(PaperRow.checked_out_by.is_(None))

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

    # Per-paper review progress: distinct reviewer emails that have clicked
    # Done.
    review_emails: dict[str, list[str]] = {}
    for paper_uid, reviewer in session.exec(
        select(PaperReview.paper_unique_id, PaperReview.reviewer_email)
    ):
        review_emails.setdefault(paper_uid, [])
        if reviewer not in review_emails[paper_uid]:
            review_emails[paper_uid].append(reviewer)

    # Aggregate re-extract flags from child rows so the dashboard badge
    # appears whenever a paper has at least one flagged estimate / outcome
    # / table, not only when the paper-level checkbox was ticked.
    papers_needing_reextract: set[str] = {
        row for row in session.exec(
            select(EffectSizeRow.paper_unique_id)
            .where(EffectSizeRow.needs_reextraction.is_(True))
            .where(EffectSizeRow.status != ReviewStatus.deleted)
        )
    }

    return templates.TemplateResponse(
        request,
        "papers.html",
        {
            "papers": papers,
            "es_counts": es_counts,
            "review_emails": review_emails,
            "max_reviewers": MAX_REVIEWERS_PER_PAPER,
            "papers_needing_reextract": papers_needing_reextract,
            "q": q or "",
            "status_filter": status_filter or "",
            "checkout_filter": checkout_filter or "",
            "statuses": [s.value for s in ReviewStatus],
            "display_name": request.session.get("display_name"),
            "me": me,
            "msg": msg,
            "err": err,
        },
    )


# ---------------------------------------------------------------------------
# Paper review: entry redirect + checkout/checkin/import
# ---------------------------------------------------------------------------


@app.get("/papers/{unique_id}", response_class=HTMLResponse)
def paper_entry(request: Request, unique_id: str):
    return RedirectResponse(
        f"/papers/{unique_id}/tables", status_code=status.HTTP_303_SEE_OTHER
    )


@app.post("/papers/{unique_id}/checkout")
def paper_checkout(
    request: Request,
    unique_id: str,
    next: Optional[str] = Form(None),
    session: Session = Depends(get_session),
):
    email = current_email(request)
    paper = _get_paper_or_404(session, unique_id)
    if paper.checked_out_by and paper.checked_out_by != email:
        return _redirect_step(unique_id, next or "tables",
                              err=f"Already+checked+out+by+{paper.checked_out_by}")
    # Dual-review cap: once two distinct reviewers have marked the paper as
    # reviewed, only those reviewers (or admins) can check it out again.
    reviewers = _reviewers_for(session, unique_id)
    if (
        len(reviewers) >= MAX_REVIEWERS_PER_PAPER
        and email not in reviewers
        and not _is_admin(request)
    ):
        return RedirectResponse(
            f"/?err=Paper+already+reviewed+by+2+reviewers+({'+%26+'.join(reviewers)})",
            status_code=status.HTTP_303_SEE_OTHER,
        )
    paper.checked_out_by = email
    paper.checked_out_at = datetime.utcnow()
    _log(session, email, "checkout", f"paper:{paper.id}")
    session.add(paper)
    session.commit()
    # New checkouts land on Paper Info so reviewers see the bibliographic
    # context first; from there they can move to Tables / Outcomes / etc.
    return _redirect_step(unique_id, next or "info")


@app.post("/papers/{unique_id}/submit")
def paper_submit(
    request: Request,
    unique_id: str,
    session: Session = Depends(get_session),
):
    """Done: release the lock + record this reviewer as having reviewed.

    Counts toward the 2-reviewer cap. Idempotent — re-clicking Done after
    a re-checkout doesn't insert a duplicate review. Blocked if any table
    is still undecided (is_effect_size IS NULL).
    """
    email = current_email(request)
    paper = _get_paper_or_404(session, unique_id)
    if paper.checked_out_by != email and not _is_admin(request):
        raise HTTPException(403, "Only the draft owner can submit.")

    # Gate: every parsed table must be classified before submitting.
    undecided = session.exec(
        select(PaperTable)
        .where(PaperTable.paper_unique_id == unique_id)
        .where(PaperTable.is_effect_size.is_(None))
        .where(PaperTable.status != ReviewStatus.deleted)
    ).all()
    if undecided:
        labels = ", ".join(t.table_label for t in undecided[:4])
        more = "" if len(undecided) <= 4 else f"+{len(undecided)-4}+more"
        return _redirect_step(
            unique_id, "tables",
            err=f"Classify+all+tables+before+marking+Done+(undecided:+{labels}{more})",
        )

    was_held_by = paper.checked_out_by
    paper.checked_out_by = None
    paper.checked_out_at = None
    paper.last_modified_by = email
    paper.updated_at = datetime.utcnow()
    recorded = False
    if was_held_by:
        recorded = _record_review(session, unique_id, was_held_by)
    _log(session, email, "submit", f"paper:{paper.id}", {
        "was_held_by": was_held_by,
        "review_recorded": recorded,
    })
    session.add(paper)
    session.commit()
    return RedirectResponse(
        f"/?msg=Marked+as+reviewed:+{unique_id}", status_code=status.HTTP_303_SEE_OTHER
    )


@app.post("/papers/{unique_id}/checkin")
def paper_checkin(
    request: Request,
    unique_id: str,
    next: Optional[str] = Form(None),
    session: Session = Depends(get_session),
):
    """Release the lock without recording a review.

    Use this when a reviewer checked out by mistake or wants to step away
    without committing to one of the 2 review slots. Admins can also use
    this to force-release someone else's checkout.
    """
    email = current_email(request)
    paper = _get_paper_or_404(session, unique_id)
    if paper.checked_out_by != email and not _is_admin(request):
        raise HTTPException(403, "Only the holder or an admin can release this.")
    was_held_by = paper.checked_out_by
    paper.checked_out_by = None
    paper.checked_out_at = None
    _log(session, email, "checkin", f"paper:{paper.id}", {"was_held_by": was_held_by})
    session.add(paper)
    session.commit()
    # Admins force-releasing someone else's draft stay on the paper page so
    # they can keep working / inspect. The holder goes back to the dashboard
    # because they intentionally relinquished access.
    if was_held_by and was_held_by != email:
        return _redirect_step(unique_id, next or "tables",
                              msg=f"Released+{was_held_by}'s+checkout")
    return RedirectResponse(
        f"/?msg=Released+{unique_id}", status_code=status.HTTP_303_SEE_OTHER
    )


@app.post("/papers/{unique_id}/import-from-sheet")
def paper_import_one(
    request: Request,
    unique_id: str,
    session: Session = Depends(get_session),
):
    email = current_email(request)
    paper = _get_paper_or_404(session, unique_id)
    if paper.checked_out_by != email:
        return _redirect_step(unique_id, "tables", err="Check+out+this+paper+before+importing")
    try:
        papers, effects, tables, outcomes, timepoints, reviews = pull_from_sheets()
    except Exception as exc:  # noqa: BLE001
        return _redirect_step(unique_id, "tables", err=str(exc)[:200])
    found, n_e = import_paper_from_sheet(
        _engine, unique_id, papers, effects,
        table_rows=tables, outcome_rows=outcomes, timepoint_rows=timepoints,
        review_rows=reviews,
    )
    if not found:
        return _redirect_step(unique_id, "tables", err="Paper+not+found+in+Sheet")
    _log(session, email, "import_paper", f"paper:{paper.id}", {"effect_sizes": n_e})
    session.commit()
    return _redirect_step(unique_id, "tables", msg=f"Refreshed+from+Sheet+(%2B{n_e}+effect+sizes)")


# ---------------------------------------------------------------------------
# Paper info view (paper-level fields)
# ---------------------------------------------------------------------------


@app.get("/papers/{unique_id}/info", response_class=HTMLResponse)
def paper_info(
    request: Request,
    unique_id: str,
    msg: Optional[str] = None,
    err: Optional[str] = None,
    session: Session = Depends(get_session),
):
    current_email(request)
    paper = _get_paper_or_404(session, unique_id)
    if (blocked := _require_draft_access(request, paper)):
        return blocked
    ctx = _shell_context(request, paper, "info", msg, err)
    ctx.update({
        "paper_fields": _editable_paper_fields(),
    })
    return templates.TemplateResponse(request, "step_info.html", ctx)


@app.post("/papers/{unique_id}/edit")
async def paper_edit(
    request: Request,
    unique_id: str,
    session: Session = Depends(get_session),
):
    email = current_email(request)
    form = await request.form()
    paper = _get_paper_or_404(session, unique_id)
    _require_checkout(request, paper)

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

    if changes:
        paper.status = ReviewStatus.modified
        paper.last_modified_by = email
        paper.updated_at = datetime.utcnow()
        _log(session, email, "modify", f"paper:{paper.id}", {"changes": changes})

    session.add(paper)
    session.commit()
    return _redirect_step(unique_id, "info")


@app.post("/papers/{unique_id}/delete")
def paper_delete(
    request: Request,
    unique_id: str,
    session: Session = Depends(get_session),
):
    email = current_email(request)
    paper = _get_paper_or_404(session, unique_id)
    _require_checkout(request, paper)
    paper.status = ReviewStatus.deleted
    paper.last_modified_by = email
    paper.updated_at = datetime.utcnow()
    _log(session, email, "delete", f"paper:{paper.id}")
    session.add(paper)
    session.commit()
    return RedirectResponse("/", status_code=status.HTTP_303_SEE_OTHER)


# ---------------------------------------------------------------------------
# Step 1 — Tables
# ---------------------------------------------------------------------------


def _list_paper_tables(session: Session, unique_id: str) -> list[PaperTable]:
    return list(
        session.exec(
            select(PaperTable)
            .where(PaperTable.paper_unique_id == unique_id)
            .where(PaperTable.status != ReviewStatus.deleted)
            .order_by(PaperTable.page, PaperTable.table_index, PaperTable.id)
        )
    )


@app.get("/papers/{unique_id}/tables", response_class=HTMLResponse)
def step_tables(
    request: Request,
    unique_id: str,
    msg: Optional[str] = None,
    err: Optional[str] = None,
    session: Session = Depends(get_session),
):
    current_email(request)
    paper = _get_paper_or_404(session, unique_id)
    if (blocked := _require_draft_access(request, paper)):
        return blocked
    tables = _list_paper_tables(session, unique_id)
    ctx = _shell_context(request, paper, "tables", msg, err)
    ctx.update({"tables": tables})
    return templates.TemplateResponse(request, "step_tables.html", ctx)


@app.post("/papers/{unique_id}/tables/{table_id}/set-effect-size")
def table_set_effect_size(
    request: Request,
    unique_id: str,
    table_id: int,
    is_effect_size: str = Form(...),  # "true" / "false" / "none"
    session: Session = Depends(get_session),
):
    email = current_email(request)
    paper = _get_paper_or_404(session, unique_id)
    _require_checkout(request, paper)
    table = session.get(PaperTable, table_id)
    if not table or table.paper_unique_id != unique_id:
        raise HTTPException(404)
    v = is_effect_size.strip().lower()
    if v == "true":
        table.is_effect_size = True
        action = "declare_table"
    elif v == "false":
        table.is_effect_size = False
        action = "undeclare_table"
    else:
        table.is_effect_size = None
        action = "reset_table"
    table.last_modified_by = email
    table.updated_at = datetime.utcnow()
    _log(session, email, action, f"table:{table.id}", {"is_effect_size": table.is_effect_size})
    session.add(table)
    session.commit()
    return _redirect_step(unique_id, "tables", anchor=f"t-{table.id}")


@app.post("/papers/{unique_id}/tables/new")
def table_create_manual(
    request: Request,
    unique_id: str,
    table_label: str = Form(...),
    session: Session = Depends(get_session),
):
    email = current_email(request)
    paper = _get_paper_or_404(session, unique_id)
    _require_checkout(request, paper)
    label = table_label.strip()
    if not label:
        return _redirect_step(unique_id, "tables", err="Table+label+required")
    table = PaperTable(
        paper_unique_id=unique_id,
        table_label=label,
        page=0,
        table_index=0,
        body_markdown="",
        is_effect_size=True,
        is_manual=True,
        last_modified_by=email,
    )
    session.add(table)
    session.commit()
    session.refresh(table)
    _log(session, email, "add_table_manual", f"table:{table.id}", {"label": label})
    session.commit()
    return _redirect_step(unique_id, "tables", anchor=f"t-{table.id}")


@app.post("/papers/{unique_id}/tables/{table_id}/delete")
def table_delete(
    request: Request,
    unique_id: str,
    table_id: int,
    session: Session = Depends(get_session),
):
    email = current_email(request)
    paper = _get_paper_or_404(session, unique_id)
    _require_checkout(request, paper)
    table = session.get(PaperTable, table_id)
    if not table or table.paper_unique_id != unique_id:
        raise HTTPException(404)
    table.status = ReviewStatus.deleted
    table.last_modified_by = email
    table.updated_at = datetime.utcnow()
    _log(session, email, "delete_table", f"table:{table.id}", {"label": table.table_label})
    session.add(table)
    session.commit()
    return _redirect_step(unique_id, "tables")


# ---------------------------------------------------------------------------
# Step 2 — Outcomes
# ---------------------------------------------------------------------------


def _effect_size_tables(session: Session, unique_id: str) -> list[PaperTable]:
    return list(
        session.exec(
            select(PaperTable)
            .where(PaperTable.paper_unique_id == unique_id)
            .where(PaperTable.is_effect_size.is_(True))
            .where(PaperTable.status != ReviewStatus.deleted)
            .order_by(PaperTable.page, PaperTable.table_index, PaperTable.id)
        )
    )


def _outcomes_for_tables(session: Session, table_ids: list[int]) -> dict[int, list[TableOutcome]]:
    if not table_ids:
        return {}
    rows = session.exec(
        select(TableOutcome)
        .where(TableOutcome.table_id.in_(table_ids))
        .where(TableOutcome.status != ReviewStatus.deleted)
        .order_by(TableOutcome.id)
    )
    out: dict[int, list[TableOutcome]] = {tid: [] for tid in table_ids}
    for r in rows:
        out.setdefault(r.table_id, []).append(r)
    return out


def _timepoints_for_tables(session: Session, table_ids: list[int]) -> dict[int, list[TableTimepoint]]:
    if not table_ids:
        return {}
    rows = session.exec(
        select(TableTimepoint)
        .where(TableTimepoint.table_id.in_(table_ids))
        .where(TableTimepoint.status != ReviewStatus.deleted)
        .order_by(TableTimepoint.id)
    )
    out: dict[int, list[TableTimepoint]] = {tid: [] for tid in table_ids}
    for r in rows:
        out.setdefault(r.table_id, []).append(r)
    return out


@app.get("/papers/{unique_id}/outcomes", response_class=HTMLResponse)
def step_outcomes(
    request: Request,
    unique_id: str,
    msg: Optional[str] = None,
    err: Optional[str] = None,
    session: Session = Depends(get_session),
):
    current_email(request)
    paper = _get_paper_or_404(session, unique_id)
    if (blocked := _require_draft_access(request, paper)):
        return blocked
    tables = _effect_size_tables(session, unique_id)
    table_ids = [t.id for t in tables if t.id is not None]
    outcomes = _outcomes_for_tables(session, table_ids)
    ctx = _shell_context(request, paper, "outcomes", msg, err)
    ctx.update({"tables": tables, "outcomes_by_table": outcomes, "show_save_bar": True})
    return templates.TemplateResponse(request, "step_outcomes.html", ctx)


@app.post("/papers/{unique_id}/tables/{table_id}/outcomes/new")
async def outcome_create(
    request: Request,
    unique_id: str,
    table_id: int,
    session: Session = Depends(get_session),
):
    email = current_email(request)
    paper = _get_paper_or_404(session, unique_id)
    _require_checkout(request, paper)
    form = await request.form()
    outcome = TableOutcome(
        table_id=table_id,
        paper_unique_id=unique_id,
        outcome_name=str(form.get("outcome_name", "")),
        outcome_domain=str(form.get("outcome_domain", "")),
        outcome_definition=str(form.get("outcome_definition", "")),
        status=ReviewStatus.modified,
        last_modified_by=email,
    )
    session.add(outcome)
    session.commit()
    session.refresh(outcome)
    _log(session, email, "add_outcome", f"outcome:{outcome.id}",
         {"table_id": table_id, "name": outcome.outcome_name})
    session.commit()
    return _redirect_step(unique_id, "outcomes", anchor=f"o-{outcome.id}")


@app.post("/outcomes/{outcome_id}/edit")
async def outcome_edit(
    request: Request,
    outcome_id: int,
    session: Session = Depends(get_session),
):
    email = current_email(request)
    form = await request.form()
    outcome = session.get(TableOutcome, outcome_id)
    if not outcome:
        raise HTTPException(404)
    paper = _get_paper_or_404(session, outcome.paper_unique_id)
    _require_checkout(request, paper)
    changes: dict[str, dict[str, str]] = {}
    for field in ("outcome_name", "outcome_domain", "outcome_definition", "reviewer_notes"):
        if field in form:
            new_val = str(form[field])
            old_val = getattr(outcome, field, "")
            if new_val != old_val:
                changes[field] = {"from": old_val, "to": new_val}
                setattr(outcome, field, new_val)
    if changes:
        outcome.status = ReviewStatus.modified
        outcome.last_modified_by = email
        outcome.updated_at = datetime.utcnow()
        _log(session, email, "modify", f"outcome:{outcome.id}", {"changes": changes})
    session.add(outcome)
    session.commit()
    return _redirect_step(outcome.paper_unique_id, "outcomes", anchor=f"o-{outcome.id}")


@app.post("/outcomes/{outcome_id}/delete")
def outcome_delete(
    request: Request,
    outcome_id: int,
    session: Session = Depends(get_session),
):
    email = current_email(request)
    outcome = session.get(TableOutcome, outcome_id)
    if not outcome:
        raise HTTPException(404)
    paper = _get_paper_or_404(session, outcome.paper_unique_id)
    _require_checkout(request, paper)
    outcome.status = ReviewStatus.deleted
    outcome.last_modified_by = email
    outcome.updated_at = datetime.utcnow()
    _log(session, email, "delete_outcome", f"outcome:{outcome.id}")
    session.add(outcome)
    session.commit()
    return _redirect_step(outcome.paper_unique_id, "outcomes")


# ---------------------------------------------------------------------------
# Step 3 — Timepoints
# ---------------------------------------------------------------------------


@app.get("/papers/{unique_id}/timepoints", response_class=HTMLResponse)
def step_timepoints(
    request: Request,
    unique_id: str,
    msg: Optional[str] = None,
    err: Optional[str] = None,
    session: Session = Depends(get_session),
):
    current_email(request)
    paper = _get_paper_or_404(session, unique_id)
    if (blocked := _require_draft_access(request, paper)):
        return blocked
    tables = _effect_size_tables(session, unique_id)
    table_ids = [t.id for t in tables if t.id is not None]
    timepoints = _timepoints_for_tables(session, table_ids)
    ctx = _shell_context(request, paper, "timepoints", msg, err)
    ctx.update({"tables": tables, "timepoints_by_table": timepoints, "show_save_bar": True})
    return templates.TemplateResponse(request, "step_timepoints.html", ctx)


@app.post("/papers/{unique_id}/tables/{table_id}/timepoints/new")
async def timepoint_create(
    request: Request,
    unique_id: str,
    table_id: int,
    session: Session = Depends(get_session),
):
    email = current_email(request)
    paper = _get_paper_or_404(session, unique_id)
    _require_checkout(request, paper)
    form = await request.form()
    tp = TableTimepoint(
        table_id=table_id,
        paper_unique_id=unique_id,
        timepoint_label=str(form.get("timepoint_label", "")),
        outcome_timeframe_months=str(form.get("outcome_timeframe_months", "")),
        status=ReviewStatus.modified,
        last_modified_by=email,
    )
    session.add(tp)
    session.commit()
    session.refresh(tp)
    _log(session, email, "add_timepoint", f"timepoint:{tp.id}",
         {"table_id": table_id, "label": tp.timepoint_label})
    session.commit()
    return _redirect_step(unique_id, "timepoints", anchor=f"tp-{tp.id}")


@app.post("/timepoints/{tp_id}/edit")
async def timepoint_edit(
    request: Request,
    tp_id: int,
    session: Session = Depends(get_session),
):
    email = current_email(request)
    form = await request.form()
    tp = session.get(TableTimepoint, tp_id)
    if not tp:
        raise HTTPException(404)
    paper = _get_paper_or_404(session, tp.paper_unique_id)
    _require_checkout(request, paper)
    changes: dict[str, dict[str, str]] = {}
    for field in ("timepoint_label", "outcome_timeframe_months", "reviewer_notes"):
        if field in form:
            new_val = str(form[field])
            old_val = getattr(tp, field, "")
            if new_val != old_val:
                changes[field] = {"from": old_val, "to": new_val}
                setattr(tp, field, new_val)
    if changes:
        tp.status = ReviewStatus.modified
        tp.last_modified_by = email
        tp.updated_at = datetime.utcnow()
        _log(session, email, "modify", f"timepoint:{tp.id}", {"changes": changes})
    session.add(tp)
    session.commit()
    return _redirect_step(tp.paper_unique_id, "timepoints", anchor=f"tp-{tp.id}")


@app.post("/timepoints/{tp_id}/delete")
def timepoint_delete(
    request: Request,
    tp_id: int,
    session: Session = Depends(get_session),
):
    email = current_email(request)
    tp = session.get(TableTimepoint, tp_id)
    if not tp:
        raise HTTPException(404)
    paper = _get_paper_or_404(session, tp.paper_unique_id)
    _require_checkout(request, paper)
    tp.status = ReviewStatus.deleted
    tp.last_modified_by = email
    tp.updated_at = datetime.utcnow()
    _log(session, email, "delete_timepoint", f"timepoint:{tp.id}")
    session.add(tp)
    session.commit()
    return _redirect_step(tp.paper_unique_id, "timepoints")


# ---------------------------------------------------------------------------
# Step 4 — Estimates
# ---------------------------------------------------------------------------


@app.get("/papers/{unique_id}/estimates", response_class=HTMLResponse)
def step_estimates(
    request: Request,
    unique_id: str,
    msg: Optional[str] = None,
    err: Optional[str] = None,
    session: Session = Depends(get_session),
):
    current_email(request)
    paper = _get_paper_or_404(session, unique_id)
    if (blocked := _require_draft_access(request, paper)):
        return blocked
    tables = _effect_size_tables(session, unique_id)
    table_ids = [t.id for t in tables if t.id is not None]
    outcomes = _outcomes_for_tables(session, table_ids)
    timepoints = _timepoints_for_tables(session, table_ids)
    estimates_by_table: dict[int, list[EffectSizeRow]] = {tid: [] for tid in table_ids}
    orphan_estimates: list[EffectSizeRow] = []
    for es in session.exec(
        select(EffectSizeRow)
        .where(EffectSizeRow.paper_unique_id == unique_id)
        .where(EffectSizeRow.status != ReviewStatus.deleted)
        .order_by(EffectSizeRow.id)
    ):
        if es.table_id in estimates_by_table:
            estimates_by_table[es.table_id].append(es)
        else:
            orphan_estimates.append(es)

    ctx = _shell_context(request, paper, "estimates", msg, err)
    ctx.update({
        "tables": tables,
        "outcomes_by_table": outcomes,
        "timepoints_by_table": timepoints,
        "estimates_by_table": estimates_by_table,
        "orphan_estimates": orphan_estimates,
        "effect_fields": _editable_effect_fields(),
        "show_save_bar": True,
    })
    return templates.TemplateResponse(request, "step_estimates.html", ctx)


@app.post("/papers/{unique_id}/effect-sizes/new")
async def effect_create(
    request: Request,
    unique_id: str,
    session: Session = Depends(get_session),
):
    email = current_email(request)
    form = await request.form()
    paper = _get_paper_or_404(session, unique_id)
    _require_checkout(request, paper)

    def _opt_int(key: str) -> Optional[int]:
        raw = form.get(key)
        if raw is None or str(raw).strip() == "":
            return None
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None

    table_id = _opt_int("table_id")
    outcome_id = _opt_int("outcome_id")
    timepoint_id = _opt_int("timepoint_id")

    kwargs = {f: str(form.get(f, "")) for f in _editable_effect_fields()}
    es = EffectSizeRow(
        paper_unique_id=unique_id,
        table_id=table_id,
        outcome_id=outcome_id,
        timepoint_id=timepoint_id,
        status=ReviewStatus.modified,
        last_modified_by=email,
        **kwargs,
    )
    session.add(es)
    session.commit()
    session.refresh(es)
    _log(session, email, "add", f"effect_size:{es.id}", {
        "table_id": table_id, "outcome_id": outcome_id, "timepoint_id": timepoint_id,
    })
    session.commit()
    return _redirect_step(unique_id, "estimates", anchor=f"es-{es.id}")


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
    paper = _get_paper_or_404(session, es.paper_unique_id)
    _require_checkout(request, paper)

    changes: dict[str, dict[str, str]] = {}
    for field in _editable_effect_fields():
        if field in form:
            new_val = str(form[field])
            old_val = getattr(es, field, "")
            if new_val != old_val:
                changes[field] = {"from": old_val, "to": new_val}
                setattr(es, field, new_val)

    # Allow re-linking estimate to a different outcome/timepoint inside the
    # same table without retyping every field.
    for fk in ("outcome_id", "timepoint_id"):
        if fk in form:
            raw = str(form[fk]).strip()
            new_val: Optional[int]
            if raw == "":
                new_val = None
            else:
                try:
                    new_val = int(raw)
                except ValueError:
                    new_val = None
            old_val = getattr(es, fk)
            if new_val != old_val:
                changes[fk] = {"from": old_val, "to": new_val}
                setattr(es, fk, new_val)

    reviewer_notes = form.get("reviewer_notes")
    if reviewer_notes is not None and reviewer_notes != es.reviewer_notes:
        changes["reviewer_notes"] = {"from": es.reviewer_notes, "to": str(reviewer_notes)}
        es.reviewer_notes = str(reviewer_notes)

    if changes:
        es.status = (
            ReviewStatus.needs_reextraction if es.needs_reextraction else ReviewStatus.modified
        )
        es.last_modified_by = email
        es.updated_at = datetime.utcnow()
        _log(session, email, "modify", f"effect_size:{es.id}", {"changes": changes})

    session.add(es)
    session.commit()
    return _redirect_step(es.paper_unique_id, "estimates", anchor=f"es-{es.id}")




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
    paper = _get_paper_or_404(session, es.paper_unique_id)
    _require_checkout(request, paper)
    es.status = ReviewStatus.deleted
    es.last_modified_by = email
    es.updated_at = datetime.utcnow()
    _log(session, email, "delete", f"effect_size:{es.id}")
    session.add(es)
    session.commit()
    return _redirect_step(es.paper_unique_id, "estimates")


@app.post("/papers/{unique_id}/flag-reextract")
def paper_flag_reextract(
    request: Request,
    unique_id: str,
    session: Session = Depends(get_session),
):
    """Toggle the paper-level re-extract flag (set + unset on same button)."""
    email = current_email(request)
    paper = _get_paper_or_404(session, unique_id)
    _require_checkout(request, paper)
    paper.needs_reextraction = not paper.needs_reextraction
    paper.status = (
        ReviewStatus.needs_reextraction if paper.needs_reextraction
        else ReviewStatus.modified
    )
    paper.last_modified_by = email
    paper.updated_at = datetime.utcnow()
    _log(
        session, email,
        "flag_reextract" if paper.needs_reextraction else "unflag_reextract",
        f"paper:{paper.id}",
    )
    session.add(paper)
    session.commit()
    return _redirect_step(unique_id, "info")


@app.post("/effect-sizes/{es_id}/flag-reextract")
def effect_flag_reextract(
    request: Request,
    es_id: int,
    session: Session = Depends(get_session),
):
    """Toggle the per-estimate re-extract flag."""
    email = current_email(request)
    es = session.get(EffectSizeRow, es_id)
    if not es:
        raise HTTPException(404)
    paper = _get_paper_or_404(session, es.paper_unique_id)
    _require_checkout(request, paper)
    es.needs_reextraction = not es.needs_reextraction
    es.status = (
        ReviewStatus.needs_reextraction if es.needs_reextraction
        else ReviewStatus.modified
    )
    es.last_modified_by = email
    es.updated_at = datetime.utcnow()
    _log(
        session, email,
        "flag_reextract" if es.needs_reextraction else "unflag_reextract",
        f"effect_size:{es.id}",
    )
    session.add(es)
    session.commit()
    return _redirect_step(es.paper_unique_id, "estimates", anchor=f"es-{es.id}")


# ---------------------------------------------------------------------------
# Admin: Sheet sync + user management
# ---------------------------------------------------------------------------


_EDIT_ACTIONS = {"modify", "submit"}
_ADD_ACTIONS = {
    "add", "add_table_manual", "add_outcome", "add_timepoint",
}
_DELETE_ACTIONS = {
    "delete", "delete_table", "delete_outcome", "delete_timepoint",
}


def _reviewer_activity(session: Session) -> list[dict]:
    """Aggregate per-reviewer counts from paper_reviews and audit_log."""
    from sqlalchemy import func

    completed = dict(
        session.exec(
            select(PaperReview.reviewer_email, func.count(PaperReview.id))
            .group_by(PaperReview.reviewer_email)
        ).all()
    )

    action_counts: dict[str, dict[str, int]] = {}
    last_seen: dict[str, datetime] = {}
    rows = session.exec(
        select(AuditLog.email, AuditLog.action, func.count(AuditLog.id), func.max(AuditLog.when))
        .group_by(AuditLog.email, AuditLog.action)
    ).all()
    for email, action, count, when in rows:
        action_counts.setdefault(email, {})[action] = count
        if when and (email not in last_seen or when > last_seen[email]):
            last_seen[email] = when

    emails = set(completed) | set(action_counts)
    stats = []
    for email in sorted(emails):
        per_action = action_counts.get(email, {})
        edits = sum(per_action.get(a, 0) for a in _EDIT_ACTIONS)
        adds = sum(per_action.get(a, 0) for a in _ADD_ACTIONS)
        deletes = sum(per_action.get(a, 0) for a in _DELETE_ACTIONS)
        stats.append({
            "email": email,
            "papers_completed": completed.get(email, 0),
            "edits": edits,
            "adds": adds,
            "deletes": deletes,
            "total_actions": sum(per_action.values()),
            "last_activity": last_seen.get(email),
        })
    stats.sort(key=lambda s: (-s["papers_completed"], -s["total_actions"], s["email"]))
    return stats


@app.get("/admin", response_class=HTMLResponse)
def admin_panel(
    request: Request,
    msg: Optional[str] = None,
    err: Optional[str] = None,
    session: Session = Depends(get_session),
):
    require_admin(request)
    users = list(session.exec(select(User).order_by(User.email)))
    activity = _reviewer_activity(session)
    return templates.TemplateResponse(
        request,
        "admin.html",
        {
            "msg": msg,
            "err": err,
            "display_name": request.session.get("display_name"),
            "sheet_id": os.environ.get("GOOGLE_SHEET_ID", ""),
            "users": users,
            "activity": activity,
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


@app.post("/refresh-from-sheet")
def refresh_from_sheet(
    request: Request,
    next: Optional[str] = Form(None),
    session: Session = Depends(get_session),
):
    """Pull the Sheet and replace all UNLOCKED papers in the local DB."""
    actor = current_email(request)
    try:
        papers, effects, tables, outcomes, timepoints, reviews = pull_from_sheets()
        n_p, n_e, skipped = import_from_sheet_rows(
            _engine, papers, effects,
            table_rows=tables, outcome_rows=outcomes, timepoint_rows=timepoints,
            review_rows=reviews,
            replace=True, preserve_checked_out=True,
        )
    except Exception as exc:  # noqa: BLE001
        return RedirectResponse(
            f"/?err={str(exc)[:300]}", status_code=status.HTTP_303_SEE_OTHER
        )
    _log(
        session, actor, "refresh_sheet", "sheet",
        {"papers": n_p, "effect_sizes": n_e, "skipped": skipped},
    )
    session.commit()
    note = f"Refreshed+{n_p}+papers+and+{n_e}+effect+sizes+from+Sheet"
    if skipped:
        note += f".+Skipped+{len(skipped)}+checked-out+paper(s)"
    target = next or "/"
    sep = "&" if "?" in target else "?"
    return RedirectResponse(
        f"{target}{sep}msg={note}",
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
        tables = list(
            session.exec(select(PaperTable).where(PaperTable.status != ReviewStatus.deleted))
        )
        outcomes = list(
            session.exec(select(TableOutcome).where(TableOutcome.status != ReviewStatus.deleted))
        )
        timepoints = list(
            session.exec(select(TableTimepoint).where(TableTimepoint.status != ReviewStatus.deleted))
        )
        effects = list(
            session.exec(select(EffectSizeRow).where(EffectSizeRow.status != ReviewStatus.deleted))
        )
        reviews = list(session.exec(select(PaperReview)))
        n_p, n_e = push_to_sheets(papers, effects, tables, outcomes, timepoints, reviews)
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
# Health check (Railway hits this)
# ---------------------------------------------------------------------------


@app.get("/help", response_class=HTMLResponse)
def help_page(request: Request):
    return templates.TemplateResponse(
        request,
        "help.html",
        {"display_name": request.session.get("display_name")},
    )


@app.get("/health")
def health() -> dict:
    return {"ok": True}


def serve() -> None:
    """Console entrypoint: `lancet-web`."""
    import uvicorn

    host = os.environ.get("WEB_HOST", "0.0.0.0")
    port = int(os.environ.get("PORT") or os.environ.get("WEB_PORT", "8000"))
    uvicorn.run("webapp.main:app", host=host, port=port, reload=False)
