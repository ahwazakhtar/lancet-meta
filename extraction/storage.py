"""
Local persistence for extracted papers and effect sizes.

The web app needs random-access reads/writes that a Google Sheet alone can't
serve fast enough; we keep a SQLite mirror on disk and treat the Sheet as the
authoritative published copy that users sync to.

Tables:
  - papers              one row per paper (paper-level fields)
  - effect_sizes        many rows per paper (linked by paper.unique_id)
  - users               web-app user accounts
"""

from __future__ import annotations

import enum
from datetime import datetime
from pathlib import Path
from typing import Optional

from sqlmodel import Column, Field, JSON, Session, SQLModel, create_engine, select

from .schema import EffectSize as EffectSizeSchema
from .schema import Paper as PaperSchema


class ReviewStatus(str, enum.Enum):
    pending = "pending"          # extracted, awaiting review
    confirmed = "confirmed"      # reviewer marked OK
    modified = "modified"        # reviewer edited
    needs_reextraction = "needs_reextraction"
    deleted = "deleted"


class PaperRow(SQLModel, table=True):
    __tablename__ = "papers"

    id: Optional[int] = Field(default=None, primary_key=True)
    unique_id: str = Field(index=True, unique=True)
    source_pdf: str = ""

    doi: str = ""
    title: str = ""
    authors: str = ""
    year: str = ""
    journal: str = ""
    country_region: str = ""
    funding_source: str = ""
    publication_type: str = ""
    design: str = ""
    unit_of_assignment: str = ""
    followup_duration: str = ""
    rob_tool: str = ""
    rob_judgment: str = ""
    setting_type: str = ""
    setting_description: str = ""
    population_description: str = ""
    baseline_value: str = ""
    sample_size: str = ""
    intervention_category: str = ""
    intervention_description: str = ""
    core_components: str = ""
    intensity_dose: str = ""
    implementation_fidelity_reported: str = ""
    implementation_description: str = ""
    comparator: str = ""
    cointerventions: str = ""
    implementation_barriers_facilitators: str = ""
    contextual_barriers_facilitators: str = ""
    notes: str = ""

    status: ReviewStatus = Field(default=ReviewStatus.pending)
    needs_reextraction: bool = Field(default=False)
    reviewer_notes: str = ""
    last_modified_by: Optional[str] = None
    extracted_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Soft lock: while non-null, only this email may edit the paper or its
    # effect sizes (admins can force-release). Setting these is "checking out".
    checked_out_by: Optional[str] = Field(default=None, index=True)
    checked_out_at: Optional[datetime] = None


class EffectSizeRow(SQLModel, table=True):
    __tablename__ = "effect_sizes"

    id: Optional[int] = Field(default=None, primary_key=True)
    paper_unique_id: str = Field(index=True)

    estimation_method: str = ""
    outcome_name: str = ""
    outcome_reference: str = ""
    outcome_domain: str = ""
    outcome_definition: str = ""
    timepoints: str = ""
    effect_size_raw: str = ""
    ci_or_se_raw: str = ""
    p_value: str = ""
    raw_data_extracted: str = ""
    direction_of_effect: str = ""
    subgroups_analyzed: str = ""
    effect_heterogeneity: str = ""
    effect_type_coded: str = ""
    effect_value: str = ""
    lower_ci: str = ""
    upper_ci: str = ""
    variance_se: str = ""
    outcome_timeframe_months: str = ""
    group1_mean: str = ""
    group1_sd: str = ""
    group1_n: str = ""
    group2_mean: str = ""
    group2_sd: str = ""
    group2_n: str = ""
    effect_size_notes: str = ""

    status: ReviewStatus = Field(default=ReviewStatus.pending)
    needs_reextraction: bool = Field(default=False)
    reviewer_notes: str = ""
    last_modified_by: Optional[str] = None
    extracted_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class User(SQLModel, table=True):
    """An allowed reviewer. No password — sign-in is by email only and is
    purely an identification mechanism for audit logging."""

    __tablename__ = "users"

    id: Optional[int] = Field(default=None, primary_key=True)
    email: str = Field(index=True, unique=True)
    display_name: str = ""
    is_admin: bool = Field(default=False)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class AuditLog(SQLModel, table=True):
    """Append-only record of every reviewer action."""

    __tablename__ = "audit_log"

    id: Optional[int] = Field(default=None, primary_key=True)
    when: datetime = Field(default_factory=datetime.utcnow, index=True)
    email: str
    action: str  # confirm | modify | delete | add | flag_reextract
    target: str  # "paper:<id>" or "effect_size:<id>"
    payload: dict = Field(default_factory=dict, sa_column=Column(JSON))


def _ensure_columns(engine, table: str, columns: dict[str, str]) -> None:
    """Add missing columns to a table (SQLite-only lightweight migration)."""
    from sqlalchemy import text

    with engine.begin() as conn:
        existing = {row[1] for row in conn.exec_driver_sql(f"PRAGMA table_info({table})").fetchall()}
        for col, sql_type in columns.items():
            if col not in existing:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col} {sql_type}"))


def get_engine(db_path: str | Path):
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    # Schema additions for existing DBs that pre-date the new columns.
    _ensure_columns(engine, "papers", {
        "checked_out_by": "TEXT",
        "checked_out_at": "DATETIME",
    })
    return engine


def upsert_paper(engine, paper: PaperSchema) -> PaperRow:
    """Insert or update a paper extraction (replaces effect sizes wholesale)."""
    with Session(engine) as session:
        existing = session.exec(
            select(PaperRow).where(PaperRow.unique_id == paper.unique_id)
        ).first()

        paper_data = paper.model_dump(exclude={"effect_sizes"})
        if existing:
            for k, v in paper_data.items():
                setattr(existing, k, v)
            existing.updated_at = datetime.utcnow()
            # If a reviewer hasn't touched it yet, keep status=pending; if they
            # had flagged for re-extraction, reset to pending on re-extract.
            if existing.status == ReviewStatus.needs_reextraction:
                existing.status = ReviewStatus.pending
                existing.needs_reextraction = False
            paper_row = existing
            session.add(paper_row)
            # Wipe and re-insert effect sizes for this paper.
            old = session.exec(
                select(EffectSizeRow).where(EffectSizeRow.paper_unique_id == paper.unique_id)
            ).all()
            for r in old:
                session.delete(r)
        else:
            paper_row = PaperRow(**paper_data)
            session.add(paper_row)

        for es in paper.effect_sizes:
            session.add(EffectSizeRow(paper_unique_id=paper.unique_id, **es.model_dump()))

        session.commit()
        session.refresh(paper_row)
        return paper_row


def list_papers(engine, include_deleted: bool = False) -> list[PaperRow]:
    with Session(engine) as session:
        stmt = select(PaperRow).order_by(PaperRow.unique_id)
        if not include_deleted:
            stmt = stmt.where(PaperRow.status != ReviewStatus.deleted)
        return list(session.exec(stmt))


def list_effect_sizes(engine, paper_unique_id: str) -> list[EffectSizeRow]:
    with Session(engine) as session:
        return list(
            session.exec(
                select(EffectSizeRow)
                .where(EffectSizeRow.paper_unique_id == paper_unique_id)
                .where(EffectSizeRow.status != ReviewStatus.deleted)
                .order_by(EffectSizeRow.id)
            )
        )


def _coerce_status(value: str) -> ReviewStatus:
    try:
        return ReviewStatus(value)
    except (ValueError, TypeError):
        return ReviewStatus.pending


def _coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


_NEVER_OVERWRITE = {"id", "checked_out_by", "checked_out_at"}
_DATETIME_COLS = {"extracted_at", "updated_at", "checked_out_at"}


def _parse_dt(value):
    """Parse a datetime string from a Sheet row. Returns None on failure so
    the SQLModel default_factory can take over."""
    if value is None or isinstance(value, datetime):
        return value if isinstance(value, datetime) else None
    s = str(value).strip()
    if not s:
        return None
    # str(datetime) uses a space separator; fromisoformat in 3.11+ accepts both.
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _clean_datetime_fields(kwargs: dict) -> dict:
    """Replace string datetimes with parsed ones; drop unparseable to use defaults."""
    for k in list(kwargs.keys()):
        if k in _DATETIME_COLS:
            parsed = _parse_dt(kwargs[k])
            if parsed is None:
                del kwargs[k]
            else:
                kwargs[k] = parsed
    return kwargs


def _paper_kwargs_from_row(row: dict) -> dict:
    cols = {c.name for c in PaperRow.__table__.columns} - _NEVER_OVERWRITE
    out = {k: row[k] for k in cols if k in row}
    out["status"] = _coerce_status(row.get("status", "pending"))
    out["needs_reextraction"] = _coerce_bool(row.get("needs_reextraction"))
    return _clean_datetime_fields(out)


def _effect_kwargs_from_row(row: dict) -> dict:
    cols = {c.name for c in EffectSizeRow.__table__.columns} - {"id"}
    out = {k: row[k] for k in cols if k in row}
    out["status"] = _coerce_status(row.get("status", "pending"))
    out["needs_reextraction"] = _coerce_bool(row.get("needs_reextraction"))
    return _clean_datetime_fields(out)


def import_from_sheet_rows(
    engine,
    paper_rows: list[dict],
    effect_rows: list[dict],
    replace: bool = True,
    preserve_checked_out: bool = True,
) -> tuple[int, int, list[str]]:
    """Bulk-load rows from Google Sheets into the local SQLite DB.

    With replace=True, the DB is wiped first so it mirrors the Sheet.
    With preserve_checked_out=True, papers currently checked out (and their
    effect sizes) are kept as-is and skipped in the inbound rows, so a
    reviewer's in-progress work isn't trashed by a bulk refresh.

    Returns (papers_imported, effects_imported, skipped_unique_ids).
    """
    with Session(engine) as session:
        held_ids: set[str] = set()
        if preserve_checked_out:
            held_ids = {
                p.unique_id
                for p in session.exec(
                    select(PaperRow).where(PaperRow.checked_out_by.is_not(None))
                )
            }

        if replace:
            # Delete only the rows we are allowed to replace.
            for r in session.exec(
                select(EffectSizeRow).where(EffectSizeRow.paper_unique_id.not_in(held_ids))
                if held_ids
                else select(EffectSizeRow)
            ).all():
                session.delete(r)
            for r in session.exec(
                select(PaperRow).where(PaperRow.unique_id.not_in(held_ids))
                if held_ids
                else select(PaperRow)
            ).all():
                session.delete(r)
            session.flush()

        n_p = 0
        for row in paper_rows:
            unique_id = (row.get("unique_id") or "").strip()
            if not unique_id or unique_id in held_ids:
                continue
            kwargs = _paper_kwargs_from_row(row)
            kwargs["unique_id"] = unique_id
            existing = session.exec(
                select(PaperRow).where(PaperRow.unique_id == unique_id)
            ).first()
            if existing and not replace:
                for k, v in kwargs.items():
                    setattr(existing, k, v)
                session.add(existing)
            else:
                session.add(PaperRow(**kwargs))
            n_p += 1

        n_e = 0
        for row in effect_rows:
            paper_uid = (row.get("paper_unique_id") or "").strip()
            if not paper_uid or paper_uid in held_ids:
                continue
            kwargs = _effect_kwargs_from_row(row)
            kwargs["paper_unique_id"] = paper_uid
            session.add(EffectSizeRow(**kwargs))
            n_e += 1

        session.commit()
    return n_p, n_e, sorted(held_ids)


def import_paper_from_sheet(
    engine,
    unique_id: str,
    paper_rows: list[dict],
    effect_rows: list[dict],
) -> tuple[bool, int]:
    """Refresh one paper from the Sheet. Preserves the checkout state.

    Returns (paper_found_in_sheet, n_effect_sizes_imported).
    """
    match = next((p for p in paper_rows if (p.get("unique_id") or "").strip() == unique_id), None)
    if not match:
        return False, 0
    matching_effects = [
        e for e in effect_rows if (e.get("paper_unique_id") or "").strip() == unique_id
    ]

    with Session(engine) as session:
        existing = session.exec(select(PaperRow).where(PaperRow.unique_id == unique_id)).first()
        held_by = existing.checked_out_by if existing else None
        held_at = existing.checked_out_at if existing else None

        kwargs = _paper_kwargs_from_row(match)
        kwargs["unique_id"] = unique_id
        if existing:
            for k, v in kwargs.items():
                setattr(existing, k, v)
            # Preserve checkout
            existing.checked_out_by = held_by
            existing.checked_out_at = held_at
            session.add(existing)
        else:
            session.add(PaperRow(**kwargs))

        # Replace this paper's effect sizes wholesale.
        for r in session.exec(
            select(EffectSizeRow).where(EffectSizeRow.paper_unique_id == unique_id)
        ).all():
            session.delete(r)

        n_e = 0
        for row in matching_effects:
            es_kwargs = _effect_kwargs_from_row(row)
            es_kwargs["paper_unique_id"] = unique_id
            session.add(EffectSizeRow(**es_kwargs))
            n_e += 1

        session.commit()
    return True, n_e
