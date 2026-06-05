"""
Local persistence for extracted papers and effect sizes.

The web app needs random-access reads/writes that a Google Sheet alone can't
serve fast enough; we keep a SQLite mirror on disk and treat the Sheet as the
authoritative published copy that users sync to.

Tables:
  - papers              one row per paper (paper-level fields)
  - paper_tables        every table parsed from the paper's markdown
  - table_outcomes      outcomes the reviewer confirms per declared table
  - table_timepoints    timepoints the reviewer confirms per declared table
  - effect_sizes        one row per estimate, linked to a (table, outcome,
                        timepoint) triple via FK columns
  - users               web-app user accounts
  - audit_log           append-only record of reviewer actions
"""

from __future__ import annotations

import enum
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

from sqlmodel import Column, Field, JSON, Session, SQLModel, create_engine, select

from .schema import Paper as PaperSchema
from .tables import ParsedTable


class ReviewStatus(str, enum.Enum):
    """Simplified review state.

    `confirmed` is the default for any row — newly-extracted data is treated
    as good unless a reviewer edits or deletes it. `pending` is retained only
    to keep legacy rows readable; the UI displays it identically to
    `confirmed` (no badge).
    """

    confirmed = "confirmed"               # default — clean / untouched
    modified = "modified"                 # reviewer edited
    needs_reextraction = "needs_reextraction"
    deleted = "deleted"
    pending = "pending"                   # legacy — treated as confirmed


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

    status: ReviewStatus = Field(default=ReviewStatus.confirmed)
    needs_reextraction: bool = Field(default=False)
    reviewer_notes: str = ""
    last_modified_by: Optional[str] = None
    extracted_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Soft lock: while non-null, only this email may edit the paper or its
    # effect sizes (admins can force-release). Setting these is "checking out".
    checked_out_by: Optional[str] = Field(default=None, index=True)
    checked_out_at: Optional[datetime] = None


class PaperTable(SQLModel, table=True):
    """Every table parsed out of a paper's preprocessed markdown.

    Step 1 of the review flow: the reviewer flips `is_effect_size` for each
    table. None means "undecided" (initial state for tables the LLM did not
    flag); True/False are explicit reviewer decisions.
    """

    __tablename__ = "paper_tables"

    id: Optional[int] = Field(default=None, primary_key=True)
    paper_unique_id: str = Field(index=True)
    table_label: str
    page: int = 0
    table_index: int = 0
    body_markdown: str = ""

    # None = undecided. True = effect-size table. False = explicitly rejected.
    is_effect_size: Optional[bool] = Field(default=None)

    # Was this table parsed out of the markdown, or added manually by a
    # reviewer (in which case body_markdown will be empty)?
    is_manual: bool = Field(default=False)

    status: ReviewStatus = Field(default=ReviewStatus.confirmed)
    last_modified_by: Optional[str] = None
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class TableOutcome(SQLModel, table=True):
    """Step 2: outcomes the reviewer confirms inside a declared table."""

    __tablename__ = "table_outcomes"

    id: Optional[int] = Field(default=None, primary_key=True)
    table_id: int = Field(index=True)
    paper_unique_id: str = Field(index=True)
    outcome_name: str = ""
    outcome_domain: str = ""
    outcome_definition: str = ""
    status: ReviewStatus = Field(default=ReviewStatus.confirmed)
    reviewer_notes: str = ""
    last_modified_by: Optional[str] = None
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class TableTimepoint(SQLModel, table=True):
    """Step 3: timepoints (baseline, endline, 12-mo, etc.) per table."""

    __tablename__ = "table_timepoints"

    id: Optional[int] = Field(default=None, primary_key=True)
    table_id: int = Field(index=True)
    paper_unique_id: str = Field(index=True)
    timepoint_label: str = ""
    outcome_timeframe_months: str = ""
    status: ReviewStatus = Field(default=ReviewStatus.confirmed)
    reviewer_notes: str = ""
    last_modified_by: Optional[str] = None
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class EffectSizeRow(SQLModel, table=True):
    """Step 4: one estimate, linked to a (table, outcome, timepoint) triple.

    FK columns are nullable: existing pre-migration rows and any estimate the
    LLM couldn't match to a parsed table/outcome/timepoint are stored with
    None so they're still visible (typically with status=needs_reextraction).
    """

    __tablename__ = "effect_sizes"

    id: Optional[int] = Field(default=None, primary_key=True)
    paper_unique_id: str = Field(index=True)
    table_id: Optional[int] = Field(default=None, index=True)
    outcome_id: Optional[int] = Field(default=None, index=True)
    timepoint_id: Optional[int] = Field(default=None, index=True)

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

    status: ReviewStatus = Field(default=ReviewStatus.confirmed)
    needs_reextraction: bool = Field(default=False)
    reviewer_notes: str = ""
    last_modified_by: Optional[str] = None
    extracted_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class PaperReview(SQLModel, table=True):
    """One row per (paper, reviewer) where the reviewer marked the paper
    as reviewed by clicking Done. A paper is "fully reviewed" once two
    distinct reviewer emails appear here. A reviewer can re-check-out the
    paper to make further edits without inserting a duplicate row.
    """

    __tablename__ = "paper_reviews"

    id: Optional[int] = Field(default=None, primary_key=True)
    paper_unique_id: str = Field(index=True)
    reviewer_email: str = Field(index=True)
    completed_at: datetime = Field(default_factory=datetime.utcnow)


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
    action: str  # confirm | modify | delete | add | flag_reextract | declare_table | ...
    target: str  # "paper:<id>" or "effect_size:<id>" or "table:<id>" or ...
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
    _ensure_columns(engine, "effect_sizes", {
        "table_id": "INTEGER",
        "outcome_id": "INTEGER",
        "timepoint_id": "INTEGER",
    })
    # One-time migration: legacy rows used `pending` as the default. The new
    # model treats all clean rows as `confirmed`. Rewrite once.
    from sqlalchemy import text

    with engine.begin() as conn:
        for table in ("papers", "effect_sizes", "paper_tables",
                      "table_outcomes", "table_timepoints"):
            conn.execute(text(
                f"UPDATE {table} SET status='confirmed' WHERE status='pending'"
            ))
    return engine


# ---------------------------------------------------------------------------
# Upsert from extraction pipeline
# ---------------------------------------------------------------------------


def _wipe_paper_children(session: Session, unique_id: str) -> None:
    for model in (EffectSizeRow, TableOutcome, TableTimepoint, PaperTable, PaperReview):
        for row in session.exec(select(model).where(model.paper_unique_id == unique_id)).all():
            session.delete(row)
    session.flush()


def upsert_paper(
    engine,
    paper: PaperSchema,
    parsed_tables: Optional[list[ParsedTable]] = None,
) -> PaperRow:
    """Insert or update a paper extraction (replaces children wholesale).

    `parsed_tables` is the deterministic list of tables parsed from the
    paper's preprocessed markdown. Every parsed table becomes a PaperTable
    row (so reviewers can flag/un-flag them); tables the LLM flagged as
    containing effect sizes get is_effect_size=True and have their outcomes/
    timepoints/estimates populated.
    """
    parsed_tables = parsed_tables or []

    with Session(engine) as session:
        existing = session.exec(
            select(PaperRow).where(PaperRow.unique_id == paper.unique_id)
        ).first()

        paper_data = paper.model_dump(exclude={"tables_with_effect_sizes"})
        if existing:
            for k, v in paper_data.items():
                setattr(existing, k, v)
            existing.updated_at = datetime.utcnow()
            if existing.status == ReviewStatus.needs_reextraction:
                existing.status = ReviewStatus.pending
                existing.needs_reextraction = False
            paper_row = existing
            session.add(paper_row)
            _wipe_paper_children(session, paper.unique_id)
        else:
            paper_row = PaperRow(**paper_data)
            session.add(paper_row)
            session.flush()

        # Index parsed tables by label so we can attach LLM data.
        parsed_by_label = {t.label: t for t in parsed_tables}
        llm_by_label = {t.table_label: t for t in paper.tables_with_effect_sizes}

        # Insert one PaperTable per parsed-table; LLM-flagged ones get
        # is_effect_size=True.
        table_id_by_label: dict[str, int] = {}
        for parsed in parsed_tables:
            row = PaperTable(
                paper_unique_id=paper.unique_id,
                table_label=parsed.label,
                page=parsed.page,
                table_index=parsed.table_index,
                body_markdown=parsed.body_markdown,
                is_effect_size=True if parsed.label in llm_by_label else None,
                is_manual=False,
            )
            session.add(row)
            session.flush()
            table_id_by_label[parsed.label] = row.id  # type: ignore[assignment]

        # If the LLM returned a label that doesn't match any parsed table,
        # drop it. The prompt is explicit ("tables only — discard in-text
        # estimates"), and an invented label almost always means the model
        # tried to surface body-text findings. Creating a manual placeholder
        # for these clutters the reviewer UI. Reviewers can still add genuinely
        # missed tables via the "Add a table manually" form in Step 1.
        for label in llm_by_label:
            if label not in table_id_by_label:
                import logging
                logging.getLogger(__name__).warning(
                    "Dropping LLM-returned table %r for %s — label doesn't match "
                    "any parsed table",
                    label, paper.unique_id,
                )

        # Populate outcomes, timepoints, and estimates for the LLM-flagged
        # tables (only those whose labels matched a parsed table — invented
        # labels were already dropped above).
        for label, llm_tbl in llm_by_label.items():
            if label not in table_id_by_label:
                continue
            table_id = table_id_by_label[label]
            outcome_id_by_name: dict[str, int] = {}
            for oc in llm_tbl.outcomes:
                outcome_row = TableOutcome(
                    table_id=table_id,
                    paper_unique_id=paper.unique_id,
                    outcome_name=oc.outcome_name,
                    outcome_domain=oc.outcome_domain,
                    outcome_definition=oc.outcome_definition,
                )
                session.add(outcome_row)
                session.flush()
                outcome_id_by_name[oc.outcome_name] = outcome_row.id  # type: ignore[assignment]

            tp_id_by_label: dict[str, int] = {}
            for tp in llm_tbl.timepoints:
                tp_row = TableTimepoint(
                    table_id=table_id,
                    paper_unique_id=paper.unique_id,
                    timepoint_label=tp.timepoint_label,
                    outcome_timeframe_months=tp.outcome_timeframe_months,
                )
                session.add(tp_row)
                session.flush()
                tp_id_by_label[tp.timepoint_label] = tp_row.id  # type: ignore[assignment]

            for est in llm_tbl.estimates:
                est_data = est.model_dump()
                matched_outcome = outcome_id_by_name.get(est.outcome_name)
                matched_timepoint = tp_id_by_label.get(est.timepoints)
                needs_reextract = (
                    matched_outcome is None or matched_timepoint is None
                )
                es_row = EffectSizeRow(
                    paper_unique_id=paper.unique_id,
                    table_id=table_id,
                    outcome_id=matched_outcome,
                    timepoint_id=matched_timepoint,
                    needs_reextraction=needs_reextract,
                    status=(
                        ReviewStatus.needs_reextraction
                        if needs_reextract
                        else ReviewStatus.confirmed
                    ),
                    **est_data,
                )
                session.add(es_row)

        session.commit()
        session.refresh(paper_row)
        return paper_row


# ---------------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------------


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


def list_paper_tables(engine, paper_unique_id: str) -> list[PaperTable]:
    with Session(engine) as session:
        return list(
            session.exec(
                select(PaperTable)
                .where(PaperTable.paper_unique_id == paper_unique_id)
                .where(PaperTable.status != ReviewStatus.deleted)
                .order_by(PaperTable.page, PaperTable.table_index, PaperTable.id)
            )
        )


def list_table_outcomes(engine, table_id: int) -> list[TableOutcome]:
    with Session(engine) as session:
        return list(
            session.exec(
                select(TableOutcome)
                .where(TableOutcome.table_id == table_id)
                .where(TableOutcome.status != ReviewStatus.deleted)
                .order_by(TableOutcome.id)
            )
        )


def list_table_timepoints(engine, table_id: int) -> list[TableTimepoint]:
    with Session(engine) as session:
        return list(
            session.exec(
                select(TableTimepoint)
                .where(TableTimepoint.table_id == table_id)
                .where(TableTimepoint.status != ReviewStatus.deleted)
                .order_by(TableTimepoint.id)
            )
        )


# ---------------------------------------------------------------------------
# Sheet round-trip
# ---------------------------------------------------------------------------


def _coerce_status(value) -> ReviewStatus:
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


def _coerce_optional_bool(value) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in {"", "none", "null", "undecided"}:
        return None
    if s in {"true", "1", "yes", "y"}:
        return True
    if s in {"false", "0", "no", "n"}:
        return False
    return None


def _coerce_optional_int(value) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


_NEVER_OVERWRITE_PAPER = {"id", "checked_out_by", "checked_out_at"}
_DATETIME_COLS = {"extracted_at", "updated_at", "checked_out_at"}


def _parse_dt(value):
    if value is None or isinstance(value, datetime):
        return value if isinstance(value, datetime) else None
    s = str(value).strip()
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _clean_datetime_fields(kwargs: dict) -> dict:
    for k in list(kwargs.keys()):
        if k in _DATETIME_COLS:
            parsed = _parse_dt(kwargs[k])
            if parsed is None:
                del kwargs[k]
            else:
                kwargs[k] = parsed
    return kwargs


def _kwargs_from_row(row: dict, model, never_overwrite: set[str]) -> dict:
    cols = {c.name for c in model.__table__.columns} - never_overwrite
    out = {k: row[k] for k in cols if k in row}
    if "status" in cols:
        out["status"] = _coerce_status(row.get("status", "confirmed"))
    if "needs_reextraction" in cols:
        out["needs_reextraction"] = _coerce_bool(row.get("needs_reextraction"))
    if "is_effect_size" in cols:
        out["is_effect_size"] = _coerce_optional_bool(row.get("is_effect_size"))
    if "is_manual" in cols:
        out["is_manual"] = _coerce_bool(row.get("is_manual"))
    for fk in ("table_id", "outcome_id", "timepoint_id", "page", "table_index"):
        if fk in cols and fk in row:
            coerced = _coerce_optional_int(row[fk])
            if coerced is None and fk in {"page", "table_index"}:
                out[fk] = 0
            else:
                out[fk] = coerced
    return _clean_datetime_fields(out)


def _paper_kwargs_from_row(row: dict) -> dict:
    return _kwargs_from_row(row, PaperRow, _NEVER_OVERWRITE_PAPER)


def _effect_kwargs_from_row(row: dict) -> dict:
    return _kwargs_from_row(row, EffectSizeRow, {"id"})


def _table_kwargs_from_row(row: dict) -> dict:
    return _kwargs_from_row(row, PaperTable, {"id"})


def _outcome_kwargs_from_row(row: dict) -> dict:
    return _kwargs_from_row(row, TableOutcome, {"id"})


def _timepoint_kwargs_from_row(row: dict) -> dict:
    return _kwargs_from_row(row, TableTimepoint, {"id"})


def _review_kwargs_from_row(row: dict) -> dict:
    paper_uid = (row.get("paper_unique_id") or "").strip()
    reviewer = (row.get("reviewer_email") or "").strip().lower()
    out: dict = {"paper_unique_id": paper_uid, "reviewer_email": reviewer}
    completed = _parse_dt(row.get("completed_at"))
    if completed is not None:
        out["completed_at"] = completed
    return out


def _strip_held(rows: list[dict], key: str, held: set[str]) -> list[dict]:
    return [r for r in rows if (r.get(key) or "").strip() not in held]


def import_from_sheet_rows(
    engine,
    paper_rows: list[dict],
    effect_rows: list[dict],
    table_rows: Optional[list[dict]] = None,
    outcome_rows: Optional[list[dict]] = None,
    timepoint_rows: Optional[list[dict]] = None,
    review_rows: Optional[list[dict]] = None,
    replace: bool = True,
    preserve_checked_out: bool = True,
) -> tuple[int, int, list[str]]:
    """Bulk-load rows from Google Sheets into the local SQLite DB.

    With replace=True, the DB is wiped first so it mirrors the Sheet.
    With preserve_checked_out=True, papers currently checked out (and their
    children) are kept as-is and skipped in the inbound rows.

    Returns (papers_imported, effects_imported, skipped_unique_ids).
    """
    table_rows = table_rows or []
    outcome_rows = outcome_rows or []
    timepoint_rows = timepoint_rows or []
    review_rows = review_rows or []

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
            for model in (EffectSizeRow, TableOutcome, TableTimepoint, PaperTable, PaperReview):
                stmt = select(model)
                if held_ids:
                    stmt = stmt.where(model.paper_unique_id.not_in(held_ids))
                for r in session.exec(stmt).all():
                    session.delete(r)
            stmt = select(PaperRow)
            if held_ids:
                stmt = stmt.where(PaperRow.unique_id.not_in(held_ids))
            for r in session.exec(stmt).all():
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
        session.flush()

        # Tables: their sheet rows carry an `id` we need so children can
        # reference it. We rewrite IDs on insert and keep an old→new map.
        table_id_map: dict[int, int] = {}
        for row in _strip_held(table_rows, "paper_unique_id", held_ids):
            old_id = _coerce_optional_int(row.get("id"))
            kwargs = _table_kwargs_from_row(row)
            kwargs["paper_unique_id"] = (row.get("paper_unique_id") or "").strip()
            new_row = PaperTable(**kwargs)
            session.add(new_row)
            session.flush()
            if old_id is not None:
                table_id_map[old_id] = new_row.id  # type: ignore[assignment]

        outcome_id_map: dict[int, int] = {}
        for row in _strip_held(outcome_rows, "paper_unique_id", held_ids):
            old_id = _coerce_optional_int(row.get("id"))
            kwargs = _outcome_kwargs_from_row(row)
            kwargs["paper_unique_id"] = (row.get("paper_unique_id") or "").strip()
            old_table_id = _coerce_optional_int(row.get("table_id"))
            kwargs["table_id"] = table_id_map.get(old_table_id) if old_table_id is not None else None
            new_row = TableOutcome(**kwargs)
            session.add(new_row)
            session.flush()
            if old_id is not None:
                outcome_id_map[old_id] = new_row.id  # type: ignore[assignment]

        timepoint_id_map: dict[int, int] = {}
        for row in _strip_held(timepoint_rows, "paper_unique_id", held_ids):
            old_id = _coerce_optional_int(row.get("id"))
            kwargs = _timepoint_kwargs_from_row(row)
            kwargs["paper_unique_id"] = (row.get("paper_unique_id") or "").strip()
            old_table_id = _coerce_optional_int(row.get("table_id"))
            kwargs["table_id"] = table_id_map.get(old_table_id) if old_table_id is not None else None
            new_row = TableTimepoint(**kwargs)
            session.add(new_row)
            session.flush()
            if old_id is not None:
                timepoint_id_map[old_id] = new_row.id  # type: ignore[assignment]

        n_e = 0
        for row in effect_rows:
            paper_uid = (row.get("paper_unique_id") or "").strip()
            if not paper_uid or paper_uid in held_ids:
                continue
            kwargs = _effect_kwargs_from_row(row)
            kwargs["paper_unique_id"] = paper_uid
            old_table_id = _coerce_optional_int(row.get("table_id"))
            old_outcome_id = _coerce_optional_int(row.get("outcome_id"))
            old_timepoint_id = _coerce_optional_int(row.get("timepoint_id"))
            kwargs["table_id"] = table_id_map.get(old_table_id) if old_table_id is not None else None
            kwargs["outcome_id"] = outcome_id_map.get(old_outcome_id) if old_outcome_id is not None else None
            kwargs["timepoint_id"] = timepoint_id_map.get(old_timepoint_id) if old_timepoint_id is not None else None
            session.add(EffectSizeRow(**kwargs))
            n_e += 1

        for row in _strip_held(review_rows, "paper_unique_id", held_ids):
            r_kwargs = _review_kwargs_from_row(row)
            if not r_kwargs["paper_unique_id"] or not r_kwargs["reviewer_email"]:
                continue
            session.add(PaperReview(**r_kwargs))

        session.commit()
    return n_p, n_e, sorted(held_ids)


def import_paper_from_sheet(
    engine,
    unique_id: str,
    paper_rows: list[dict],
    effect_rows: list[dict],
    table_rows: Optional[list[dict]] = None,
    outcome_rows: Optional[list[dict]] = None,
    timepoint_rows: Optional[list[dict]] = None,
    review_rows: Optional[list[dict]] = None,
) -> tuple[bool, int]:
    """Refresh one paper from the Sheet. Preserves the checkout state.

    Returns (paper_found_in_sheet, n_effect_sizes_imported).
    """
    table_rows = table_rows or []
    outcome_rows = outcome_rows or []
    timepoint_rows = timepoint_rows or []
    review_rows = review_rows or []

    match = next(
        (p for p in paper_rows if (p.get("unique_id") or "").strip() == unique_id),
        None,
    )
    if not match:
        return False, 0

    def _filter(rows: Iterable[dict]) -> list[dict]:
        return [r for r in rows if (r.get("paper_unique_id") or "").strip() == unique_id]

    matching_tables = _filter(table_rows)
    matching_outcomes = _filter(outcome_rows)
    matching_timepoints = _filter(timepoint_rows)
    matching_effects = _filter(effect_rows)
    matching_reviews = _filter(review_rows)

    with Session(engine) as session:
        existing = session.exec(
            select(PaperRow).where(PaperRow.unique_id == unique_id)
        ).first()
        held_by = existing.checked_out_by if existing else None
        held_at = existing.checked_out_at if existing else None

        kwargs = _paper_kwargs_from_row(match)
        kwargs["unique_id"] = unique_id
        if existing:
            for k, v in kwargs.items():
                setattr(existing, k, v)
            existing.checked_out_by = held_by
            existing.checked_out_at = held_at
            session.add(existing)
        else:
            session.add(PaperRow(**kwargs))

        # Wipe this paper's children.
        _wipe_paper_children(session, unique_id)
        session.flush()

        # Tables.
        table_id_map: dict[int, int] = {}
        for row in matching_tables:
            old_id = _coerce_optional_int(row.get("id"))
            t_kwargs = _table_kwargs_from_row(row)
            t_kwargs["paper_unique_id"] = unique_id
            new_row = PaperTable(**t_kwargs)
            session.add(new_row)
            session.flush()
            if old_id is not None:
                table_id_map[old_id] = new_row.id  # type: ignore[assignment]

        # Outcomes.
        outcome_id_map: dict[int, int] = {}
        for row in matching_outcomes:
            old_id = _coerce_optional_int(row.get("id"))
            o_kwargs = _outcome_kwargs_from_row(row)
            o_kwargs["paper_unique_id"] = unique_id
            old_table_id = _coerce_optional_int(row.get("table_id"))
            o_kwargs["table_id"] = table_id_map.get(old_table_id) if old_table_id is not None else None
            new_row = TableOutcome(**o_kwargs)
            session.add(new_row)
            session.flush()
            if old_id is not None:
                outcome_id_map[old_id] = new_row.id  # type: ignore[assignment]

        # Timepoints.
        tp_id_map: dict[int, int] = {}
        for row in matching_timepoints:
            old_id = _coerce_optional_int(row.get("id"))
            tp_kwargs = _timepoint_kwargs_from_row(row)
            tp_kwargs["paper_unique_id"] = unique_id
            old_table_id = _coerce_optional_int(row.get("table_id"))
            tp_kwargs["table_id"] = table_id_map.get(old_table_id) if old_table_id is not None else None
            new_row = TableTimepoint(**tp_kwargs)
            session.add(new_row)
            session.flush()
            if old_id is not None:
                tp_id_map[old_id] = new_row.id  # type: ignore[assignment]

        # Estimates.
        n_e = 0
        for row in matching_effects:
            es_kwargs = _effect_kwargs_from_row(row)
            es_kwargs["paper_unique_id"] = unique_id
            old_table_id = _coerce_optional_int(row.get("table_id"))
            old_outcome_id = _coerce_optional_int(row.get("outcome_id"))
            old_timepoint_id = _coerce_optional_int(row.get("timepoint_id"))
            es_kwargs["table_id"] = table_id_map.get(old_table_id) if old_table_id is not None else None
            es_kwargs["outcome_id"] = outcome_id_map.get(old_outcome_id) if old_outcome_id is not None else None
            es_kwargs["timepoint_id"] = tp_id_map.get(old_timepoint_id) if old_timepoint_id is not None else None
            session.add(EffectSizeRow(**es_kwargs))
            n_e += 1

        for row in matching_reviews:
            r_kwargs = _review_kwargs_from_row(row)
            r_kwargs["paper_unique_id"] = unique_id
            if not r_kwargs["reviewer_email"]:
                continue
            session.add(PaperReview(**r_kwargs))

        session.commit()
    return True, n_e
