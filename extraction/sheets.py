"""
Google Sheets sync.

Five tabs:
  - papers           (one row per paper)
  - paper_tables     (one row per markdown table parsed from a paper)
  - table_outcomes   (one row per outcome confirmed in a declared table)
  - table_timepoints (one row per timepoint confirmed in a declared table)
  - effect_sizes     (one row per estimate, linked by paper_unique_id + FKs)

Authentication: either
  - GOOGLE_SERVICE_ACCOUNT_FILE (path to JSON key) — convenient locally
  - GOOGLE_SERVICE_ACCOUNT_JSON (raw JSON content) — convenient on Railway,
    where secrets live in env vars.

Share the target sheet with the service account's email as Editor.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Iterable

import gspread
from google.oauth2.service_account import Credentials

from .schema import effect_size_field_names, paper_field_names
from .storage import (
    EffectSizeRow,
    PaperRow,
    PaperTable,
    TableOutcome,
    TableTimepoint,
)

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]


PAPER_SHEET_COLUMNS = [
    "unique_id",
    "source_pdf",
    *(f for f in paper_field_names() if f != "unique_id"),
    "status",
    "needs_reextraction",
    "reviewer_notes",
    "last_modified_by",
    "updated_at",
]

PAPER_TABLE_SHEET_COLUMNS = [
    "id",
    "paper_unique_id",
    "table_label",
    "page",
    "table_index",
    "is_effect_size",
    "is_manual",
    "status",
    "last_modified_by",
    "updated_at",
    # `body_markdown` deliberately omitted from Sheet — Sheet cells choke on
    # large blobs and reviewers don't edit raw markdown. It's re-parsed from
    # the preprocessed md on extraction.
]

TABLE_OUTCOME_SHEET_COLUMNS = [
    "id",
    "table_id",
    "paper_unique_id",
    "outcome_name",
    "outcome_domain",
    "outcome_definition",
    "status",
    "reviewer_notes",
    "last_modified_by",
    "updated_at",
]

TABLE_TIMEPOINT_SHEET_COLUMNS = [
    "id",
    "table_id",
    "paper_unique_id",
    "timepoint_label",
    "outcome_timeframe_months",
    "status",
    "reviewer_notes",
    "last_modified_by",
    "updated_at",
]

EFFECT_SIZE_SHEET_COLUMNS = [
    "es_id",
    "paper_unique_id",
    "table_id",
    "outcome_id",
    "timepoint_id",
    *effect_size_field_names(),
    "status",
    "needs_reextraction",
    "reviewer_notes",
    "last_modified_by",
    "updated_at",
]


def _credentials() -> Credentials:
    sa_email = os.environ.get("GOOGLE_CLIENT_EMAIL")
    sa_key = os.environ.get("GOOGLE_PRIVATE_KEY")
    if sa_email and sa_key:
        info = {
            "type": "service_account",
            "project_id": os.environ.get("GOOGLE_PROJECT_ID", ""),
            "private_key_id": os.environ.get("GOOGLE_PRIVATE_KEY_ID", ""),
            "private_key": sa_key.replace("\\n", "\n").strip().strip('"').strip("'") + "\n",
            "client_email": sa_email.strip(),
            "client_id": os.environ.get("GOOGLE_CLIENT_ID", ""),
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        }
        pk = info["private_key"]
        if "BEGIN PRIVATE KEY" not in pk or "END PRIVATE KEY" not in pk:
            raise RuntimeError(
                "GOOGLE_PRIVATE_KEY is set but doesn't look like a PEM private key. "
                "Paste the full value of the `private_key` field from the JSON, "
                "including the -----BEGIN PRIVATE KEY----- and -----END PRIVATE KEY----- "
                "lines. Literal '\\n' will be converted back to newlines automatically."
            )
        return Credentials.from_service_account_info(info, scopes=SCOPES)

    raw_b64 = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON_B64")
    if raw_b64:
        import binascii
        import base64

        cleaned = raw_b64.strip().strip('"').strip("'")
        cleaned = "".join(cleaned.split())
        try:
            decoded_bytes = base64.b64decode(cleaned)
        except (binascii.Error, ValueError) as exc:
            raise RuntimeError(
                "GOOGLE_SERVICE_ACCOUNT_JSON_B64 is set but couldn't be decoded as "
                "base64. Re-generate from your key file with PowerShell: "
                "`[Convert]::ToBase64String([IO.File]::ReadAllBytes('key.json'))`. "
                f"Decoder said: {exc}"
            ) from exc
        try:
            info = json.loads(decoded_bytes)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            preview = decoded_bytes[:80]
            raise RuntimeError(
                "GOOGLE_SERVICE_ACCOUNT_JSON_B64 decoded but the result isn't "
                "valid JSON. Most likely causes: you encoded a .p12 file instead "
                "of the .json key, or the file is corrupted. The decoded content "
                f"starts with: {preview!r}. Parser said: {exc}"
            ) from exc
        if not isinstance(info, dict) or "client_email" not in info:
            raise RuntimeError(
                "GOOGLE_SERVICE_ACCOUNT_JSON_B64 decoded into JSON but it doesn't "
                "look like a Google service account key (missing 'client_email'). "
                "Make sure you encoded the .json service-account key from Google "
                "Cloud, not a different file."
            )
        return Credentials.from_service_account_info(info, scopes=SCOPES)

    raw_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    if raw_json:
        try:
            info = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "GOOGLE_SERVICE_ACCOUNT_JSON is set but does not contain valid JSON. "
                "Re-paste using Railway's RAW Editor, or switch to "
                "GOOGLE_SERVICE_ACCOUNT_JSON_B64 (base64-encoded). "
                f"Parser said: {exc}"
            ) from exc
        return Credentials.from_service_account_info(info, scopes=SCOPES)

    creds_path = os.environ.get("GOOGLE_SERVICE_ACCOUNT_FILE")
    if creds_path and Path(creds_path).exists():
        return Credentials.from_service_account_file(creds_path, scopes=SCOPES)

    raise RuntimeError(
        "Google Sheets credentials are not configured. Set one of: "
        "GOOGLE_CLIENT_EMAIL + GOOGLE_PRIVATE_KEY (easiest on Railway), "
        "GOOGLE_SERVICE_ACCOUNT_JSON_B64 (base64-encoded JSON), "
        "GOOGLE_SERVICE_ACCOUNT_JSON (raw JSON), "
        "or GOOGLE_SERVICE_ACCOUNT_FILE (path to a JSON key, local only)."
    )


def _client() -> gspread.Client:
    return gspread.authorize(_credentials())


def _open_sheet():
    sheet_id = os.environ.get("GOOGLE_SHEET_ID")
    if not sheet_id:
        raise RuntimeError("GOOGLE_SHEET_ID is not set.")
    return _client().open_by_key(sheet_id)


def _open_or_create_tab(spreadsheet, name: str, header: list[str]):
    try:
        ws = spreadsheet.worksheet(name)
    except gspread.WorksheetNotFound:
        ws = spreadsheet.add_worksheet(title=name, rows=2000, cols=max(len(header), 30))
        ws.update("A1", [header])
        return ws
    existing = ws.row_values(1)
    if existing != header:
        ws.update("A1", [header])
    return ws


def _tab_names() -> dict[str, str]:
    return {
        "papers": os.environ.get("PAPERS_SHEET_NAME", "papers"),
        "tables": os.environ.get("PAPER_TABLES_SHEET_NAME", "paper_tables"),
        "outcomes": os.environ.get("TABLE_OUTCOMES_SHEET_NAME", "table_outcomes"),
        "timepoints": os.environ.get("TABLE_TIMEPOINTS_SHEET_NAME", "table_timepoints"),
        "effects": os.environ.get("EFFECT_SIZES_SHEET_NAME", "effect_sizes"),
    }


def _cell(v) -> str:
    if hasattr(v, "value"):
        return v.value
    return str(v) if v is not None else ""


def _row_for_paper(p: PaperRow) -> list[str]:
    return [_cell(getattr(p, c, "")) for c in PAPER_SHEET_COLUMNS]


def _row_for_table(t: PaperTable) -> list[str]:
    return [_cell(getattr(t, c, "")) for c in PAPER_TABLE_SHEET_COLUMNS]


def _row_for_outcome(o: TableOutcome) -> list[str]:
    return [_cell(getattr(o, c, "")) for c in TABLE_OUTCOME_SHEET_COLUMNS]


def _row_for_timepoint(tp: TableTimepoint) -> list[str]:
    return [_cell(getattr(tp, c, "")) for c in TABLE_TIMEPOINT_SHEET_COLUMNS]


def _row_for_effect(e: EffectSizeRow) -> list[str]:
    def g(k: str) -> str:
        if k == "es_id":
            return str(e.id or "")
        return _cell(getattr(e, k, ""))
    return [g(c) for c in EFFECT_SIZE_SHEET_COLUMNS]


def push_to_sheets(
    papers: Iterable[PaperRow],
    effect_sizes: Iterable[EffectSizeRow],
    tables: Iterable[PaperTable] = (),
    outcomes: Iterable[TableOutcome] = (),
    timepoints: Iterable[TableTimepoint] = (),
) -> tuple[int, int]:
    """Replace the contents of all five tabs with the supplied rows.

    Returns (papers_pushed, effect_sizes_pushed) for backwards-compatible
    logging; tables/outcomes/timepoints counts are logged but not returned.
    """
    sh = _open_sheet()
    names = _tab_names()

    p_ws = _open_or_create_tab(sh, names["papers"], PAPER_SHEET_COLUMNS)
    t_ws = _open_or_create_tab(sh, names["tables"], PAPER_TABLE_SHEET_COLUMNS)
    o_ws = _open_or_create_tab(sh, names["outcomes"], TABLE_OUTCOME_SHEET_COLUMNS)
    tp_ws = _open_or_create_tab(sh, names["timepoints"], TABLE_TIMEPOINT_SHEET_COLUMNS)
    e_ws = _open_or_create_tab(sh, names["effects"], EFFECT_SIZE_SHEET_COLUMNS)

    p_rows = [PAPER_SHEET_COLUMNS] + [_row_for_paper(p) for p in papers]
    t_rows = [PAPER_TABLE_SHEET_COLUMNS] + [_row_for_table(t) for t in tables]
    o_rows = [TABLE_OUTCOME_SHEET_COLUMNS] + [_row_for_outcome(o) for o in outcomes]
    tp_rows = [TABLE_TIMEPOINT_SHEET_COLUMNS] + [_row_for_timepoint(t) for t in timepoints]
    e_rows = [EFFECT_SIZE_SHEET_COLUMNS] + [_row_for_effect(e) for e in effect_sizes]

    for ws, rows in (
        (p_ws, p_rows),
        (t_ws, t_rows),
        (o_ws, o_rows),
        (tp_ws, tp_rows),
        (e_ws, e_rows),
    ):
        ws.clear()
        ws.update("A1", rows)

    n_p, n_e = len(p_rows) - 1, len(e_rows) - 1
    logger.info(
        "Pushed %d papers, %d tables, %d outcomes, %d timepoints, %d effect sizes to Sheet",
        n_p, len(t_rows) - 1, len(o_rows) - 1, len(tp_rows) - 1, n_e,
    )
    return n_p, n_e


def pull_from_sheets() -> tuple[
    list[dict[str, str]],  # papers
    list[dict[str, str]],  # effects
    list[dict[str, str]],  # tables
    list[dict[str, str]],  # outcomes
    list[dict[str, str]],  # timepoints
]:
    """Read all five tabs and return raw row dicts.

    Tables / outcomes / timepoints are returned as empty lists if the tabs
    don't exist yet (older Sheets won't have them).
    """
    sh = _open_sheet()
    names = _tab_names()

    def _read(tab: str, required: bool) -> list[dict[str, str]]:
        try:
            ws = sh.worksheet(tab)
        except gspread.WorksheetNotFound:
            if required:
                raise RuntimeError(f"Sheet tab missing: {tab}")
            return []
        return ws.get_all_records(default_blank="")

    papers = _read(names["papers"], required=True)
    effects = _read(names["effects"], required=True)
    tables = _read(names["tables"], required=False)
    outcomes = _read(names["outcomes"], required=False)
    timepoints = _read(names["timepoints"], required=False)

    logger.info(
        "Pulled %d papers, %d tables, %d outcomes, %d timepoints, %d effect sizes from Sheet",
        len(papers), len(tables), len(outcomes), len(timepoints), len(effects),
    )
    return papers, effects, tables, outcomes, timepoints
