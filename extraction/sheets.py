"""
Google Sheets sync.

Two tabs:
  - papers       (one row per paper)
  - effect_sizes (one row per effect size, linked by paper_unique_id)

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
from .storage import EffectSizeRow, PaperRow

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

EFFECT_SIZE_SHEET_COLUMNS = [
    "es_id",
    "paper_unique_id",
    *effect_size_field_names(),
    "status",
    "needs_reextraction",
    "reviewer_notes",
    "last_modified_by",
    "updated_at",
]


def _credentials() -> Credentials:
    raw_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    if raw_json:
        try:
            info = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "GOOGLE_SERVICE_ACCOUNT_JSON is set but does not contain valid JSON. "
                "Re-paste the entire content of the service-account key file (a single "
                "JSON object starting with '{' and ending with '}'). "
                f"Parser said: {exc}"
            ) from exc
        return Credentials.from_service_account_info(info, scopes=SCOPES)
    creds_path = os.environ.get("GOOGLE_SERVICE_ACCOUNT_FILE")
    if creds_path and Path(creds_path).exists():
        return Credentials.from_service_account_file(creds_path, scopes=SCOPES)
    raise RuntimeError(
        "Google Sheets credentials are not configured. Set either "
        "GOOGLE_SERVICE_ACCOUNT_JSON (raw JSON content, recommended on Railway) "
        "or GOOGLE_SERVICE_ACCOUNT_FILE (path to a JSON key, recommended locally)."
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


def _row_for_paper(p: PaperRow) -> list[str]:
    def g(k: str) -> str:
        v = getattr(p, k, "")
        if hasattr(v, "value"):
            return v.value
        return str(v) if v is not None else ""

    return [g(c) for c in PAPER_SHEET_COLUMNS]


def _row_for_effect(e: EffectSizeRow) -> list[str]:
    def g(k: str) -> str:
        if k == "es_id":
            return str(e.id or "")
        v = getattr(e, k, "")
        if hasattr(v, "value"):
            return v.value
        return str(v) if v is not None else ""

    return [g(c) for c in EFFECT_SIZE_SHEET_COLUMNS]


def push_to_sheets(papers: Iterable[PaperRow], effect_sizes: Iterable[EffectSizeRow]) -> tuple[int, int]:
    """Replace the contents of the two tabs with the supplied rows."""
    sh = _open_sheet()
    papers_tab = os.environ.get("PAPERS_SHEET_NAME", "papers")
    effects_tab = os.environ.get("EFFECT_SIZES_SHEET_NAME", "effect_sizes")

    p_ws = _open_or_create_tab(sh, papers_tab, PAPER_SHEET_COLUMNS)
    e_ws = _open_or_create_tab(sh, effects_tab, EFFECT_SIZE_SHEET_COLUMNS)

    p_rows = [PAPER_SHEET_COLUMNS] + [_row_for_paper(p) for p in papers]
    e_rows = [EFFECT_SIZE_SHEET_COLUMNS] + [_row_for_effect(e) for e in effect_sizes]

    p_ws.clear()
    p_ws.update("A1", p_rows)
    e_ws.clear()
    e_ws.update("A1", e_rows)

    n_p, n_e = len(p_rows) - 1, len(e_rows) - 1
    logger.info("Pushed %d papers and %d effect sizes to Sheet", n_p, n_e)
    return n_p, n_e


def pull_from_sheets() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Read the two tabs and return raw row dicts."""
    sh = _open_sheet()
    papers_tab = os.environ.get("PAPERS_SHEET_NAME", "papers")
    effects_tab = os.environ.get("EFFECT_SIZES_SHEET_NAME", "effect_sizes")

    try:
        p_ws = sh.worksheet(papers_tab)
        e_ws = sh.worksheet(effects_tab)
    except gspread.WorksheetNotFound as exc:
        raise RuntimeError(f"Sheet tab missing: {exc}") from exc

    papers = p_ws.get_all_records(default_blank="")
    effects = e_ws.get_all_records(default_blank="")
    logger.info("Pulled %d papers and %d effect sizes from Sheet", len(papers), len(effects))
    return papers, effects
