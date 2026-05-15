"""
Google Sheets sync.

A service-account JSON file at the path in `GOOGLE_SERVICE_ACCOUNT_FILE`
authenticates writes. Share the target sheet with the service account's email
address with edit permissions.

Two tabs:
  - papers       (one row per paper)
  - effect_sizes (one row per effect size, linked by unique_id)
"""

from __future__ import annotations

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
    "paper_unique_id",
    *effect_size_field_names(),
    "status",
    "needs_reextraction",
    "reviewer_notes",
    "last_modified_by",
    "updated_at",
]


def _client() -> gspread.Client:
    creds_path = os.environ.get("GOOGLE_SERVICE_ACCOUNT_FILE")
    if not creds_path or not Path(creds_path).exists():
        raise RuntimeError(
            "GOOGLE_SERVICE_ACCOUNT_FILE is not set or does not point to a real file."
        )
    creds = Credentials.from_service_account_file(creds_path, scopes=SCOPES)
    return gspread.authorize(creds)


def _open_or_create_tab(spreadsheet, name: str, header: list[str]):
    try:
        ws = spreadsheet.worksheet(name)
    except gspread.WorksheetNotFound:
        ws = spreadsheet.add_worksheet(title=name, rows=2000, cols=len(header))
        ws.update("A1", [header])
        return ws
    # Ensure header row matches; if not, overwrite it.
    existing = ws.row_values(1)
    if existing != header:
        ws.update("A1", [header])
    return ws


def _row_for_paper(p: PaperRow) -> list[str]:
    def g(k: str) -> str:
        v = getattr(p, k, "")
        if hasattr(v, "value"):  # Enum
            return v.value
        return str(v) if v is not None else ""

    return [g(c) for c in PAPER_SHEET_COLUMNS]


def _row_for_effect(e: EffectSizeRow) -> list[str]:
    def g(k: str) -> str:
        v = getattr(e, k, "")
        if hasattr(v, "value"):
            return v.value
        return str(v) if v is not None else ""

    return [g(c) for c in EFFECT_SIZE_SHEET_COLUMNS]


def sync_to_sheets(papers: Iterable[PaperRow], effect_sizes: Iterable[EffectSizeRow]) -> None:
    """Replace the contents of the two tabs with the supplied rows."""
    sheet_id = os.environ["GOOGLE_SHEET_ID"]
    papers_tab = os.environ.get("PAPERS_SHEET_NAME", "papers")
    effects_tab = os.environ.get("EFFECT_SIZES_SHEET_NAME", "effect_sizes")

    gc = _client()
    sh = gc.open_by_key(sheet_id)

    p_ws = _open_or_create_tab(sh, papers_tab, PAPER_SHEET_COLUMNS)
    e_ws = _open_or_create_tab(sh, effects_tab, EFFECT_SIZE_SHEET_COLUMNS)

    p_rows = [PAPER_SHEET_COLUMNS] + [_row_for_paper(p) for p in papers]
    e_rows = [EFFECT_SIZE_SHEET_COLUMNS] + [_row_for_effect(e) for e in effect_sizes]

    p_ws.clear()
    p_ws.update("A1", p_rows)
    e_ws.clear()
    e_ws.update("A1", e_rows)

    logger.info(
        "Synced %d papers and %d effect sizes to Google Sheet %s",
        len(p_rows) - 1,
        len(e_rows) - 1,
        sheet_id,
    )
