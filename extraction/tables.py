"""
Markdown table parser for preprocessed papers.

The preprocess step writes a `## Tables` section followed by
`### Page {page} · Table {idx}` blocks containing pipe-delimited tables.
This module turns that section into a structured list of tables that the
review UI can iterate over.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


TABLES_SECTION_RE = re.compile(r"^##\s+Tables\s*$", re.MULTILINE)
TABLE_HEADER_RE = re.compile(
    r"^###\s+Page\s+(\d+)\s+·\s+Table\s+(\d+)\s*$",
    re.MULTILINE,
)


@dataclass(frozen=True)
class ParsedTable:
    label: str           # e.g. "Page 4 · Table 1"
    page: int
    table_index: int
    body_markdown: str


def _is_meaningful_body(body: str) -> bool:
    """Drop tables that pdfplumber emitted as empty / cell-less stubs.

    pdfplumber sometimes detects a "table" that is really just whitespace —
    those bodies look like `| |\n| --- |\n| |`. We filter them out so the
    reviewer doesn't see noise. A body is meaningful if at least one cell
    contains a non-whitespace character.
    """
    for line in body.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        # Strip the leading/trailing pipes and look at each cell.
        cells = [c.strip() for c in line.strip("|").split("|")]
        # The separator row is all dashes — ignore it.
        if cells and all(set(c) <= set("-: ") for c in cells if c):
            continue
        if any(cell for cell in cells):
            return True
    return False


def parse_tables(md_text: str) -> list[ParsedTable]:
    """Extract every table from a preprocessed markdown's `## Tables` section.

    Returns an empty list if the section is missing. Tables whose body has
    no real content are dropped — see `_is_meaningful_body`.
    """
    section_match = TABLES_SECTION_RE.search(md_text)
    if not section_match:
        return []
    section = md_text[section_match.end():]

    headers = list(TABLE_HEADER_RE.finditer(section))
    if not headers:
        return []

    tables: list[ParsedTable] = []
    for i, m in enumerate(headers):
        page = int(m.group(1))
        idx = int(m.group(2))
        body_start = m.end()
        body_end = headers[i + 1].start() if i + 1 < len(headers) else len(section)
        body = section[body_start:body_end].strip("\n")
        if not _is_meaningful_body(body):
            continue
        tables.append(ParsedTable(
            label=f"Page {page} · Table {idx}",
            page=page,
            table_index=idx,
            body_markdown=body,
        ))
    return tables


def parse_tables_from_path(md_path: Path | str) -> list[ParsedTable]:
    return parse_tables(Path(md_path).read_text(encoding="utf-8"))


def format_label_list(tables: Iterable[ParsedTable]) -> str:
    """Build a bullet-list of table labels for embedding in the LLM prompt."""
    return "\n".join(f"- {t.label}" for t in tables)
