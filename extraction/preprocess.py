"""
Pre-process PDFs into DOI-standardized markdown files.

Why: PDFs can be named anything on disk, and the LLM does better with clean
text + tables. We:

1. Extract the DOI from the PDF (metadata, then a regex sweep of the first
   pages).
2. Build a markdown document containing:
   - bibliographic header
   - per-page text (PyMuPDF)
   - every table the PDF contains, rendered as a markdown table (pdfplumber)
3. Save the markdown at `<out_dir>/<sanitized-doi>.md` so downstream steps
   key off DOI rather than the original filename.

If a paper truly has no DOI in the file, we fall back to the PDF stem so the
pipeline still progresses.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
import pdfplumber

logger = logging.getLogger(__name__)

# DOIs look like 10.xxxx/some.path; we deliberately strip trailing punctuation
# that often follows a DOI in running text (".", ",", ")", ";").
DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)


@dataclass
class Preprocessed:
    md_path: Path
    doi: Optional[str]
    source_pdf: Path
    n_pages: int
    n_tables: int
    skipped: bool = False  # True when an existing MD was reused


def sanitize_doi(doi: str) -> str:
    """Filesystem-safe form of a DOI."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", doi.strip().lower()).strip("_")


def _strip_doi_trailing(doi: str) -> str:
    return doi.rstrip(".,);]")


def extract_doi(pdf_path: Path) -> Optional[str]:
    """Find a DOI in the PDF's metadata or the first few pages of text."""
    with fitz.open(pdf_path) as doc:
        meta = doc.metadata or {}
        for key in ("doi", "DOI", "subject", "title", "keywords"):
            val = (meta.get(key) or "").strip()
            if val:
                m = DOI_RE.search(val)
                if m:
                    return _strip_doi_trailing(m.group(0))

        text_chunks: list[str] = []
        for page in doc[: min(3, len(doc))]:
            text_chunks.append(page.get_text("text"))
            if sum(len(c) for c in text_chunks) > 12000:
                break
        head = "\n".join(text_chunks)

    m = DOI_RE.search(head)
    if m:
        return _strip_doi_trailing(m.group(0))
    return None


def _table_to_markdown(table: list[list[Optional[str]]]) -> str:
    """Render a pdfplumber table (list of list of strings) as a markdown table."""
    rows = [[((cell or "").strip().replace("\n", " ").replace("|", "\\|")) for cell in row] for row in table if row]
    if not rows:
        return ""
    n_cols = max(len(r) for r in rows)
    rows = [r + [""] * (n_cols - len(r)) for r in rows]
    header, *body = rows
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in range(n_cols)) + " |"]
    out.extend("| " + " | ".join(r) + " |" for r in body)
    return "\n".join(out)


def preprocess_pdf(pdf_path: Path, out_dir: Path, skip_existing: bool = True) -> Preprocessed:
    """PDF -> DOI-keyed markdown. If `skip_existing` and the target MD already
    exists (i.e., this DOI has been preprocessed before), do nothing and
    return a Preprocessed with skipped=True."""
    pdf_path = pdf_path.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    doi = extract_doi(pdf_path)
    stem = sanitize_doi(doi) if doi else pdf_path.stem
    md_path = out_dir / f"{stem}.md"

    if skip_existing and md_path.exists():
        logger.info("Skipping %s (already preprocessed -> %s)", pdf_path.name, md_path.name)
        return Preprocessed(
            md_path=md_path, doi=doi, source_pdf=pdf_path,
            n_pages=0, n_tables=0, skipped=True,
        )

    parts: list[str] = []
    parts.append(f"# Source PDF: {pdf_path.name}")
    if doi:
        parts.append(f"DOI: {doi}")
    parts.append("")

    with fitz.open(pdf_path) as doc:
        n_pages = len(doc)
        parts.append("## Full text")
        for i, page in enumerate(doc, 1):
            text = page.get_text("text").strip()
            if not text:
                continue
            parts.append(f"\n### Page {i}\n")
            parts.append(text)

    n_tables = 0
    parts.append("\n## Tables")
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            try:
                tables = page.extract_tables()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Table extraction failed on page %d of %s: %s", i, pdf_path.name, exc)
                continue
            for j, table in enumerate(tables, 1):
                md = _table_to_markdown(table)
                if not md:
                    continue
                parts.append(f"\n### Page {i} · Table {j}\n")
                parts.append(md)
                n_tables += 1

    md_path.write_text("\n".join(parts), encoding="utf-8")
    logger.info("Preprocessed %s -> %s (doi=%s, tables=%d)", pdf_path.name, md_path.name, doi, n_tables)
    return Preprocessed(md_path=md_path, doi=doi, source_pdf=pdf_path, n_pages=n_pages, n_tables=n_tables)
