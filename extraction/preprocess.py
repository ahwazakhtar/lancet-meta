"""
Pre-process PDFs into DOI-standardized markdown files via Claude.

Why: PDFs can be named anything on disk, and downstream extraction does better
when each paper is keyed by DOI and ships with cleanly-rendered tables.
pdfplumber's table extractor was producing noisy / empty cells; Claude reads
the PDF directly (via the Agent SDK's Read tool) and emits cleaner markdown.

Steps:

1. Extract the DOI deterministically with PyMuPDF (metadata + regex sweep of
   the first few pages). This is cheap, free, and gives us the filename
   target without needing an LLM call.
2. Ask Claude Code (via the Agent SDK) to read the full PDF and emit a single
   markdown document with the required shape:

       # Source PDF: <filename>
       DOI: <doi>

       ## Full text
       ### Page 1
       ... page 1 body text ...
       ### Page 2
       ... etc ...

       ## Tables
       ### Page N · Table M
       | header | ... |
       | ---    | --- |
       | row    | ... |

3. Save the markdown at `<out_dir>/<sanitized-doi>.md`.

If a paper truly has no DOI in the file, we fall back to the PDF stem so the
pipeline still progresses.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF — used only for DOI extraction now.

logger = logging.getLogger(__name__)

# DOIs look like 10.xxxx/some.path; we deliberately strip trailing punctuation
# that often follows a DOI in running text (".", ",", ")", ";").
DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)

PAGE_HEADER_RE = re.compile(r"^###\s+Page\s+\d+\s*$", re.MULTILINE)
TABLE_HEADER_RE = re.compile(r"^###\s+Page\s+\d+\s+·\s+Table\s+\d+\s*$", re.MULTILINE)


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


def _count_pages(pdf_path: Path) -> int:
    with fitz.open(pdf_path) as doc:
        return len(doc)


def _build_preprocess_prompt(pdf_path: Path, doi: Optional[str]) -> str:
    doi_line = f"DOI: {doi}" if doi else "(DOI not detected — omit the DOI line.)"
    return f"""You are converting a research-paper PDF into clean markdown for a
downstream extraction pipeline.

Read the PDF at path: `{pdf_path.name}` (it is in your current working
directory). Use the Read tool — Claude Code can read PDFs natively. If the
PDF is long, read it in multiple chunks using the `pages` argument until you
have covered every page.

# Required output

Respond with the FULL markdown document and NOTHING ELSE — no preamble, no
code fences, no commentary, no closing remarks. Output it verbatim with this
exact structure:

```
# Source PDF: {pdf_path.name}
{doi_line}

## Full text

### Page 1

<verbatim body text of page 1>

### Page 2

<verbatim body text of page 2>

... (one `### Page N` block per page in the PDF, in order)

## Tables

### Page N · Table M

| col1 | col2 | ... |
| --- | --- | --- |
| row | ... | ... |

... (one block per distinct table; M is the table's index within its page,
     starting at 1)
```

# Rules

- The `### Page N` headers must use the page numbers as they appear in the
  PDF's physical page ordering (1-indexed). Every page that has body text
  must appear, in order.
- The `### Page N · Table M` headers use a middle-dot `·` (U+00B7) — not a
  regular period or hyphen.
- For each table, render it as a pipe-delimited markdown table with a `---`
  separator row. Preserve the table's reported numbers verbatim — do not
  reformat, recompute, or round. Escape any literal `|` inside cells as
  `\\|`.
- Drop figure captions, page headers/footers, and copyright boilerplate
  from the body text only if they clearly aren't part of the paper's
  content. When in doubt, keep them.
- Do not include the `## Tables` section if the PDF has no tables. Emit it
  empty (just the header) instead of omitting the structure.
- Do not invent tables that aren't actually in the PDF. Do not extract
  effect sizes — leave that to a downstream step.
- The DOI line is exactly as shown above — emit `{doi_line}` on its own
  line (or omit it entirely if there's no DOI).

Start your reply with `# Source PDF:` — no leading whitespace, no fence.
"""


async def _run_preprocess_agent(prompt: str, work_dir: Path) -> str:
    import os
    from claude_agent_sdk import ClaudeAgentOptions, query  # type: ignore

    options = ClaudeAgentOptions(
        allowed_tools=["Read", "Bash"],
        cwd=str(work_dir),
        permission_mode="bypassPermissions",
        # Sonnet is plenty for PDF -> markdown rendering and ~5x cheaper than
        # Opus. Override with PREPROCESS_MODEL if needed.
        model=os.environ.get("PREPROCESS_MODEL", "claude-sonnet-4-6"),
        system_prompt=(
            "You convert research-paper PDFs into clean markdown. You read "
            "the PDF with the Read tool and output the markdown verbatim "
            "with no commentary, code fences, or wrapping prose."
        ),
    )

    parts: list[str] = []
    async for message in query(prompt=prompt, options=options):
        content = getattr(message, "content", None)
        if isinstance(content, list):
            for block in content:
                text = getattr(block, "text", None)
                if text:
                    parts.append(text)
        elif isinstance(content, str):
            parts.append(content)
    return "\n".join(parts).strip()


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:markdown|md)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _normalize_md(raw: str, pdf_path: Path, doi: Optional[str]) -> str:
    """Snap the LLM's output into the canonical shape if it drifted slightly.

    Specifically, prefix the `# Source PDF:` and `DOI:` lines from known
    values so downstream `_doi_from_md` keeps working even if Claude omitted
    the header.
    """
    body = _strip_code_fence(raw)
    if not body.lstrip().startswith("# Source PDF:"):
        header_lines = [f"# Source PDF: {pdf_path.name}"]
        if doi:
            header_lines.append(f"DOI: {doi}")
        header_lines.append("")
        body = "\n".join(header_lines) + body
    return body.rstrip() + "\n"


def preprocess_pdf(pdf_path: Path, out_dir: Path, skip_existing: bool = True) -> Preprocessed:
    """PDF -> DOI-keyed markdown (Claude-rendered).

    If `skip_existing` and the target MD already exists, do nothing and
    return a `Preprocessed` with `skipped=True`.
    """
    return asyncio.run(preprocess_pdf_async(pdf_path, out_dir, skip_existing))


async def preprocess_pdf_async(
    pdf_path: Path, out_dir: Path, skip_existing: bool = True
) -> Preprocessed:
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

    n_pages_pdf = _count_pages(pdf_path)
    prompt = _build_preprocess_prompt(pdf_path, doi)
    raw = await _run_preprocess_agent(prompt, work_dir=pdf_path.parent)
    if not raw.strip():
        raise RuntimeError(
            f"Claude returned empty preprocessing output for {pdf_path.name}. "
            "Check that the `claude` CLI is signed in and the PDF is readable."
        )

    md_text = _normalize_md(raw, pdf_path, doi)
    md_path.write_text(md_text, encoding="utf-8")

    n_pages = len(PAGE_HEADER_RE.findall(md_text))
    n_tables = len(TABLE_HEADER_RE.findall(md_text))
    logger.info(
        "Preprocessed %s -> %s (doi=%s, pdf_pages=%d, md_pages=%d, tables=%d)",
        pdf_path.name, md_path.name, doi, n_pages_pdf, n_pages, n_tables,
    )
    return Preprocessed(
        md_path=md_path, doi=doi, source_pdf=pdf_path,
        n_pages=n_pages or n_pages_pdf, n_tables=n_tables,
    )
