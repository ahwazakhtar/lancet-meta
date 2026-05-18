"""
Command-line interface for the local extraction pipeline.

Typical flow:

  python -m extraction preprocess              # pdfs/ -> data/preprocessed/<doi>.md
  python -m extraction extract                  # markdown -> SQLite + JSON cache
  python -m extraction publish                  # SQLite -> Google Sheet

Other commands:

  python -m extraction status                   # progress summary
  python -m extraction add-user --email you@example.com --admin
  python -m extraction reload-cache             # rebuild SQLite from cached JSON
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from sqlmodel import Session, select

from .extractor import (
    ExtractionError,
    _parse_paper_payload,
    extract_from_markdown,
    extract_from_markdown_async,
)
from .paper_list import load_paper_list, lookup_by_doi
from .preprocess import preprocess_pdf
from .schema import Paper
from .sheets import push_to_sheets
from .storage import (
    EffectSizeRow,
    PaperRow,
    PaperTable,
    ReviewStatus,
    TableOutcome,
    TableTimepoint,
    User,
    get_engine,
    list_papers,
    upsert_paper,
)
from .tables import parse_tables_from_path

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = typer.Typer(help="Lancet meta-analysis extraction pipeline.")
console = Console()


def _db_path() -> str:
    return os.environ.get("WEB_DB_PATH", "data/app.db")


def _pdf_dir() -> Path:
    return Path(os.environ.get("PDF_DIR", "pdfs"))


def _preprocessed_dir() -> Path:
    p = Path(os.environ.get("PREPROCESSED_DIR", "data/preprocessed"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def _extracted_dir() -> Path:
    p = Path(os.environ.get("EXTRACTED_DIR", "data/extracted"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def _xlsx_path() -> Path:
    return Path(os.environ.get("PAPER_LIST_XLSX", "base-data/field and paper list.xlsx"))


# ---------------------------------------------------------------------------
# Preprocess: PDF -> markdown keyed by DOI
# ---------------------------------------------------------------------------


@app.command()
def preprocess(
    pdf: Optional[Path] = typer.Option(None, help="Single PDF to preprocess; otherwise the whole PDF_DIR."),
    skip_existing: bool = typer.Option(True, help="Skip PDFs whose markdown already exists."),
    limit: Optional[int] = typer.Option(None, help="Process at most N PDFs."),
) -> None:
    """Convert PDFs into DOI-standardized markdown (text + tables)."""
    out_dir = _preprocessed_dir()

    if pdf is not None:
        pdfs = [pdf]
    else:
        pdfs = sorted(_pdf_dir().glob("*.pdf"))
        if not pdfs:
            console.print(f"[yellow]No PDFs in {_pdf_dir()}[/yellow]")
            raise typer.Exit(0)
    if limit is not None:
        pdfs = pdfs[:limit]

    console.print(f"Preprocessing [bold]{len(pdfs)}[/bold] PDF(s) -> {out_dir}")

    n_skipped = n_done = n_fail = 0
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), TimeElapsedColumn(), console=console) as prog:
        task = prog.add_task("Preprocessing", total=len(pdfs))
        for p in pdfs:
            try:
                result = preprocess_pdf(p, out_dir, skip_existing=skip_existing)
            except Exception as exc:  # noqa: BLE001
                prog.console.log(f"[red]FAIL[/red] {p.name}: {exc}")
                n_fail += 1
                prog.advance(task)
                continue
            doi_repr = result.doi or "(no DOI)"
            if result.skipped:
                prog.console.log(f"[dim]skip[/dim] {p.name} (already done -> {result.md_path.name})")
                n_skipped += 1
            else:
                prog.console.log(
                    f"[green]OK[/green] {p.name} -> {result.md_path.name} "
                    f"(doi={doi_repr}, pages={result.n_pages}, tables={result.n_tables})"
                )
                n_done += 1
            prog.advance(task)
    console.print(f"[bold]Done.[/bold] preprocessed={n_done}, skipped={n_skipped}, failed={n_fail}")


# ---------------------------------------------------------------------------
# Extract: markdown -> SQLite + JSON cache
# ---------------------------------------------------------------------------


@app.command()
def extract(
    md: Optional[Path] = typer.Option(None, help="Single markdown file; otherwise the whole preprocessed dir."),
    skip_existing: bool = typer.Option(True, help="Skip markdowns already in the JSON cache."),
    limit: Optional[int] = typer.Option(None, help="Process at most N files."),
    require_doi: bool = typer.Option(False, help="Skip papers with no DOI match in the xlsx."),
    concurrency: int = typer.Option(5, help="Number of papers to extract in parallel."),
    publish: bool = typer.Option(
        False,
        "--publish/--no-publish",
        help="After successful extraction, push the SQLite contents to the Google Sheet.",
    ),
) -> None:
    """Run Claude over each preprocessed markdown to produce structured data (parallel)."""
    engine = get_engine(_db_path())
    cache_dir = _extracted_dir()
    by_doi, _all = load_paper_list(_xlsx_path())

    if md is not None:
        files = [md]
    else:
        files = sorted(_preprocessed_dir().glob("*.md"))
        if not files:
            console.print(f"[yellow]No markdown in {_preprocessed_dir()} — did you run `preprocess` first?[/yellow]")
            raise typer.Exit(0)
    if limit is not None:
        files = files[:limit]

    console.print(f"Extracting [bold]{len(files)}[/bold] paper(s) (concurrency={concurrency})")
    asyncio.run(_extract_all(files, engine, cache_dir, by_doi, skip_existing, require_doi, concurrency))

    if publish:
        console.print("Publishing to Google Sheet...")
        _do_publish(engine)


async def _extract_all(files, engine, cache_dir, by_doi, skip_existing, require_doi, concurrency):
    sem = asyncio.Semaphore(concurrency)
    db_lock = asyncio.Lock()

    async def process_one(md_path):
        cache_file = cache_dir / f"{md_path.stem}.json"
        if skip_existing and cache_file.exists():
            console.print(f"skip (cached): {md_path.name}")
            return

        doi = _doi_from_md(md_path)
        entry = lookup_by_doi(by_doi, doi) if doi else None
        if require_doi and entry is None:
            console.print(f"[yellow]skip[/yellow] {md_path.name}: no xlsx match for doi={doi}")
            return

        async with sem:
            try:
                paper, parsed_tables = await extract_from_markdown_async(
                    md_path, source_pdf=md_path.name, xlsx_entry=entry
                )
            except ExtractionError as exc:
                console.print(f"[red]FAIL[/red] {md_path.name}: {exc}")
                return

        cache_file.write_text(paper.model_dump_json(indent=2), encoding="utf-8")
        async with db_lock:
            upsert_paper(engine, paper, parsed_tables=parsed_tables)
        n_estimates = sum(len(t.estimates) for t in paper.tables_with_effect_sizes)
        n_tables = len(paper.tables_with_effect_sizes)
        console.print(
            f"[green]OK[/green] {md_path.name}: {paper.unique_id} · "
            f"{n_tables} table(s) · {n_estimates} estimate(s)"
        )

    await asyncio.gather(*[process_one(f) for f in files])


def _doi_from_md(md_path: Path) -> Optional[str]:
    """Read the DOI line from a preprocessed markdown file."""
    try:
        with md_path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i > 5:
                    break
                if line.startswith("DOI:"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


# ---------------------------------------------------------------------------
# Cache / publish utilities
# ---------------------------------------------------------------------------


@app.command()
def reload_cache() -> None:
    """Re-import every JSON file in EXTRACTED_DIR into the SQLite DB.

    Handles both the new (`tables_with_effect_sizes`) and legacy (flat
    `effect_sizes`) JSON shapes — legacy is converted on the fly.
    """
    import json

    engine = get_engine(_db_path())
    files = sorted(_extracted_dir().glob("*.json"))
    md_dir = _preprocessed_dir()
    console.print(f"Reloading {len(files)} cached extractions...")
    for f in files:
        try:
            payload = json.loads(f.read_text(encoding="utf-8"))
            md_path = md_dir / f"{f.stem}.md"
            parsed_tables = parse_tables_from_path(md_path) if md_path.exists() else []
            source_pdf = payload.get("source_pdf") or f.stem
            paper = _parse_paper_payload(
                payload, source_pdf=source_pdf, parsed_tables=parsed_tables
            )
            upsert_paper(engine, paper, parsed_tables=parsed_tables)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]FAIL[/red] {f.name}: {exc}")
    console.print("[green]Done[/green]")


def _do_publish(engine) -> tuple[int, int]:
    with Session(engine) as session:
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
    n_p, n_e = push_to_sheets(papers, effects, tables, outcomes, timepoints)
    console.print(
        f"[green]Published[/green] {n_p} paper(s), {len(tables)} table(s), "
        f"{len(outcomes)} outcome(s), {len(timepoints)} timepoint(s), "
        f"{n_e} effect size(s)."
    )
    return n_p, n_e


@app.command()
def publish() -> None:
    """Push the local SQLite contents to Google Sheets (overwrites all five tabs)."""
    _do_publish(get_engine(_db_path()))


# ---------------------------------------------------------------------------
# User management + status
# ---------------------------------------------------------------------------


@app.command("add-user")
def add_user(
    email: str = typer.Option(..., prompt=True, help="Reviewer's email address."),
    name: str = typer.Option("", help="Display name (optional)."),
    admin: bool = typer.Option(False, help="Grant admin privileges."),
) -> None:
    """Allow an email to sign in to the review UI (no password)."""
    from webapp.auth import EMAIL_RE, normalize_email

    norm = normalize_email(email)
    if not EMAIL_RE.match(norm):
        console.print(f"[red]Invalid email: {email}[/red]")
        raise typer.Exit(1)

    engine = get_engine(_db_path())
    with Session(engine) as session:
        existing = session.exec(select(User).where(User.email == norm)).first()
        if existing:
            existing.display_name = name or existing.display_name
            existing.is_admin = admin
            session.add(existing)
            verb = "updated"
        else:
            session.add(User(email=norm, display_name=name, is_admin=admin))
            verb = "added"
        session.commit()
    console.print(f"[green]Reviewer {verb}: {norm} (admin={admin})[/green]")


@app.command()
def status() -> None:
    """Show a summary of extraction progress."""
    engine = get_engine(_db_path())
    papers = list_papers(engine)
    by_status: dict[str, int] = {}
    for p in papers:
        by_status[p.status.value] = by_status.get(p.status.value, 0) + 1
    console.print(f"Papers in DB: [bold]{len(papers)}[/bold]")
    for s, n in by_status.items():
        console.print(f"  {s}: {n}")

    with Session(engine) as session:
        n_tables = session.exec(select(PaperTable).where(PaperTable.status != ReviewStatus.deleted)).all()
        n_effect_tables = [t for t in n_tables if t.is_effect_size]
        n_outcomes = session.exec(select(TableOutcome).where(TableOutcome.status != ReviewStatus.deleted)).all()
        n_timepoints = session.exec(select(TableTimepoint).where(TableTimepoint.status != ReviewStatus.deleted)).all()
        n_effects = session.exec(select(EffectSizeRow).where(EffectSizeRow.status != ReviewStatus.deleted)).all()
    console.print(
        f"Tables: [bold]{len(n_tables)}[/bold] parsed "
        f"({len(n_effect_tables)} flagged as effect-size)"
    )
    console.print(f"Outcomes: {len(n_outcomes)}")
    console.print(f"Timepoints: {len(n_timepoints)}")
    console.print(f"Effect-size estimates: {len(n_effects)}")

    pdf_count = len(list(_pdf_dir().glob("*.pdf")))
    md_count = len(list(_preprocessed_dir().glob("*.md")))
    cache_count = len(list(_extracted_dir().glob("*.json")))
    console.print(f"PDFs available: {pdf_count}")
    console.print(f"Preprocessed markdown: {md_count}")
    console.print(f"Cached JSON extractions: {cache_count}")


if __name__ == "__main__":
    app()
