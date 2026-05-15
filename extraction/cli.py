"""
Command-line interface for the extraction pipeline.

Examples:

  # extract every PDF in ./pdfs into the local SQLite DB and JSON cache
  python -m extraction extract

  # extract one PDF
  python -m extraction extract --pdf pdfs/Abdallah2021.pdf

  # push the SQLite state to Google Sheets
  python -m extraction sync

  # create a reviewer account for the web app
  python -m extraction create-user --username ahwaz --admin
"""

from __future__ import annotations

import getpass
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from sqlmodel import Session, select

from .extractor import ExtractionError, extract_paper
from .schema import Paper
from .sheets import sync_to_sheets
from .storage import (
    EffectSizeRow,
    PaperRow,
    ReviewStatus,
    User,
    get_engine,
    list_effect_sizes,
    list_papers,
    upsert_paper,
)

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = typer.Typer(help="Lancet meta-analysis extraction pipeline.")
console = Console()


def _db_path() -> str:
    return os.environ.get("WEB_DB_PATH", "data/app.db")


def _pdf_dir() -> Path:
    return Path(os.environ.get("PDF_DIR", "pdfs"))


def _extracted_dir() -> Path:
    p = Path(os.environ.get("EXTRACTED_DIR", "data/extracted"))
    p.mkdir(parents=True, exist_ok=True)
    return p


@app.command()
def extract(
    pdf: Optional[Path] = typer.Option(None, help="Single PDF to extract; otherwise process the whole PDF_DIR."),
    skip_existing: bool = typer.Option(True, help="Skip PDFs already extracted (look for cache JSON)."),
    limit: Optional[int] = typer.Option(None, help="Process at most N PDFs (useful for testing)."),
) -> None:
    """Run extraction over one PDF or every PDF in PDF_DIR."""

    engine = get_engine(_db_path())
    cache_dir = _extracted_dir()

    if pdf is not None:
        pdfs = [pdf]
    else:
        pdfs = sorted(_pdf_dir().glob("*.pdf"))
        if not pdfs:
            console.print(f"[yellow]No PDFs found in {_pdf_dir()}[/yellow]")
            raise typer.Exit(0)

    if limit is not None:
        pdfs = pdfs[:limit]

    console.print(f"Found [bold]{len(pdfs)}[/bold] PDF(s) to process")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Extracting...", total=len(pdfs))
        for pdf_path in pdfs:
            cache_file = cache_dir / f"{pdf_path.stem}.json"
            if skip_existing and cache_file.exists():
                progress.console.log(f"skip (cached): {pdf_path.name}")
                progress.advance(task)
                continue

            progress.update(task, description=f"Extracting {pdf_path.name}")
            try:
                paper = extract_paper(pdf_path)
            except ExtractionError as exc:
                progress.console.log(f"[red]FAIL[/red] {pdf_path.name}: {exc}")
                progress.advance(task)
                continue

            cache_file.write_text(paper.model_dump_json(indent=2))
            upsert_paper(engine, paper)
            progress.console.log(
                f"[green]OK[/green] {pdf_path.name}: {len(paper.effect_sizes)} effect size(s)"
            )
            progress.advance(task)


@app.command()
def reload_cache() -> None:
    """Re-import every JSON file in EXTRACTED_DIR into the SQLite DB."""
    engine = get_engine(_db_path())
    cache_dir = _extracted_dir()
    files = sorted(cache_dir.glob("*.json"))
    console.print(f"Reloading {len(files)} cached extractions...")
    for f in files:
        try:
            paper = Paper.model_validate_json(f.read_text())
            upsert_paper(engine, paper)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]FAIL[/red] {f.name}: {exc}")
    console.print("[green]Done[/green]")


@app.command()
def sync() -> None:
    """Push the current SQLite contents to Google Sheets."""
    engine = get_engine(_db_path())
    with Session(engine) as session:
        papers = list(
            session.exec(
                select(PaperRow).where(PaperRow.status != ReviewStatus.deleted)
            )
        )
        effects = list(
            session.exec(
                select(EffectSizeRow).where(EffectSizeRow.status != ReviewStatus.deleted)
            )
        )
    sync_to_sheets(papers, effects)
    console.print(
        f"[green]Pushed[/green] {len(papers)} papers and {len(effects)} effect sizes."
    )


@app.command("create-user")
def create_user(
    username: str = typer.Option(..., prompt=True),
    admin: bool = typer.Option(False),
    password: Optional[str] = typer.Option(None, help="Password (if omitted, will prompt)."),
) -> None:
    """Create a reviewer account for the web app."""
    from webapp.auth import hash_password

    if password is None:
        password = getpass.getpass("Password: ")
        confirm = getpass.getpass("Confirm: ")
        if password != confirm:
            console.print("[red]Passwords do not match[/red]")
            raise typer.Exit(1)

    engine = get_engine(_db_path())
    with Session(engine) as session:
        existing = session.exec(select(User).where(User.username == username)).first()
        if existing:
            console.print(f"[yellow]User '{username}' already exists; updating password.[/yellow]")
            existing.password_hash = hash_password(password)
            existing.is_admin = admin
            session.add(existing)
        else:
            session.add(User(username=username, password_hash=hash_password(password), is_admin=admin))
        session.commit()
    console.print(f"[green]User '{username}' created (admin={admin}).[/green]")


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

    pdf_count = len(list(_pdf_dir().glob("*.pdf")))
    cache_count = len(list(_extracted_dir().glob("*.json")))
    console.print(f"PDFs available: {pdf_count}")
    console.print(f"Cached JSON extractions: {cache_count}")


if __name__ == "__main__":
    app()
