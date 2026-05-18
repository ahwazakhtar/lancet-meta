"""
Driver that runs Claude (via the Claude Agent SDK) on a pre-processed
markdown file and produces a structured `Paper` object.

The pipeline runs on the user's local machine. It authenticates via the local
Claude Code login (so a Claude Max / Pro subscription is sufficient — no API
key needed). The Agent SDK spawns Claude Code as a subprocess and grants it
read access to the markdown.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

from .paper_list import PaperListEntry, study_id_to_unique_id
from .prompts import build_extraction_prompt
from .schema import (
    EffectSize,
    ExtractedTable,
    NA,
    Paper,
    TableOutcome,
    TableTimepoint,
)
from .tables import ParsedTable, parse_tables_from_path

logger = logging.getLogger(__name__)


class ExtractionError(RuntimeError):
    pass


def _strip_code_fence(text: str) -> str:
    """Strip ```json ... ``` fences if Claude added them despite the prompt."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _coerce_value(v: Any) -> str:
    if v is None:
        return NA
    if isinstance(v, bool):
        return "Yes" if v else "No"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        s = v.strip()
        return s if s else NA
    return json.dumps(v, ensure_ascii=False)


def _apply_xlsx_overrides(paper: Paper, entry: Optional[PaperListEntry]) -> Paper:
    """xlsx is authoritative for bibliographic metadata."""
    if entry is None:
        return paper
    if entry.doi:
        paper.doi = entry.doi
    if entry.title:
        paper.title = entry.title
    if entry.authors:
        paper.authors = entry.authors
    if entry.year:
        paper.year = entry.year
    if entry.journal:
        paper.journal = entry.journal
    uid = study_id_to_unique_id(entry.study_id)
    if uid:
        paper.unique_id = uid
    return paper


def _build_outcome(d: Any) -> TableOutcome:
    if not isinstance(d, dict):
        return TableOutcome()
    return TableOutcome(
        outcome_name=_coerce_value(d.get("outcome_name") or d.get("name")),
        outcome_domain=_coerce_value(d.get("outcome_domain") or d.get("domain")),
        outcome_definition=_coerce_value(d.get("outcome_definition") or d.get("definition")),
    )


def _build_timepoint(d: Any) -> TableTimepoint:
    if not isinstance(d, dict):
        return TableTimepoint()
    return TableTimepoint(
        timepoint_label=_coerce_value(d.get("timepoint_label") or d.get("label")),
        outcome_timeframe_months=_coerce_value(
            d.get("outcome_timeframe_months") or d.get("timeframe_months")
        ),
    )


def _build_estimate(d: Any) -> EffectSize:
    if not isinstance(d, dict):
        return EffectSize()
    return EffectSize(
        **{name: _coerce_value(d.get(name)) for name in EffectSize.model_fields}
    )


def _parse_paper_payload(
    payload: dict[str, Any],
    source_pdf: str,
    xlsx_entry: Optional[PaperListEntry] = None,
    parsed_tables: Optional[list[ParsedTable]] = None,
) -> Paper:
    parsed_label_set = {t.label for t in (parsed_tables or [])}

    raw_tables = payload.pop("tables_with_effect_sizes", None) or []
    # Tolerate older cached extractions that still use the flat `effect_sizes`
    # shape — wrap them in a single synthetic table so the rest of the
    # pipeline doesn't need a second code path.
    legacy_effect_sizes = payload.pop("effect_sizes", None) or []

    tables_extracted: list[ExtractedTable] = []
    for entry in raw_tables:
        if not isinstance(entry, dict):
            continue
        label = _coerce_value(entry.get("table_label"))
        if label == NA or not label:
            continue
        if parsed_label_set and label not in parsed_label_set:
            logger.warning(
                "LLM returned table_label %r that doesn't match a parsed "
                "table; keeping it but marking as is_manual.",
                label,
            )
        outcomes = [_build_outcome(o) for o in entry.get("outcomes", []) if isinstance(o, dict)]
        timepoints = [_build_timepoint(t) for t in entry.get("timepoints", []) if isinstance(t, dict)]
        estimates = [_build_estimate(e) for e in entry.get("estimates", []) if isinstance(e, dict)]
        tables_extracted.append(ExtractedTable(
            table_label=label,
            outcomes=outcomes,
            timepoints=timepoints,
            estimates=estimates,
        ))

    if legacy_effect_sizes and not tables_extracted:
        # Best-effort upgrade: emit one synthetic table per distinct
        # outcome_reference so old cached data still loads.
        groups: dict[str, list[dict]] = {}
        for es in legacy_effect_sizes:
            if not isinstance(es, dict):
                continue
            label = _coerce_value(es.get("outcome_reference"))
            groups.setdefault(label if label != NA else "Legacy effect sizes", []).append(es)
        for label, items in groups.items():
            outcomes = [
                TableOutcome(outcome_name=_coerce_value(es.get("outcome_name")))
                for es in items
            ]
            timepoints = [
                TableTimepoint(timepoint_label=_coerce_value(es.get("timepoints")))
                for es in items
            ]
            estimates = [_build_estimate(es) for es in items]
            tables_extracted.append(ExtractedTable(
                table_label=label,
                outcomes=outcomes,
                timepoints=timepoints,
                estimates=estimates,
            ))

    paper_kwargs: dict[str, str] = {"source_pdf": source_pdf}
    for field_name in Paper.model_fields:
        if field_name in ("source_pdf", "tables_with_effect_sizes"):
            continue
        paper_kwargs[field_name] = _coerce_value(payload.get(field_name))

    paper = Paper(**paper_kwargs, tables_with_effect_sizes=tables_extracted)
    return _apply_xlsx_overrides(paper, xlsx_entry)


async def _run_agent(prompt: str, work_dir: Path) -> str:
    import os
    from claude_agent_sdk import ClaudeAgentOptions, query  # type: ignore

    options = ClaudeAgentOptions(
        allowed_tools=["Read", "Bash"],
        cwd=str(work_dir),
        permission_mode="bypassPermissions",
        # Extraction is the harder reasoning step (which tables, which
        # outcomes, which timepoints, link estimates to them) — pin Opus
        # by default. Override with EXTRACT_MODEL.
        model=os.environ.get("EXTRACT_MODEL", "claude-opus-4-7"),
        system_prompt=(
            "You are a careful evidence-synthesis assistant. You will receive "
            "a path to a markdown file containing a research paper's text and "
            "tables, plus a schema. Read it carefully (especially the tables) "
            "and respond with a single JSON object as instructed. Never invent "
            "values; use \"data not available\" when missing."
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


def extract_from_markdown(
    md_path: Path,
    source_pdf: str,
    xlsx_entry: Optional[PaperListEntry] = None,
) -> tuple[Paper, list[ParsedTable]]:
    """Extract a single pre-processed markdown into a `Paper` plus tables."""
    return asyncio.run(extract_from_markdown_async(md_path, source_pdf, xlsx_entry))


async def extract_from_markdown_async(
    md_path: Path,
    source_pdf: str,
    xlsx_entry: Optional[PaperListEntry] = None,
) -> tuple[Paper, list[ParsedTable]]:
    md_path = md_path.resolve()
    if not md_path.exists():
        raise ExtractionError(f"Markdown does not exist: {md_path}")

    parsed_tables = parse_tables_from_path(md_path)
    table_labels = [t.label for t in parsed_tables]

    logger.info("Extracting %s via Claude Agent SDK", md_path.name)
    prompt = build_extraction_prompt(
        md_path.name, xlsx_entry, table_labels=table_labels
    )
    raw = await _run_agent(prompt, md_path.parent)
    cleaned = _strip_code_fence(raw)

    payload = _parse_json_or_fail(cleaned, md_path)
    paper = _parse_paper_payload(
        payload,
        source_pdf=source_pdf,
        xlsx_entry=xlsx_entry,
        parsed_tables=parsed_tables,
    )
    return paper, parsed_tables


def _parse_json_or_fail(cleaned: str, md_path: Path) -> dict:
    """Parse cleaned LLM output as JSON; dump it to a debug file on failure."""

    def _dump_debug(text: str) -> Path:
        debug_dir = md_path.parent.parent / "extracted_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        debug_path = debug_dir / f"{md_path.stem}.raw.txt"
        debug_path.write_text(text or "<empty>", encoding="utf-8")
        return debug_path

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        debug = _dump_debug(cleaned)
        raise ExtractionError(
            f"Could not parse JSON from LLM response for {md_path.name}: {exc}. "
            f"Raw response saved to {debug}. First 500 chars: {cleaned[:500]!r}"
        ) from exc
