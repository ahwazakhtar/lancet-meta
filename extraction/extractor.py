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
from .prompts import build_extraction_prompt, build_extraction_prompt_with_content
from .schema import NA, EffectSize, Paper

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


def _parse_paper_payload(
    payload: dict[str, Any],
    source_pdf: str,
    xlsx_entry: Optional[PaperListEntry] = None,
) -> Paper:
    raw_effect_sizes = payload.pop("effect_sizes", []) or []
    paper_kwargs: dict[str, str] = {"source_pdf": source_pdf}
    for field_name in Paper.model_fields:
        if field_name in ("source_pdf", "effect_sizes"):
            continue
        paper_kwargs[field_name] = _coerce_value(payload.get(field_name))

    effect_sizes: list[EffectSize] = []
    for es in raw_effect_sizes:
        if not isinstance(es, dict):
            continue
        es_kwargs = {name: _coerce_value(es.get(name)) for name in EffectSize.model_fields}
        effect_sizes.append(EffectSize(**es_kwargs))

    paper = Paper(**paper_kwargs, effect_sizes=effect_sizes)
    return _apply_xlsx_overrides(paper, xlsx_entry)


async def _run_agent(prompt: str, work_dir: Path) -> str:
    from claude_agent_sdk import ClaudeAgentOptions, query  # type: ignore

    options = ClaudeAgentOptions(
        allowed_tools=["Read", "Bash"],
        cwd=str(work_dir),
        permission_mode="bypassPermissions",
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


async def _run_openai(prompt: str) -> str:
    import os
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    model = os.environ.get("OPENAI_MODEL", "gpt-4.1")
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content or ""


def _use_openai() -> bool:
    import os
    return bool(os.environ.get("OPENAI_API_KEY"))


def extract_from_markdown(
    md_path: Path,
    source_pdf: str,
    xlsx_entry: Optional[PaperListEntry] = None,
) -> Paper:
    """Extract a single pre-processed markdown into a `Paper`."""
    return asyncio.run(extract_from_markdown_async(md_path, source_pdf, xlsx_entry))


async def extract_from_markdown_async(
    md_path: Path,
    source_pdf: str,
    xlsx_entry: Optional[PaperListEntry] = None,
) -> Paper:
    md_path = md_path.resolve()
    if not md_path.exists():
        raise ExtractionError(f"Markdown does not exist: {md_path}")

    if _use_openai():
        logger.info("Extracting %s via OpenAI", md_path.name)
        md_content = md_path.read_text(encoding="utf-8")
        prompt = build_extraction_prompt_with_content(md_content, xlsx_entry)
        raw = await _run_openai(prompt)
    else:
        logger.info("Extracting %s via Claude Agent SDK", md_path.name)
        prompt = build_extraction_prompt(md_path.name, xlsx_entry)
        raw = await _run_agent(prompt, md_path.parent)
    cleaned = _strip_code_fence(raw)

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise ExtractionError(
                f"Could not parse JSON from Claude response for {md_path.name}: {exc}\n"
                f"--- response ---\n{cleaned[:2000]}"
            ) from exc
        payload = json.loads(match.group(0))

    return _parse_paper_payload(payload, source_pdf=source_pdf, xlsx_entry=xlsx_entry)
