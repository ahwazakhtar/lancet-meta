"""
Driver that runs Claude (via the Claude Agent SDK) over a PDF and produces a
structured `Paper` object.

The pipeline runs on the user's local machine. It authenticates via the local
Claude Code login (so a Claude Max / Pro subscription is sufficient — no API
key needed). The Agent SDK spawns Claude Code as a subprocess and grants it
read access to the PDF.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any

from .prompts import build_extraction_prompt
from .schema import NA, EffectSize, Paper

logger = logging.getLogger(__name__)


class ExtractionError(RuntimeError):
    pass


def _strip_code_fence(text: str) -> str:
    """Strip ```json ... ``` fences if Claude added them despite the prompt."""
    text = text.strip()
    if text.startswith("```"):
        # remove leading fence and optional language tag
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _coerce_value(v: Any) -> str:
    """Coerce any extracted value into the string form our schema expects."""
    if v is None:
        return NA
    if isinstance(v, bool):
        return "Yes" if v else "No"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        s = v.strip()
        return s if s else NA
    # lists / dicts — flatten compactly so review users can still read it
    return json.dumps(v, ensure_ascii=False)


def _parse_paper_payload(payload: dict[str, Any], source_pdf: str) -> Paper:
    """Convert raw LLM JSON into a validated `Paper`."""

    raw_effect_sizes = payload.pop("effect_sizes", []) or []
    # Coerce paper-level fields onto the model.
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
    return paper


async def _run_agent(prompt: str, pdf_dir: Path) -> str:
    """Call Claude via the Agent SDK and return the assistant's final text."""
    # Import lazily so the rest of the pipeline (and tests) work without the
    # SDK installed.
    from claude_agent_sdk import ClaudeAgentOptions, query  # type: ignore

    options = ClaudeAgentOptions(
        allowed_tools=["Read", "Bash"],
        cwd=str(pdf_dir),
        permission_mode="bypassPermissions",
        system_prompt=(
            "You are a careful evidence-synthesis assistant. You will receive a "
            "path to a PDF and a schema. Read the PDF carefully (including all "
            "tables) and respond with a single JSON object as instructed. "
            "Never invent values; use \"data not available\" when missing."
        ),
    )

    parts: list[str] = []
    async for message in query(prompt=prompt, options=options):
        # The SDK yields message objects; we collect any text blocks.
        content = getattr(message, "content", None)
        if isinstance(content, list):
            for block in content:
                text = getattr(block, "text", None)
                if text:
                    parts.append(text)
        elif isinstance(content, str):
            parts.append(content)
    return "\n".join(parts).strip()


def extract_paper(pdf_path: Path) -> Paper:
    """Synchronous entrypoint: extract a single PDF into a `Paper`."""

    pdf_path = pdf_path.resolve()
    if not pdf_path.exists():
        raise ExtractionError(f"PDF does not exist: {pdf_path}")

    prompt = build_extraction_prompt(pdf_path.name)
    logger.info("Extracting %s via Claude Agent SDK", pdf_path.name)

    raw = asyncio.run(_run_agent(prompt, pdf_path.parent))
    cleaned = _strip_code_fence(raw)

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        # Sometimes the model wraps JSON in prose; try to locate the largest
        # JSON object in the response.
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise ExtractionError(
                f"Could not parse JSON from Claude response for {pdf_path.name}: {exc}\n"
                f"--- response ---\n{cleaned[:2000]}"
            ) from exc
        payload = json.loads(match.group(0))

    return _parse_paper_payload(payload, source_pdf=pdf_path.name)
