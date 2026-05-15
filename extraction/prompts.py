"""Prompts for the extraction pipeline."""

from __future__ import annotations

import json
from typing import Optional

from .paper_list import PaperListEntry
from .schema import EFFECT_SIZE_FIELDS, PAPER_FIELDS, FieldSpec

NA = "data not available"


def _field_table(fields: list[FieldSpec]) -> str:
    lines = []
    for f in fields:
        line = f"- `{f.name}`: {f.notes}"
        if f.example:
            line += f" (example: {f.example})"
        lines.append(line)
    return "\n".join(lines)


def build_extraction_prompt_with_content(
    md_content: str, xlsx_entry: Optional[PaperListEntry] = None
) -> str:
    """Prompt for direct API calls — embeds the markdown content inline."""
    return build_extraction_prompt(md_path=None, xlsx_entry=xlsx_entry, md_content=md_content)


def build_extraction_prompt(
    md_path: Optional[str],
    xlsx_entry: Optional[PaperListEntry] = None,
    md_content: Optional[str] = None,
) -> str:
    """Prompt sent to the LLM to extract a single paper.

    Pass `md_path` for Agent SDK (model reads the file itself).
    Pass `md_content` for direct API calls (content embedded inline).
    """

    paper_fields = _field_table(PAPER_FIELDS)
    es_fields = _field_table(EFFECT_SIZE_FIELDS)

    schema_skeleton = {
        "doi": NA,
        "unique_id": NA,
        "title": NA,
        "authors": NA,
        "year": NA,
        "journal": NA,
        "...": "all paper-level fields below",
        "effect_sizes": [
            {
                "outcome_name": NA,
                "effect_size_raw": NA,
                "...": "all effect-size fields below",
            }
        ],
    }

    if xlsx_entry is not None:
        biblio_block = f"""
# Authoritative bibliographic metadata (from curated paper list)

Use these values verbatim for the corresponding fields — they have already
been hand-curated and should not be re-derived from the PDF.

- `doi`: {xlsx_entry.doi or NA}
- `title`: {xlsx_entry.title or NA}
- `authors`: {xlsx_entry.authors or NA}
- `year`: {xlsx_entry.year or NA}
- `journal`: {xlsx_entry.journal or NA}
- `unique_id`: {(xlsx_entry.study_id or NA).replace(" ", "")}
"""
    else:
        biblio_block = ""

    if md_content is not None:
        paper_block = f"""# Paper content

{md_content}"""
    else:
        paper_block = f"Read the pre-processed markdown for the paper at path: `{md_path}`"

    return f"""You are an expert evidence-synthesis assistant extracting structured data
from a research paper for a Lancet meta-analysis on gun violence interventions.

# Task

{paper_block}

The markdown contains the paper's full text by page, followed by every table
extracted from the PDF rendered as a markdown table. Effect sizes will most
often be in the tables — read them carefully.
{biblio_block}
# What to extract

1. **Paper-level fields** that describe the whole paper (design,
   intervention, etc.).
2. **Effect sizes**: every quantitative estimate of the difference between
   comparison groups that the paper reports. Look in the tables, then the
   results text. Effect sizes include: difference in means, regression
   coefficient, odds ratio, risk ratio, incidence rate ratio, hazard ratio,
   standardised mean difference, percentage change, beta coefficient, etc.

   **Fallback rule (per the protocol):** if a comparison reports only group
   means and SDs (no effect size), still record it as an effect-size row and
   fill `group1_mean / group1_sd / group1_n / group2_mean / group2_sd /
   group2_n` instead of the standard effect fields.

# Rules

- Only extract what the markdown actually contains. **Do not invent values.**
- For any field the paper does not provide, use the exact string
  "data not available".
- Preserve the paper's reported numbers verbatim (don't reformat units,
  recompute, or round).
- Numeric fields (`effect_value`, `lower_ci`, `upper_ci`, `variance_se`,
  `outcome_timeframe_months`, `year`) should hold the number as a string when
  available, else "data not available".
- Direction of effect: one of "↓ beneficial", "↑ harmful", "↔ null", or
  "data not available" if direction is unclear.
- Be exhaustive about effect sizes — papers often have many across tables.
  Each subgroup × timepoint × outcome combination typically warrants its own
  row.

# Paper-level fields

{paper_fields}

# Effect-size-level fields (one row each)

{es_fields}

# Output

Respond with a SINGLE JSON object — no markdown fences, no commentary before
or after, just the JSON — matching this shape:

{json.dumps(schema_skeleton, indent=2)}

Every paper-level field above must appear as a key, even if the value is
"data not available". `effect_sizes` is a list (possibly empty if the paper
truly reports no quantitative comparisons).
"""
