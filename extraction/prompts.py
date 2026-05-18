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


def build_extraction_prompt(
    md_path: Optional[str],
    xlsx_entry: Optional[PaperListEntry] = None,
    table_labels: Optional[list[str]] = None,
) -> str:
    """Prompt sent to Claude (via the Agent SDK) to extract a single paper.

    The agent has Read tool access and reads the preprocessed markdown at
    `md_path` itself. `table_labels` is the parsed list of `Page N · Table M`
    labels; we tell the LLM to only return labels from this set.
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
        "tables_with_effect_sizes": [
            {
                "table_label": "Page N · Table M",
                "outcomes": [
                    {"outcome_name": NA, "outcome_domain": NA, "outcome_definition": NA},
                ],
                "timepoints": [
                    {"timepoint_label": NA, "outcome_timeframe_months": NA},
                ],
                "estimates": [
                    {
                        "outcome_name": "must match one of the outcomes above",
                        "timepoints": "must match one of the timepoint_labels above",
                        "effect_size_raw": NA,
                        "...": "all effect-size fields below",
                    }
                ],
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

    paper_block = f"Read the pre-processed markdown for the paper at path: `{md_path}`"

    if table_labels:
        label_lines = "\n".join(f"- {lab}" for lab in table_labels)
        allowed_labels_block = f"""
# Allowed table labels

These are the tables that exist in the markdown's `## Tables` section. Every
`table_label` you return MUST be one of these exact strings — do not invent
labels, paraphrase, or merge tables.

{label_lines}
"""
    else:
        allowed_labels_block = ""

    return f"""You are an expert evidence-synthesis assistant extracting structured data
from a research paper for a Lancet meta-analysis on gun violence interventions.

# Task

{paper_block}

The markdown contains the paper's full text by page, followed by every table
extracted from the PDF rendered as a markdown table.
{biblio_block}{allowed_labels_block}
# What to extract

1. **Paper-level fields** that describe the whole paper (design,
   intervention, etc.).

2. **Tables that contain effect sizes**. For each such table, return:
   - `table_label`: the exact `Page N · Table M` label from the markdown.
   - `outcomes`: the list of outcomes the table reports (one entry per
     distinct outcome — e.g., "homicide rate", "non-fatal shootings").
   - `timepoints`: the list of timepoints the table reports (e.g.,
     "baseline", "endline", "12 mo", "1991-2016"). If the table reports a
     single overall window, return one timepoint for it.
   - `estimates`: every quantitative estimate the table reports. Each
     estimate's `outcome_name` MUST match one of the `outcomes` you declared
     for that table, and its `timepoints` value MUST match one of the
     `timepoint_label` values you declared.

3. **Effect sizes** include: difference in means, regression coefficient,
   odds ratio, risk ratio, incidence rate ratio, hazard ratio, standardised
   mean difference, percentage change, beta coefficient, etc.

   **Fallback rule (per the protocol):** if a comparison reports only group
   means and SDs (no effect size), still record it as an estimate row and
   fill `group1_mean / group1_sd / group1_n / group2_mean / group2_sd /
   group2_n` instead of the standard effect fields.

# Critical rules

- **Tables only.** If an effect size is reported in the body text but NOT
  inside any of the labelled tables above, DROP IT. Do not invent a table
  label, and do not return a `tables_with_effect_sizes` entry for body-text
  estimates.
- Only extract what the markdown actually contains. **Do not invent values.**
- For any field the paper does not provide, use the exact string
  "data not available".
- Preserve the paper's reported numbers verbatim (don't reformat units,
  recompute, or round).
- Numeric fields (`effect_value`, `lower_ci`, `upper_ci`, `variance_se`,
  `outcome_timeframe_months`, `year`) should hold the number as a string
  when available, else "data not available".
- Direction of effect: one of "↓ beneficial", "↑ harmful", "↔ null", or
  "data not available" if direction is unclear.
- Be exhaustive about effect sizes — papers often have many across tables.
  Each subgroup × timepoint × outcome combination typically warrants its
  own estimate row.

# Paper-level fields

{paper_fields}

# Effect-size-level fields (one row each, inside `estimates`)

{es_fields}

# Output

Respond with a SINGLE JSON object — no markdown fences, no commentary
before or after, just the JSON — matching this shape:

{json.dumps(schema_skeleton, indent=2)}

Every paper-level field above must appear as a top-level key, even if the
value is "data not available". `tables_with_effect_sizes` is a list
(possibly empty if the paper has no tables that report quantitative
between-group comparisons).
"""
