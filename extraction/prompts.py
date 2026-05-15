"""Prompts for the extraction pipeline."""

from __future__ import annotations

import json

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


def build_extraction_prompt(pdf_path: str) -> str:
    """Prompt sent to Claude (via Agent SDK) to extract a single paper."""

    paper_fields = _field_table(PAPER_FIELDS)
    es_fields = _field_table(EFFECT_SIZE_FIELDS)

    schema_skeleton = {
        "doi": NA,
        "unique_id": NA,
        "authors": NA,
        "year": NA,
        "country_region": NA,
        "...": "all paper-level fields below",
        "effect_sizes": [
            {
                "outcome_name": NA,
                "effect_size_raw": NA,
                "...": "all effect-size fields below",
            }
        ],
    }

    return f"""You are an expert evidence-synthesis assistant extracting structured data from a
research paper for a Lancet meta-analysis on gun violence interventions.

# Task

Read the PDF at path: `{pdf_path}`

Extract two things:

1. **Paper-level fields** that describe the whole paper (authors, design,
   intervention, etc.).
2. **Effect sizes**: every quantitative estimate of the difference between
   comparison groups that the paper reports. Look in tables, figures, and
   results text. An effect size is any of: difference in means, regression
   coefficient, odds ratio, risk ratio, incidence rate ratio, hazard ratio,
   standardised mean difference, percentage change, beta coefficient, etc.

   **Fallback rule (per the protocol):** if a comparison reports only group
   means and SDs (no effect size), still record it as an effect-size row and
   fill `group1_mean / group1_sd / group1_n / group2_mean / group2_sd /
   group2_n` instead of the standard effect fields.

# Rules

- Only extract what the PDF actually contains. **Do not invent values.**
- For any field where the paper does not provide a value, set it to the
  exact string "data not available".
- Preserve the paper's reported numbers verbatim (don't reformat units, don't
  recompute, don't round).
- Numeric fields (`effect_value`, `lower_ci`, `upper_ci`, `variance_se`,
  `outcome_timeframe_months`, `year`) should be the number as a string when
  available; otherwise "data not available".
- `unique_id` should be `<FirstAuthorSurname><Year>`, e.g. `Wilcox2013`.
- Direction of effect should be one of: "↓ beneficial", "↑ harmful",
  "↔ null", or "data not available" if direction is unclear.
- Be exhaustive about effect sizes — papers often have many across tables.
  Each subgroup or timepoint typically warrants a separate row.
- If a paper reports effect sizes for multiple outcomes (e.g. homicide rate
  AND non-fatal injury), emit one effect-size row per outcome × timepoint ×
  subgroup combination.

# Paper-level fields

{paper_fields}

# Effect-size-level fields (one row each)

{es_fields}

# Output

Respond with a SINGLE JSON object — no markdown fences, no commentary before
or after, just JSON — that matches this shape:

{json.dumps(schema_skeleton, indent=2)}

Make sure every paper-level field above appears as a key, even if the value
is "data not available". `effect_sizes` is a list (possibly empty if the
paper truly reports no quantitative comparisons).
"""
