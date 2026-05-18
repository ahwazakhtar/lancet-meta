"""
Data schema for paper-level metadata and the table-scoped extraction.

The field list mirrors the Template sheet in
`base-data/field and paper list.xlsx`. We split it into:

  * Paper-level: applies to the whole paper (one row per paper).
  * Effect-size-level: 25 fields describing a single estimate.

Effect sizes are now nested under tables: every estimate must reference one
of the paper's preprocessed markdown tables. The LLM is instructed to drop
any estimate reported only in body text (see `prompts.py`).

Per vision.md, any field that cannot be located in the PDF is recorded as
the sentinel string "data not available" — never as `None` or invented.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

NA = "data not available"


class FieldSpec(BaseModel):
    """Describes a single extraction field and the guidance shown to the LLM."""

    name: str
    notes: str = ""
    example: str = ""


# Paper-level fields (applies to the whole paper).
PAPER_FIELDS: list[FieldSpec] = [
    FieldSpec(name="doi", notes="DOI link of the paper", example="10.1007/s11121-019-01064-8"),
    FieldSpec(name="unique_id", notes="Concatenation of first-author surname + year", example="Wilcox2013"),
    FieldSpec(name="title", notes="Paper title", example="A comprehensive evaluation of …"),
    FieldSpec(name="authors", notes="Full citation string of the authors", example="Wilcox, D.W. et al."),
    FieldSpec(name="year", notes="Publication year", example="2013"),
    FieldSpec(name="journal", notes="Journal / venue", example="JAMA Pediatrics"),
    FieldSpec(name="country_region", notes="Free text or ISO-3 code", example="USA – Baltimore MD"),
    FieldSpec(name="funding_source", notes="Government, foundation, undisclosed, etc.", example="CDC"),
    FieldSpec(name="publication_type", notes="Journal article / report / thesis / pre-print", example="Peer-reviewed"),
    FieldSpec(name="design", notes="RCT; quasi-experimental; interrupted time series; before-after; cross-sectional", example="Interrupted time series"),
    FieldSpec(name="unit_of_assignment", notes="Person / neighborhood / city / state", example="City"),
    FieldSpec(name="followup_duration", notes="Follow-up duration. Prefer months.", example="48 mo"),
    FieldSpec(name="rob_tool", notes="Risk-of-bias tool used (ROBINS-I / Cochrane RoB 2 / EPOC)", example="ROBINS-I"),
    FieldSpec(name="rob_judgment", notes="Overall RoB judgment: Low / Moderate / Serious / Critical", example="Moderate"),
    FieldSpec(name="setting_type", notes="Urban / suburban / rural / mixed", example="Urban"),
    FieldSpec(name="setting_description", notes="Anything specific about the setting", example="Focused on low-income urban areas"),
    FieldSpec(name="population_description", notes="Age range, sex, socioeconomic profile", example="Residents ≥ 15 y"),
    FieldSpec(name="baseline_value", notes="Baseline value(s) in units reported by study", example="29.5 injuries / 100 000"),
    FieldSpec(name="sample_size", notes="N at baseline and follow-up", example="1 240 000 pop."),
    FieldSpec(name="intervention_category", notes="Legislation; policing; community-based; hospital-based; etc.", example="Community-based violence interruption (Cure Violence)"),
    FieldSpec(name="intervention_description", notes="Free text (1-2 sentences)", example="Street outreach workers mediate conflicts…"),
    FieldSpec(name="core_components", notes="Comma-separated: purchase restrictions, mediation, counseling, etc.", example="Outreach, conflict mediation, referrals"),
    FieldSpec(name="intensity_dose", notes="Comment on dose if available", example="15 outreach workers covering 6 neighborhoods"),
    FieldSpec(name="implementation_fidelity_reported", notes="Yes / No", example="Yes"),
    FieldSpec(name="implementation_description", notes="Implementation details that might influence outcomes", example="Some states adopted Stand Your Ground laws but they were challenged in court"),
    FieldSpec(name="comparator", notes="Comparator / control condition", example="Matched neighborhoods without program"),
    FieldSpec(name="cointerventions", notes="Concurrent changes, external shocks", example="State passed 'Safe Streets' act"),
    FieldSpec(name="implementation_barriers_facilitators", notes="Qualitative notes on barriers and facilitators", example="Community mistrust initially hindered uptake"),
    FieldSpec(name="contextual_barriers_facilitators", notes="Wider contextual barriers / facilitators", example=""),
    FieldSpec(name="notes", notes="Free-text for anything else worth noting at the paper level", example=""),
]


# Effect-size-level fields (one per effect size found in the paper).
EFFECT_SIZE_FIELDS: list[FieldSpec] = [
    FieldSpec(name="estimation_method", notes="Fixed effect model; negative binomial; difference-in-differences; etc.", example="Negative binomial"),
    FieldSpec(name="outcome_name", notes="Must match one of the outcomes declared for this table.", example="Homicide rate"),
    FieldSpec(name="outcome_reference", notes="Optional: cell / row reference inside the table", example="Row 2"),
    FieldSpec(name="outcome_domain", notes="Mortality; injury; crime; psychosocial; economic", example="Mortality"),
    FieldSpec(name="outcome_definition", notes="Definition / metric, specify unit or formula", example="Deaths / 100 000 pop."),
    FieldSpec(name="timepoints", notes="Must match one of the timepoint labels declared for this table.", example="12 mo"),
    FieldSpec(name="effect_size_raw", notes="Effect size as reported in text (e.g. IRR = 0.62)", example="Incidence Rate Ratio = 0.62"),
    FieldSpec(name="ci_or_se_raw", notes="95% CI or SE as reported in text", example="0.45-0.85"),
    FieldSpec(name="p_value", notes="P-value as reported", example="0.003"),
    FieldSpec(name="raw_data_extracted", notes="Underlying counts / means±SD", example="28 deaths vs 45 expected"),
    FieldSpec(name="direction_of_effect", notes="↓ beneficial; ↑ harmful; ↔ null", example="↓ beneficial"),
    FieldSpec(name="subgroups_analyzed", notes="Age, sex, race/ethnicity, deprivation", example="Yes – age & race"),
    FieldSpec(name="effect_heterogeneity", notes="Describe key subgroup findings", example="Larger reduction among 15-24 y males"),
    FieldSpec(name="effect_type_coded", notes="Coded effect type: IRR, OR, RR, beta, DID, SMD, mean diff, etc.", example="IRR"),
    FieldSpec(name="effect_value", notes="Numeric effect value", example="0.62"),
    FieldSpec(name="lower_ci", notes="Numeric lower confidence bound", example="0.45"),
    FieldSpec(name="upper_ci", notes="Numeric upper confidence bound", example="0.85"),
    FieldSpec(name="variance_se", notes="Numeric variance or SE", example="0.040"),
    FieldSpec(name="outcome_timeframe_months", notes="Outcome timeframe in months (numeric)", example="12"),
    FieldSpec(name="group1_mean", notes="If no effect size: intervention group mean", example=""),
    FieldSpec(name="group1_sd", notes="If no effect size: intervention group SD", example=""),
    FieldSpec(name="group1_n", notes="If no effect size: intervention group N", example=""),
    FieldSpec(name="group2_mean", notes="If no effect size: comparator group mean", example=""),
    FieldSpec(name="group2_sd", notes="If no effect size: comparator group SD", example=""),
    FieldSpec(name="group2_n", notes="If no effect size: comparator group N", example=""),
    FieldSpec(name="effect_size_notes", notes="Anything else about this effect size", example=""),
]


class EffectSize(BaseModel):
    """Single effect-size row (one estimate inside a table)."""

    model_config = ConfigDict(extra="forbid")

    estimation_method: str = NA
    outcome_name: str = NA
    outcome_reference: str = NA
    outcome_domain: str = NA
    outcome_definition: str = NA
    timepoints: str = NA
    effect_size_raw: str = NA
    ci_or_se_raw: str = NA
    p_value: str = NA
    raw_data_extracted: str = NA
    direction_of_effect: str = NA
    subgroups_analyzed: str = NA
    effect_heterogeneity: str = NA
    effect_type_coded: str = NA
    effect_value: str = NA
    lower_ci: str = NA
    upper_ci: str = NA
    variance_se: str = NA
    outcome_timeframe_months: str = NA
    group1_mean: str = NA
    group1_sd: str = NA
    group1_n: str = NA
    group2_mean: str = NA
    group2_sd: str = NA
    group2_n: str = NA
    effect_size_notes: str = NA


class TableOutcome(BaseModel):
    """One outcome reported in a table (step 2 of the review flow)."""

    model_config = ConfigDict(extra="ignore")

    outcome_name: str = NA
    outcome_domain: str = NA
    outcome_definition: str = NA


class TableTimepoint(BaseModel):
    """One timepoint (baseline, endline, 12-mo, etc.) reported in a table."""

    model_config = ConfigDict(extra="ignore")

    timepoint_label: str = NA
    outcome_timeframe_months: str = NA


class ExtractedTable(BaseModel):
    """A markdown table the LLM flagged as containing effect sizes."""

    model_config = ConfigDict(extra="ignore")

    table_label: str  # e.g. "Page 4 · Table 1" — must match a parsed table
    outcomes: list[TableOutcome] = Field(default_factory=list)
    timepoints: list[TableTimepoint] = Field(default_factory=list)
    estimates: list[EffectSize] = Field(default_factory=list)


class Paper(BaseModel):
    """Paper-level metadata plus the LLM's table-scoped effect-size extraction."""

    model_config = ConfigDict(extra="forbid")

    # Source tracking (not in the extraction template; filled by the pipeline).
    source_pdf: str = Field(description="Filename of the source PDF")

    # Paper-level fields.
    doi: str = NA
    unique_id: str = NA
    title: str = NA
    authors: str = NA
    year: str = NA
    journal: str = NA
    country_region: str = NA
    funding_source: str = NA
    publication_type: str = NA
    design: str = NA
    unit_of_assignment: str = NA
    followup_duration: str = NA
    rob_tool: str = NA
    rob_judgment: str = NA
    setting_type: str = NA
    setting_description: str = NA
    population_description: str = NA
    baseline_value: str = NA
    sample_size: str = NA
    intervention_category: str = NA
    intervention_description: str = NA
    core_components: str = NA
    intensity_dose: str = NA
    implementation_fidelity_reported: str = NA
    implementation_description: str = NA
    comparator: str = NA
    cointerventions: str = NA
    implementation_barriers_facilitators: str = NA
    contextual_barriers_facilitators: str = NA
    notes: str = NA

    tables_with_effect_sizes: list[ExtractedTable] = Field(default_factory=list)


def paper_field_names() -> list[str]:
    return [f.name for f in PAPER_FIELDS]


def effect_size_field_names() -> list[str]:
    return [f.name for f in EFFECT_SIZE_FIELDS]
