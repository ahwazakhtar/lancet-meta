# lancet-meta

PDF → structured-extraction pipeline plus reviewer web app for the Lancet
gun-violence meta-analysis described in [`vision.md`](vision.md).

## What it does

1. **Extract** — reads each paper PDF locally, asks Claude (via the Claude
   Agent SDK using your local Claude Max / Pro login) to fill in the fields
   defined in `base-data/field and paper list.xlsx`, and pulls out every
   effect size reported in the paper.
2. **Store** — extractions land in a local SQLite database (`data/app.db`)
   and a JSON cache (`data/extracted/<paper>.json`). One command pushes the
   current state to a Google Sheet (two tabs: `papers` and `effect_sizes`).
3. **Review** — a FastAPI + Jinja + HTMX web app lets multiple reviewers log
   in, confirm / modify / delete / add effect sizes per paper, and flag
   anything that needs re-extraction.

## Layout

```
extraction/        Pipeline (Claude Agent SDK + Google Sheets sync)
  schema.py        Pydantic models + field list mirrored from the xlsx
  prompts.py       The extraction prompt
  extractor.py     Claude Agent SDK driver (one paper -> one Paper object)
  storage.py       SQLite tables (papers, effect_sizes, users, audit_log)
  sheets.py        Google Sheets sync
  cli.py           `python -m extraction ...`

webapp/            Reviewer UI
  main.py          FastAPI app
  auth.py          Session cookie + bcrypt
  templates/       Jinja2 templates
  static/app.css   Pico CSS overrides

base-data/         Source-of-truth field list and paper list (xlsx) + example CSV
pdfs/              Drop PDFs here (gitignored). Filename = unique_id + .pdf
data/              Extracted JSON + SQLite (gitignored)
.credentials/      Google service-account JSON (gitignored)
```

## Setup

Requirements:

- Python 3.11+
- Node.js (the Claude Agent SDK spawns Claude Code as a subprocess)
- The `claude` CLI installed and signed in with your Claude Max / Pro
  account (`claude login`)

```bash
# 1. Install
python -m venv .venv && source .venv/bin/activate
pip install -e .

# 2. Configure
cp .env.example .env
# Edit .env: at minimum set GOOGLE_SHEET_ID, GOOGLE_SERVICE_ACCOUNT_FILE,
# and WEB_SECRET_KEY.

# 3. Google Sheets
#   - Create a service account in Google Cloud, download the JSON key
#     to .credentials/google-service-account.json
#   - Create a Google Sheet, share it (Editor) with the service account email
#   - Put the Sheet ID (from the URL) into .env

# 4. Create your first user
python -m extraction create-user --username ahwaz --admin
```

## Usage

### Extracting papers

Drop the PDFs into `./pdfs/`. Name each file so the prefix matches the
"Study" column in `base-data/field and paper list.xlsx` (e.g.
`Abdallah2021.pdf`). Then:

```bash
# Extract every PDF in ./pdfs (skips ones with a cached JSON)
python -m extraction extract

# Or a single paper
python -m extraction extract --pdf pdfs/Abdallah2021.pdf

# Quick test with the first 3 PDFs
python -m extraction extract --limit 3 --skip-existing

# See progress
python -m extraction status
```

What happens for each PDF:
1. The Agent SDK launches Claude Code locally with read access to the PDF.
2. Claude returns a JSON object with paper-level fields + a list of effect
   sizes (one row per effect size found in tables or text).
3. The JSON is cached to `data/extracted/<stem>.json`.
4. The SQLite DB is upserted: the paper row + all its effect sizes.

Missing values are recorded as the literal string `"data not available"` per
the vision doc — the LLM is instructed never to invent data.

### Pushing to Google Sheets

```bash
python -m extraction sync
```

This wipes the `papers` and `effect_sizes` tabs and writes the current
SQLite contents. Status changes from the reviewer UI are included.

### Running the review UI

```bash
lancet-web
# or
uvicorn webapp.main:app --reload --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/`, sign in with the user you created. From the
paper-list page click any paper to:

- Edit paper-level fields and save.
- Confirm / delete the paper.
- Edit each effect size in place. Save / Confirm / Delete / flag
  "Needs re-extraction".
- Add a new effect size that the pipeline missed.

Every change is appended to the `audit_log` table with the username and
timestamp.

### Flagged for re-extraction

When a reviewer flags a paper or effect size as "needs re-extraction", the
paper's status is set to `needs_reextraction`. To re-process those papers,
delete their cached JSON (and optionally their effect-size rows) and re-run
`extract`:

```bash
python -m extraction extract --pdf pdfs/Wilcox2013.pdf --no-skip-existing
```

## Notes on the LLM

- The pipeline uses [`claude-agent-sdk`](https://pypi.org/project/claude-agent-sdk/),
  which spawns `claude` (Claude Code) as a subprocess and authenticates via
  your existing local login. **No API key is required** if your `claude`
  CLI is signed into Claude Max / Pro.
- The system prompt and extraction prompt are in `extraction/prompts.py`.
- The PDF is passed by *path* to Claude (which uses its `Read` tool). PDFs
  up to Claude's normal document limits work directly; very long PDFs may
  need pre-processing (`pdfplumber` is already a dependency for future
  table-only fallbacks).

## Known limitations / next steps

- No table-extraction pre-pass yet — Claude reads PDFs directly. If you see
  table data being missed, we can add a `pdfplumber`-based pre-extraction
  step that supplies tables in markdown alongside the PDF.
- The Sheets sync is a one-way push from SQLite. If you want edits made
  directly in the Sheet to flow back, that's an additional `pull` command
  to add.
- No bulk-import of the paper list from `base-data/field and paper
  list.xlsx` yet — the pipeline indexes by PDF filename. We can add a
  "seed papers from xlsx" command if you want all 351 listed papers visible
  in the UI before their PDFs arrive.
