# lancet-meta

PDF → structured-extraction pipeline plus reviewer web app for the Lancet
gun-violence meta-analysis described in [`vision.md`](vision.md).

## Architecture

```
[Your laptop]                                 [Google Sheet]                 [Railway: web app]
                                                                            (SQLite on volume)
PDFs in ./pdfs/ (any filename)
  │
  ▼ python -m extraction preprocess
data/preprocessed/<sanitized-doi>.md         ┌── papers tab ─────────────┐
  + DOI extracted from PDF text              │   effect_sizes tab        │
  + text & tables in markdown                └───────────────────────────┘
  │                                                ▲                ▼
  ▼ python -m extraction extract                   │           Admin clicks
data/extracted/<doi>.json (cache)                  │       "Import from Sheet"
data/app.db (local SQLite)                         │                ▼
  + xlsx lookup by DOI fills in                    │           Railway SQLite
    title/authors/year/journal                     │                ▼
  │                                                │           Reviewers edit
  ▼ python -m extraction publish ─────────────────►│                ▼
                                                   ◄──── "Publish to Sheet"
```

- **Local**: PDFs are renamed-by-DOI during preprocessing, parsed to markdown
  (text + every table), then Claude extracts effect sizes from the markdown
  using your **Claude Max / Pro account** via the local `claude` CLI (no API
  key needed). Bibliographic metadata is pulled from
  `base-data/field and paper list.xlsx` so the LLM doesn't have to guess.
- **Google Sheet**: the integration point. Local pipeline writes to it;
  Railway web app reads from / writes to it. Reviewers do **not** edit the
  Sheet directly — the Sheet is overwritten by both sides.
- **Railway**: hosts the FastAPI + Jinja + HTMX review UI on a managed
  container with a SQLite database on a mounted volume.

## Layout

```
extraction/        Local pipeline
  preprocess.py    PDF -> markdown keyed by DOI
  paper_list.py    Load base-data xlsx and match by DOI
  prompts.py       Claude extraction prompt
  extractor.py     Claude Agent SDK driver
  storage.py       SQLite schema + Sheet-row importer
  sheets.py        Google Sheets push/pull
  cli.py           `python -m extraction ...`

webapp/            Reviewer UI (deployable)
  main.py          FastAPI app
  auth.py          Email-only sign-in (no password)
  templates/       Jinja2 templates (Pico CSS + HTMX)
  static/app.css

base-data/         Curated field list + paper list (xlsx) + example CSV
pdfs/              Drop PDFs here, any filename (local, gitignored)
data/              Preprocessed MD, extracted JSON, SQLite (gitignored)
.credentials/      Google service-account JSON (local, gitignored)

Procfile           Railway start command
railway.toml       Railway deploy config
nixpacks.toml      Railway build config
```

## Local setup (extraction)

Requirements:

- Python 3.11+
- Node.js (the Claude Agent SDK spawns Claude Code as a subprocess)
- The `claude` CLI installed and signed in with your Claude Max / Pro
  account (`claude login`)
- A Google Cloud service account with the Sheets API enabled

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env
# Edit .env: GOOGLE_SHEET_ID, GOOGLE_SERVICE_ACCOUNT_FILE, WEB_SECRET_KEY

# Google Sheet: create one, share it (Editor) with the service account email,
# put its ID into GOOGLE_SHEET_ID.

# Drop the PDFs anywhere into ./pdfs/ (names don't matter).
mkdir -p pdfs
```

### Running the pipeline

```bash
# 1. Preprocess every PDF -> data/preprocessed/<sanitized-doi>.md
python -m extraction preprocess

# 2. Run Claude over each markdown -> SQLite + JSON cache
#    Skip papers without a DOI match in the xlsx with --require-doi.
python -m extraction extract

# 3. Push the SQLite contents to the Google Sheet
python -m extraction publish

# Other useful commands
python -m extraction status                  # progress summary
python -m extraction extract --limit 3       # try a few PDFs first
python -m extraction reload-cache            # rebuild SQLite from cached JSON
```

Notes:
- **DOIs are required for xlsx matching** — papers without a discoverable DOI
  still get extracted but won't have curated title/authors/year prefilled.
- Missing fields are stored as the literal string `"data not available"` per
  the protocol. Claude is instructed never to invent values.

## Deploying the web app to Railway

The deployed web app runs **independently** of your local pipeline. It uses
the same code, just hits a different SQLite path (on a mounted volume) and
talks to the same Google Sheet.

### 1. Push this repo

Railway can deploy from GitHub.

### 2. Create the project

In the Railway dashboard:

1. **New project → Deploy from GitHub repo** → point at this repo.
2. Railway detects the `nixpacks.toml` / `Procfile` and starts a Python build.
3. **Add a Volume**: in the service settings, attach a 1 GB volume with
   mount path `/data`.

### 3. Environment variables

In the service's Variables tab, set:

| Variable | Value |
| -------- | ----- |
| `WEB_DB_PATH` | `/data/app.db` |
| `WEB_SECRET_KEY` | a long random string (`openssl rand -hex 32`) |
| `GOOGLE_SHEET_ID` | the ID of the Sheet you created |
| `GOOGLE_SERVICE_ACCOUNT_JSON` | **paste the raw JSON** of the service-account key (one line, no file) |
| `PAPERS_SHEET_NAME` | `papers` (optional, default) |
| `EFFECT_SIZES_SHEET_NAME` | `effect_sizes` (optional, default) |

Railway sets `PORT` automatically; you don't need to set `WEB_HOST` or
`WEB_PORT`.

### 4. Generate a public domain

Settings → Networking → **Generate Domain**. Railway gives you something
like `your-app.up.railway.app`.

### 5. Create reviewer accounts

Sign-in is **email-only** — no password. Anyone whose email is on the
reviewer list can sign in; the email is recorded against every edit in the
audit log. This is for attribution, not for security.

The simplest way to bootstrap the first admin on Railway is to set
`ADMIN_BOOTSTRAP_EMAIL=your@email.com` in the service's env vars. On the
first start, if no users exist yet, that email is auto-added as an admin.
You can then sign in and add the rest of your team from `/admin`.

(You can also run `python -m extraction add-user --email x@y.com --admin`
via `railway run` if you prefer a CLI.)

### 6. Usage on the deployed app

- Reviewers visit the domain and sign in (email only).
- The dashboard shows every paper with its checkout state. Anyone who's
  currently working on a paper appears next to it (`you` or their email).
- A reviewer picks an available paper, clicks **Review**, then **Check
  out** — that locks the paper to them. Only they (and admins) can edit it
  until they **Check in**.
- A held-by-me paper has an extra button: **Import this paper from Sheet**.
  That refreshes just that paper's fields and effect sizes from the latest
  Sheet contents — useful after the local pipeline re-extracts that PDF.
- Admins can **bulk import** from the Sheet via `/admin`. The bulk import
  always skips papers that someone has checked out, so in-progress work
  isn't trashed.
- When reviewers finish a batch of edits, an admin clicks **Publish web app
  → Sheet** to overwrite the Sheet with the current state.

### Notes / caveats

- The SQLite DB on the Railway volume is the source of truth for **reviewer
  edits**. Each "Import from Sheet" **replaces** the entire DB with the
  Sheet contents — so always **publish before importing** if there are
  unsaved edits, otherwise they'll be overwritten by the Sheet.
- Railway free tier is fine for ~351 papers / a few thousand effect sizes.
- The CLI commands (`extract`, `publish`, etc.) are **not** intended to be
  run on Railway routinely — the local pipeline does the LLM work, the
  cloud app does the review work. The only CLI usage on Railway is the
  optional `add-user`, and even that's usually unnecessary if you use
  `ADMIN_BOOTSTRAP_EMAIL`.

## How the data flows back to the Sheet

Reviewer actions:
- **Confirm** — marks a row as confirmed (status badge in the UI).
- **Edit + Save** — overwrites fields; status becomes `modified`.
- **Delete** — soft-deletes; row is hidden from list and excluded from
  publishes.
- **Add effect size** — inserts a new row tied to that paper.
- **Flag re-extraction** — sets status to `needs_reextraction` so the local
  pipeline can re-process that paper.

Every action is appended to the `audit_log` table with the reviewer's
email and timestamp. The Sheet column layout is in `extraction/sheets.py`
(`PAPER_SHEET_COLUMNS`, `EFFECT_SIZE_SHEET_COLUMNS`).
