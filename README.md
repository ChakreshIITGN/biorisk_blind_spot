# Biorisk Blind Spot — DURC / PEPP / Governance Assessment of bioRxiv Preprints

Preliminary data, prompts, and scripts for a two-stage LLM-assisted screening pipeline
that identifies dual-use research of concern (DURC), potential pandemic pathogen (PEPP)
research, and governance-relevant technical disclosures in bioRxiv preprints.

> **Status:** Preliminary findings shared for transparency. This work is under active
> development and has not yet undergone peer review.

---

## License

This repository is released under **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)**.

You may share this material with attribution for non-commercial purposes only.
No derivatives or adaptations are permitted without explicit written permission.

Full license text: <https://creativecommons.org/licenses/by-nc-nd/4.0/>

---

## Repository Structure

```
data/
  trial_data.parquet        # trial subset (~1,000 papers) used for the study
  biorXiv24_25.parquet      # full dataset: 52,713 deduplicated English bioRxiv preprints (2024–25)

prompts/
  durc.json                 # DURC assessment prompt (D1–D9 categories)
  pepp.json                 # PEPP assessment prompt (P1–P3 criteria)
  governance.json           # Governance assessment prompt (G1–G5 categories)
  system_prompt.json        # shared system prompt skeleton

scripts/
  fetch_biorxiv.py          # fetch bioRxiv / medRxiv metadata via official API
  fetch_arxiv.py            # fetch arXiv metadata (non-biology negative-control set)
  pipeline.py               # Stage 1 lexical screening (string + semantic matching)
  string_matcher.py         # deterministic keyword/agent-name matcher
  semantic_matcher.py       # embedding-based fallback matcher
  client.py                 # Stage 2 LLM assessment — Ollama (local models)
  openai_client.py          # Stage 2 LLM assessment — OpenAI Batch API (GPT-4.1)

pyproject.toml              # Python dependencies (managed with uv)
```

---

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/<anonymous>/biorisk_blind_spot.git
cd biorisk_blind_spot
uv sync
```

For the OpenAI client, add your API key to a `.env` file:

```
OPENAI_API_KEY=sk-...
```

For the Ollama client, install [Ollama](https://ollama.com) and pull your model of choice locally.

---

## Reproducing the Pipeline

### Step 0 — Use the provided data or fetch fresh

The `data/` folder contains `biorXiv24_25.parquet` (52,713 preprints, 2024–25).
To fetch a fresh corpus:

```bash
# bioRxiv + medRxiv (default: 2024-01-01 to today)
uv run python scripts/fetch_biorxiv.py

# arXiv negative-control set (500 papers, non-biology categories)
uv run python scripts/fetch_arxiv.py
```

---

### Step 1 — Lexical screening (Stage 1)

Filters the full corpus against agent names, technical protocols, and policy terms.
Outputs a JSONL file of flagged papers.

```bash
uv run python scripts/pipeline.py \
    --input data/biorXiv24_25.parquet \
    --output jsonl
# → data/biorXiv24_25_screenv2.jsonl
```

---

### Step 2 — LLM assessment (Stage 2)

Each flagged paper is assessed across 17 criteria (D1–D9, P1–P3, G1–G5).
Three independent runs per paper with permuted category orderings.
Scores are 0–5 ordinal; results are written to DuckDB.

**Option A — OpenAI (GPT-4.1, recommended):**

```bash
# Trial run (first 50 papers)
uv run python scripts/openai_client.py \
    --parquet data/trial_data.parquet \
    --limit 50

# Full run
uv run python scripts/openai_client.py \
    --parquet data/biorXiv24_25.parquet

# Split across collaborators
uv run python scripts/openai_client.py --parquet data/biorXiv24_25.parquet --limit 500 --skip 0
uv run python scripts/openai_client.py --parquet data/biorXiv24_25.parquet --limit 500 --skip 500
```

**Option B — Ollama (local model):**

```bash
uv run python scripts/client.py \
    --db papers.duckdb \
    --workers 8 --rps 4
```

---

## Assessment Dimensions

| ID | Type | Description |
|----|------|-------------|
| D1–D9 | DURC | Nine dual-use research of concern indicators (enhanced virulence, host range, aerosol transmission, etc.) |
| P1–P3 | PEPP | Three potential pandemic pathogen markers (pathogen criterion, enhancement, transmission) |
| G1–G5 | Governance | Five governance-relevant disclosure categories (quantitative parameters, production methods, weaponisation detail, etc.) |

Scores 0–5 per criterion per paper. A paper is flagged if mean score ≥ 3 on any criterion.
Governance categories G1–G5 are operationally defined for this study and have not been externally validated.

---

## Data Fields

`biorXiv24_25.parquet` columns: `paper_id`, `doi`, `title`, `abstract`, `authors`, `date`.

`trial_data.parquet` is the ~1,000 paper subset used for this study.

---

## Citation

If you use this dataset or pipeline, please cite this repository and note the
CC BY-NC-ND 4.0 license. A preprint is forthcoming.

---

## Contact

For questions about the dataset or methodology, open an issue in this repository.
