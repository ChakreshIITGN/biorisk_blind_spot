"""
Stage 2 biosecurity assessment pipeline — OpenAI Batch API variant.

Submits DURC / PEPP / Governance assessments via the OpenAI Batch API using
gpt-5.1 with prompt caching and logprobs.  All API calls are asynchronous:
requests are written to a JSONL file, uploaded, submitted as a batch job, and
polled until complete.  Results are then downloaded and written to DuckDB.

Prompt caching works automatically: within each batch file every request shares
an identical system prompt, so OpenAI caches it after the first request and
charges the cached-input rate ($0.0625/M) for all subsequent papers in that file.

Token accounting columns stored per row in DuckDB:
  in_tokens      — total prompt tokens charged
  out_tokens     — completion tokens charged
  cached_tokens  — prompt tokens served from cache (subset of in_tokens)
  score_logprobs — JSON: {category: {lp: float, alt: {score: lp, ...}}}

No reasoning tokens: gpt-5.1 is not an o-series model.

Usage:
    uv run python openai_client.py --parquet data/trial_data.parquet
    uv run python openai_client.py --parquet data/trial_data.parquet --limit 50
    uv run python openai_client.py --parquet data/trial_data.parquet --batch-size 100

Split across collaborators (papers ordered by paper_id):
    # Collaborator A — papers 1–100
    uv run python openai_client.py --parquet data/trial_data.parquet --limit 100 --batch-size 10
    # Collaborator B — papers 101–400
    uv run python openai_client.py --parquet data/trial_data.parquet --skip 100 --limit 300 --batch-size 10
    # Collaborator C — papers 401–end
    uv run python openai_client.py --parquet data/trial_data.parquet --skip 400
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import random
import time
from pathlib import Path
from typing import Annotated

import duckdb
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────────────

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")

# Fail at startup if the API key is missing — avoids loading all data and
# initialising the DB before discovering the key is absent.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY_1")
if not OPENAI_API_KEY:
    raise EnvironmentError(
        "OPENAI_API_KEY_1 is not set. Add it to your .env file."
    )

# OpenAI hard limit for top_logprobs is 5.  Scores are 0–5 (6 values), so the
# least-likely score at each position is silently absent — acceptable because
# if it is not in the top 5, its probability is negligible.
TOP_LOGPROBS = 5

# gpt-5.1 requires max_completion_tokens (max_tokens is rejected with HTTP 400).
# Per-assessment caps: model emits valid JSON well within these limits, then pads
# with whitespace. Lower caps stop charging for whitespace sooner; the parser
# recovers via content.strip() + json.loads().
MAX_COMPLETION_TOKENS: dict[str, int] = {
    "pepp":       400,   # 3 categories
    "governance": 600,   # 5 categories
    "durc":       900,   # 9 categories
}

POLL_INTERVAL_MIN = 30    # seconds — minimum poll wait
POLL_INTERVAL_MAX = 3600  # seconds — cap at 1 hour for long batches
POLL_MAX_WAIT     = 86400 # 24-hour OpenAI SLA


# ── Pydantic output schemas ───────────────────────────────────────────────────
# One model per assessment type.  Used for:
#   (a) building the JSON schema sent to OpenAI's structured output API
#   (b) validating parsed LLM output before writing to DuckDB
#
# Score is range-constrained 0–5 so hallucinated values (e.g. 7, -1) are
# caught at validation time and stored as errors rather than silently written.

ScoreInt = Annotated[int, Field(ge=0, le=5)]

class DURCScore(BaseModel):
    score:    ScoreInt
    evidence: str

class DURCResult(BaseModel):
    id: str
    D1: DURCScore; D2: DURCScore; D3: DURCScore
    D4: DURCScore; D5: DURCScore; D6: DURCScore
    D7: DURCScore; D8: DURCScore; D9: DURCScore

class PEPPScore(BaseModel):
    score:    ScoreInt
    evidence: str

class PEPPResult(BaseModel):
    id: str
    P1: PEPPScore; P2: PEPPScore; P3: PEPPScore

class GovernanceScore(BaseModel):
    score:    ScoreInt
    evidence: str

class GovernanceResult(BaseModel):
    id: str
    G1: GovernanceScore; G2: GovernanceScore; G3: GovernanceScore
    G4: GovernanceScore; G5: GovernanceScore

# Lookup tables — avoids repeating if/elif chains throughout.
_CATS: dict[str, list[str]] = {
    "durc":       ["D1","D2","D3","D4","D5","D6","D7","D8","D9"],
    "pepp":       ["P1","P2","P3"],
    "governance": ["G1","G2","G3","G4","G5"],
}
_RESULT_TYPES: dict[str, type[BaseModel]] = {
    "durc": DURCResult, "pepp": PEPPResult, "governance": GovernanceResult,
}
_SCORE_CLS = {
    "durc": DURCScore, "pepp": PEPPScore, "governance": GovernanceScore,
}
_ZERO_SCORE = {"score": 0, "evidence": ""}


# ── Sparse JSON schema ────────────────────────────────────────────────────────
# OpenAI strict structured output enforces two rules:
#   1. Every key in `properties` must appear in `required`.
#   2. `additionalProperties` must be false at every level.
#
# To allow sparse output (model omits zero-score categories) we wrap every
# non-id category field as anyOf([original_ref, null]).  The model can then
# output either a valid score object OR null for that category.
# All fields stay in `required` — strict mode is fully satisfied.
# Null values are zero-filled in parse_result_line() before Pydantic validation.
#
# Within sub-models (DURCScore etc.) we pop `required` so `score` and
# `evidence` are not individually enforced — this permits the model to omit
# `evidence` when returning a null category (belt-and-suspenders).

def _sparse_schema(result_type: type[BaseModel], assessment: str) -> dict:
    schema = copy.deepcopy(result_type.model_json_schema())
    schema["additionalProperties"] = False

    # Sub-models (DURCScore / PEPPScore / GovernanceScore) must keep their
    # `required` list — OpenAI strict mode enforces required at every $defs level.
    for sub in schema.get("$defs", {}).values():
        sub["additionalProperties"] = False

    # Rename property keys from internal IDs (D1, P1 …) to natural-language labels
    # ("enhanced virulence", "pathogen criterion" …) so the LLM outputs readable
    # keys.  _translate_to_ids() maps them back before Pydantic validation.
    id_to_label = _ID_TO_LABEL[assessment]
    old_props   = schema["properties"]
    new_props   = {}
    for key, val in old_props.items():
        new_key = id_to_label.get(key, key)   # "id" → "id" (unchanged)
        if new_key != "id":
            val = {"anyOf": [val, {"type": "null"}]}
        new_props[new_key] = val
    schema["properties"] = new_props
    schema["required"]   = [id_to_label.get(k, k) for k in schema.get("required", [])]
    return schema


# ── Zero-fill ────────────────────────────────────────────────────────────────
# After receiving sparse LLM output, replace any null or missing category with
# score=0 so Pydantic validation against the full model succeeds.

def _fill_zeros(assessment: str, raw: dict) -> dict:
    for k in _CATS[assessment]:
        # Treat both absent keys and explicit null the same way.
        if raw.get(k) is None:
            raw[k] = _ZERO_SCORE
    return raw

def _make_zero_result(assessment: str, result_id: str) -> BaseModel:
    """All-zero result — used when LLM returns the NONE FOUND sentinel."""
    return _RESULT_TYPES[assessment](
        id=result_id,
        **{k: _SCORE_CLS[assessment](**_ZERO_SCORE) for k in _CATS[assessment]},
    )


# ── Prompt loading ────────────────────────────────────────────────────────────

_PROMPTS_DIR = Path(__file__).parent / "prompts"

with open(_PROMPTS_DIR / "durc.json")       as _f: _DURC = json.load(_f)
with open(_PROMPTS_DIR / "pepp.json")       as _f: _PEPP = json.load(_f)
with open(_PROMPTS_DIR / "governance.json") as _f: _GOV  = json.load(_f)

# ── Label ↔ ID mappings ───────────────────────────────────────────────────────
# The LLM outputs category names in natural lowercase ("enhanced virulence").
# These are translated back to IDs (D1, P1, G1 …) before Pydantic validation
# and DB insert — the DB schema and Pydantic models are untouched.

_ID_TO_LABEL: dict[str, dict[str, str]] = {
    "durc":       {id_: cat["name"].lower() for id_, cat in _DURC["categories"].items()},
    "pepp":       {id_: cat["name"].lower() for id_, cat in _PEPP["criteria"].items()},
    "governance": {id_: cat["name"].lower() for id_, cat in _GOV["categories"].items()},
}
_LABEL_TO_ID: dict[str, dict[str, str]] = {
    asmnt: {v: k for k, v in m.items()}
    for asmnt, m in _ID_TO_LABEL.items()
}
# Label-ordered lists (D1→D9, P1→P3, G1→G5) for permutation generation.
_LABEL_ORDER: dict[str, list[str]] = {
    asmnt: list(m.values()) for asmnt, m in _ID_TO_LABEL.items()
}
# Category dicts keyed by label — used to build system prompt sections.
_CATS_BY_LABEL: dict[str, dict] = {
    "durc":       {cat["name"].lower(): cat for cat in _DURC["categories"].values()},
    "pepp":       {cat["name"].lower(): cat for cat in _PEPP["criteria"].values()},
    "governance": {cat["name"].lower(): cat for cat in _GOV["categories"].values()},
}


def _translate_to_ids(assessment: str, raw: dict) -> dict:
    """Rename label keys ("enhanced virulence") to ID keys ("D1") in-place."""
    mapping = _LABEL_TO_ID[assessment]
    return {mapping.get(k, k): v for k, v in raw.items()}


# ── Permutation sets ──────────────────────────────────────────────────────────
# Same seed as client.py — both pipelines use identical category orderings so
# results from Ollama and OpenAI runs are directly comparable.

_RNG = random.Random(42)

def _generate_permutations(keys: list[str], n: int) -> list[list[str]]:
    """Select n permutations minimising positional repetition across runs."""
    pool = list(itertools.permutations(keys))
    _RNG.shuffle(pool)
    selected: list[list[str]] = []
    pos_counts: list[dict[str, int]] = [{} for _ in keys]
    for perm in pool:
        if len(selected) == n:
            break
        if all(pos_counts[i].get(k, 0) < 2 for i, k in enumerate(perm)):
            selected.append(list(perm))
            for i, k in enumerate(perm):
                pos_counts[i][k] = pos_counts[i].get(k, 0) + 1
    idx = 0
    while len(selected) < n:
        selected.append(list(pool[idx % len(pool)]))
        idx += 1
    return selected

_DURC_PERMS = _generate_permutations(_LABEL_ORDER["durc"],       n=3)
_PEPP_PERMS = [                                                  # Latin square
    _LABEL_ORDER["pepp"],
    _LABEL_ORDER["pepp"][1:] + _LABEL_ORDER["pepp"][:1],
    _LABEL_ORDER["pepp"][2:] + _LABEL_ORDER["pepp"][:2],
]
_GOV_PERMS  = _generate_permutations(_LABEL_ORDER["governance"], n=3)


# ── Prompt assembly ───────────────────────────────────────────────────────────

def _fmt_examples(examples: list) -> str:
    return "\n\n".join(
        f"[{ex['label']}]\nInput: {json.dumps(ex['input'])}\n"
        f"Output: {json.dumps(ex['output']) if isinstance(ex['output'], dict) else ex['output']}"
        for ex in examples
    )

def _fmt_section(items_by_label: dict, labels: list[str]) -> str:
    # [label] is the exact JSON key the LLM must use in its output.
    return "\n\n".join(
        f"[{lbl}] {items_by_label[lbl]['name']}\n"
        f"{items_by_label[lbl]['description']}\n"
        f"Keywords: {items_by_label[lbl]['keywords']}"
        for lbl in labels
    )

def _build_durc_prompt(perm: list[str]) -> str:
    return (_DURC["instruction"]
            .replace("{examples}",           _fmt_examples(_DURC["examples"]))
            .replace("{ordered_categories}", _fmt_section(_CATS_BY_LABEL["durc"], perm)))

def _build_pepp_prompt(perm: list[str]) -> str:
    return (_PEPP["instruction"]
            .replace("{examples}",         _fmt_examples(_PEPP["examples"]))
            .replace("{ordered_criteria}", _fmt_section(_CATS_BY_LABEL["pepp"], perm)))

def _build_gov_prompt(perm: list[str]) -> str:
    return (_GOV["instruction"]
            .replace("{examples}",           _fmt_examples(_GOV["examples"]))
            .replace("{ordered_categories}", _fmt_section(_CATS_BY_LABEL["governance"], perm)))

# Pre-build all 9 system prompts at startup — one per (assessment, run) pair.
# All requests within a batch file share the same system prompt, which triggers
# OpenAI's automatic prompt caching after the first request in each file.
_SYSTEM_PROMPTS: dict[tuple[str, int], str] = {
    ("durc",       1): _build_durc_prompt(_DURC_PERMS[0]),
    ("durc",       2): _build_durc_prompt(_DURC_PERMS[1]),
    ("durc",       3): _build_durc_prompt(_DURC_PERMS[2]),
    ("pepp",       1): _build_pepp_prompt(_PEPP_PERMS[0]),
    ("pepp",       2): _build_pepp_prompt(_PEPP_PERMS[1]),
    ("pepp",       3): _build_pepp_prompt(_PEPP_PERMS[2]),
    ("governance", 1): _build_gov_prompt(_GOV_PERMS[0]),
    ("governance", 2): _build_gov_prompt(_GOV_PERMS[1]),
    ("governance", 3): _build_gov_prompt(_GOV_PERMS[2]),
}


# ── Logprob extraction ────────────────────────────────────────────────────────
# OpenAI returns one logprob entry per output token.  We only want the score
# token — the digit (0–5) that immediately follows "score": in the JSON output.
#
# top_logprobs=5 returns the 5 most-probable alternatives at that position.
# Since scores span 0–5 (6 values), the 6th least-likely value may be absent,
# which is acceptable — it means its probability is negligible.
#
# Result stored as: {"D4": {"lp": -0.04, "alt": {"0":-9.1, "2":-3.2, ...}}}

def _extract_score_logprobs(logprobs_content: list[dict], assessment: str) -> dict:
    result: dict[str, dict] = {}
    labels       = _LABEL_ORDER[assessment]   # ["enhanced virulence", ...]
    id_to_label  = _ID_TO_LABEL[assessment]   # to store result keyed by ID
    label_to_id  = _LABEL_TO_ID[assessment]
    tokens       = [t["token"] for t in logprobs_content]
    score_digits = {"0","1","2","3","4","5"}

    for i, tok in enumerate(tokens):
        if tok not in score_digits:
            continue

        # The non-whitespace token immediately before the digit must be ":"
        prev = i - 1
        while prev >= 0 and tokens[prev].strip() == "":
            prev -= 1
        if prev < 0 or tokens[prev] != ":":
            continue

        # Labels are multi-token ("enhanced", " virulence") so concatenate the
        # window and search for the quoted label string within it.
        window_str = "".join(tokens[max(0, i - 30): i])
        label = next((lbl for lbl in labels if f'"{lbl}"' in window_str), None)
        if label is None:
            continue
        cat_id = label_to_id[label]
        if cat_id in result:
            continue

        top = logprobs_content[i].get("top_logprobs", [])
        result[cat_id] = {
            "lp":  logprobs_content[i]["logprob"],
            "alt": {t["token"]: t["logprob"] for t in top if t["token"] in score_digits},
        }

    return result


# ── Database ──────────────────────────────────────────────────────────────────

def init_db(db_path: str) -> duckdb.DuckDBPyConnection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(db_path)

    con.execute("""
        CREATE TABLE IF NOT EXISTS durc_results (
            paper_id           INTEGER,
            id                 TEXT,
            run                INTEGER,            -- 1–3
            presentation_order TEXT,               -- e.g. "D4,D1,D7,..."
            D1_score INTEGER, D1_evidence TEXT,
            D2_score INTEGER, D2_evidence TEXT,
            D3_score INTEGER, D3_evidence TEXT,
            D4_score INTEGER, D4_evidence TEXT,
            D5_score INTEGER, D5_evidence TEXT,
            D6_score INTEGER, D6_evidence TEXT,
            D7_score INTEGER, D7_evidence TEXT,
            D8_score INTEGER, D8_evidence TEXT,
            D9_score INTEGER, D9_evidence TEXT,
            error          TEXT,
            in_tokens      INTEGER,
            out_tokens     INTEGER,
            cached_tokens  INTEGER,                -- prompt tokens served from cache
            score_logprobs TEXT,                   -- JSON: per-category score logprobs
            assessed_at    TIMESTAMP DEFAULT current_timestamp,
            PRIMARY KEY (paper_id, run)
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS pepp_results (
            paper_id           INTEGER,
            id                 TEXT,
            run                INTEGER,
            presentation_order TEXT,
            P1_score INTEGER, P1_evidence TEXT,
            P2_score INTEGER, P2_evidence TEXT,
            P3_score INTEGER, P3_evidence TEXT,
            error          TEXT,
            in_tokens      INTEGER,
            out_tokens     INTEGER,
            cached_tokens  INTEGER,
            score_logprobs TEXT,
            assessed_at    TIMESTAMP DEFAULT current_timestamp,
            PRIMARY KEY (paper_id, run)
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS governance_results (
            paper_id           INTEGER,
            id                 TEXT,
            run                INTEGER,
            presentation_order TEXT,
            G1_score INTEGER, G1_evidence TEXT,
            G2_score INTEGER, G2_evidence TEXT,
            G3_score INTEGER, G3_evidence TEXT,
            G4_score INTEGER, G4_evidence TEXT,
            G5_score INTEGER, G5_evidence TEXT,
            error          TEXT,
            in_tokens      INTEGER,
            out_tokens     INTEGER,
            cached_tokens  INTEGER,
            score_logprobs TEXT,
            assessed_at    TIMESTAMP DEFAULT current_timestamp,
            PRIMARY KEY (paper_id, run)
        )
    """)

    return con


def _already_done(con: duckdb.DuckDBPyConnection,
                  assessment: str, paper_ids: list[int], run: int) -> set[int]:
    """
    Return the set of paper_ids that already have a successful result for this
    (assessment, run) pair.  Used to skip re-submission on resume.
    """
    tbl          = f"{assessment}_results"
    placeholders = ",".join("?" * len(paper_ids))
    rows = con.execute(
        f"SELECT paper_id FROM {tbl} "
        f"WHERE run = ? AND paper_id IN ({placeholders}) AND error IS NULL",
        [run, *paper_ids],
    ).fetchall()
    return {r[0] for r in rows}


# ── DB write helpers ──────────────────────────────────────────────────────────
# INSERT OR REPLACE — re-runs overwrite previous results cleanly.
# Null-pad count for the error path is derived from _CATS so it stays correct
# if categories are ever added (score + evidence = 2 columns per category).

def _insert_durc(con, paper_id, id_, run, perm, result, error,
                 in_tok, out_tok, cached_tok, lp_json):
    order     = ",".join(_LABEL_TO_ID["durc"][l] for l in perm)
    null_cats = [None] * (len(_CATS["durc"]) * 2)   # score + evidence per category
    vals = (
        [paper_id, id_, run, order,
         result.D1.score, result.D1.evidence, result.D2.score, result.D2.evidence,
         result.D3.score, result.D3.evidence, result.D4.score, result.D4.evidence,
         result.D5.score, result.D5.evidence, result.D6.score, result.D6.evidence,
         result.D7.score, result.D7.evidence, result.D8.score, result.D8.evidence,
         result.D9.score, result.D9.evidence,
         error, in_tok, out_tok, cached_tok, lp_json]
        if result else
        [paper_id, id_, run, order, *null_cats, error, in_tok, out_tok, cached_tok, lp_json]
    )
    con.execute("""
        INSERT OR REPLACE INTO durc_results
        (paper_id, id, run, presentation_order,
         D1_score, D1_evidence, D2_score, D2_evidence,
         D3_score, D3_evidence, D4_score, D4_evidence,
         D5_score, D5_evidence, D6_score, D6_evidence,
         D7_score, D7_evidence, D8_score, D8_evidence,
         D9_score, D9_evidence,
         error, in_tokens, out_tokens, cached_tokens, score_logprobs)
        VALUES (?,?,?,?, ?,?,?,?, ?,?,?,?, ?,?,?,?, ?,?,?,?, ?,?, ?,?,?,?,?)
    """, vals)


def _insert_pepp(con, paper_id, id_, run, perm, result, error,
                 in_tok, out_tok, cached_tok, lp_json):
    order     = ",".join(_LABEL_TO_ID["pepp"][l] for l in perm)
    null_cats = [None] * (len(_CATS["pepp"]) * 2)
    vals = (
        [paper_id, id_, run, order,
         result.P1.score, result.P1.evidence,
         result.P2.score, result.P2.evidence,
         result.P3.score, result.P3.evidence,
         error, in_tok, out_tok, cached_tok, lp_json]
        if result else
        [paper_id, id_, run, order, *null_cats, error, in_tok, out_tok, cached_tok, lp_json]
    )
    con.execute("""
        INSERT OR REPLACE INTO pepp_results
        (paper_id, id, run, presentation_order,
         P1_score, P1_evidence, P2_score, P2_evidence, P3_score, P3_evidence,
         error, in_tokens, out_tokens, cached_tokens, score_logprobs)
        VALUES (?,?,?,?, ?,?,?,?, ?,?, ?,?,?,?,?)
    """, vals)


def _insert_governance(con, paper_id, id_, run, perm, result, error,
                        in_tok, out_tok, cached_tok, lp_json):
    order     = ",".join(_LABEL_TO_ID["governance"][l] for l in perm)
    null_cats = [None] * (len(_CATS["governance"]) * 2)
    vals = (
        [paper_id, id_, run, order,
         result.G1.score, result.G1.evidence,
         result.G2.score, result.G2.evidence,
         result.G3.score, result.G3.evidence,
         result.G4.score, result.G4.evidence,
         result.G5.score, result.G5.evidence,
         error, in_tok, out_tok, cached_tok, lp_json]
        if result else
        [paper_id, id_, run, order, *null_cats, error, in_tok, out_tok, cached_tok, lp_json]
    )
    con.execute("""
        INSERT OR REPLACE INTO governance_results
        (paper_id, id, run, presentation_order,
         G1_score, G1_evidence, G2_score, G2_evidence,
         G3_score, G3_evidence, G4_score, G4_evidence, G5_score, G5_evidence,
         error, in_tokens, out_tokens, cached_tokens, score_logprobs)
        VALUES (?,?,?,?, ?,?,?,?, ?,?,?,?, ?,?, ?,?,?,?,?)
    """, vals)


_INSERT_FN = {
    "durc": _insert_durc, "pepp": _insert_pepp, "governance": _insert_governance,
}


# ── JSONL batch builder ───────────────────────────────────────────────────────

def _make_request_line(custom_id: str, system_prompt: str,
                        user_content: str, assessment: str) -> dict:
    """
    One line of the OpenAI batch JSONL file.

    custom_id encodes routing metadata: {paper_id}__{assessment}__{run}
    Parsed back in parse_result_line() to route results to the right DB row.

    Key parameters:
      max_completion_tokens — gpt-5.1 requires this; max_tokens returns HTTP 400
      logprobs=True         — return per-token log-probabilities in the response
      top_logprobs=5        — top-5 alternatives at each token (API hard limit)
      response_format       — strict structured output with our nullable schema
    """
    return {
        "custom_id": custom_id,
        "method":    "POST",
        "url":       "/v1/chat/completions",
        "body": {
            "model":                 OPENAI_MODEL,
            "max_completion_tokens": MAX_COMPLETION_TOKENS[assessment],
            "logprobs":              True,
            "top_logprobs":          TOP_LOGPROBS,
            "response_format": {
                "type":        "json_schema",
                "json_schema": {
                    "name":   f"{assessment}_result",
                    "strict": True,
                    "schema": _sparse_schema(_RESULT_TYPES[assessment], assessment),
                },
            },
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_content},
            ],
        },
    }


def build_batch_jsonl(papers: list[dict], assessment: str, run: int) -> list[dict]:
    """
    Build one JSONL batch (list of request dicts) for a given (assessment, run).
    All requests share the same system prompt — this triggers OpenAI's automatic
    prompt caching after the first request in each file.
    """
    system = _SYSTEM_PROMPTS[(assessment, run)]
    return [
        _make_request_line(
            custom_id     = f"{p['paper_id']}__{assessment}__{run}",
            system_prompt = system,
            user_content  = json.dumps({"id":       str(p["paper_id"]),
                                        "title":    p["title"],
                                        "abstract": p["abstract"]}),
            assessment    = assessment,
        )
        for p in papers
    ]


# ── Batch API submission & polling ────────────────────────────────────────────

def submit_batch(client: OpenAI, lines: list[dict], description: str) -> str:
    """Upload JSONL to Files API, create batch job.  Returns batch_id."""
    jsonl_bytes = "\n".join(json.dumps(l) for l in lines).encode()
    uploaded = client.files.create(
        file=("batch.jsonl", jsonl_bytes, "application/jsonl"),
        purpose="batch",
    )
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": description},
    )
    print(f"    submitted {batch.id}  ({len(lines)} requests)  [{description}]", flush=True)
    return batch.id


def poll_batch(client: OpenAI, batch_id: str):
    """
    Block until the batch reaches a terminal state.  Returns the batch object.
    Uses exponential backoff (30s → 1h) to avoid unnecessary status calls on
    long-running batches while remaining responsive for short ones.
    """
    start   = time.monotonic()
    attempt = 0
    while True:
        batch  = client.batches.retrieve(batch_id)
        counts = batch.request_counts
        print(
            f"    [{batch_id}] {batch.status}  "
            f"total={counts.total}  done={counts.completed}  failed={counts.failed}  "
            f"elapsed={time.monotonic()-start:.0f}s",
            flush=True,
        )
        if batch.status in ("completed", "failed", "expired", "cancelled"):
            return batch
        if time.monotonic() - start > POLL_MAX_WAIT:
            raise TimeoutError(f"Batch {batch_id} did not complete within 24h")

        # Exponential backoff: 30s, 60s, 120s, … capped at 1h
        interval = min(POLL_INTERVAL_MIN * (2 ** min(attempt, 7)), POLL_INTERVAL_MAX)
        attempt += 1
        time.sleep(interval)


def download_results(client: OpenAI, batch) -> list[dict]:
    """
    Download output JSONL and delete the uploaded input file.

    The input file is ALWAYS deleted in a finally block — OpenAI charges
    $0.10/GB/day for stored files, so leaking them on batch failure accumulates
    costs.  The output file is not deleted here (useful for debugging).
    """
    try:
        rows = []
        if batch.output_file_id:
            rows = [
                json.loads(line)
                for line in client.files.content(batch.output_file_id).text.splitlines()
                if line.strip()
            ]
        return rows
    finally:
        # Delete regardless of whether the batch succeeded or failed.
        if batch.input_file_id:
            try:
                client.files.delete(batch.input_file_id)
            except Exception:
                pass


# ── Result parsing ────────────────────────────────────────────────────────────

def parse_result_line(line: dict) -> tuple:
    """
    Parse one output line from the downloaded batch JSONL.

    Returns:
        (paper_id, assessment, run, result_or_None, error_or_None,
         in_tokens, out_tokens, cached_tokens, score_logprobs_json)

    Handles three failure modes:
      1. line["error"] is set        — batch-level failure (network, timeout)
      2. response.status_code != 200 — API rejected the request (bad param etc.)
         Note: in the batch JSONL, HTTP errors have line["error"]=null but
         response.status_code != 200 — both paths must be checked.
      3. JSON parse / Pydantic error — model returned malformed output
    """
    paper_id_s, assessment, run_s = line["custom_id"].split("__")
    paper_id = int(paper_id_s)
    run      = int(run_s)

    # Failure mode 1: batch-level error
    if line.get("error"):
        return paper_id, assessment, run, None, str(line["error"]), 0, 0, 0, "null"

    resp        = line["response"]
    status_code = resp.get("status_code", 200)
    body        = resp.get("body", {})

    # Failure mode 2: HTTP 4xx/5xx — error lives in body["error"], not line["error"]
    if status_code != 200:
        msg = body.get("error", {}).get("message", f"HTTP {status_code}")
        return paper_id, assessment, run, None, msg, 0, 0, 0, "null"

    usage      = body.get("usage", {})
    in_tok     = usage.get("prompt_tokens", 0)
    out_tok    = usage.get("completion_tokens", 0)
    cached_tok = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)

    choice  = body["choices"][0]
    content = choice["message"]["content"].strip()

    # Extract score logprobs before parsing content
    lp_content    = (choice.get("logprobs") or {}).get("content") or []
    score_lp      = _extract_score_logprobs(lp_content, assessment)
    score_lp_json = json.dumps(score_lp) if score_lp else "null"

    # Failure mode 3: parse / validation error
    try:
        if content == "NONE FOUND":
            result = _make_zero_result(assessment, paper_id_s)
        else:
            raw = json.loads(content)
            # Normalise id to string — Pydantic would silently coerce an integer
            # id to string, masking cases where the model returned the wrong paper.
            raw["id"] = str(raw.get("id", paper_id_s))
            raw    = _translate_to_ids(assessment, raw)  # labels → IDs (D1, P1 …)
            raw    = _fill_zeros(assessment, raw)         # null/missing → score=0
            result = _RESULT_TYPES[assessment].model_validate(raw)
            if result.id != paper_id_s:
                raise ValueError(f"id mismatch: sent {paper_id_s!r}, got {result.id!r}")
        return paper_id, assessment, run, result, None, in_tok, out_tok, cached_tok, score_lp_json
    except Exception as e:
        return paper_id, assessment, run, None, str(e), in_tok, out_tok, cached_tok, score_lp_json


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Stage 2 biosecurity assessment via OpenAI Batch API (gpt-5.1)."
    )
    parser.add_argument("--parquet",    required=True,
                        help="Parquet file of flagged papers (paper_id, title, abstract)")
    parser.add_argument("--db",         default=None,
                        help="DuckDB output file (auto-named from paper range if omitted)")
    parser.add_argument("--limit",      type=int, default=None,
                        help="Cap number of papers (useful for test runs)")
    parser.add_argument("--skip",       type=int, default=0,
                        help="Skip the first N papers (by paper_id order); use to split work across collaborators")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Papers per batch submission (default 50)")
    args = parser.parse_args()

    # Auto-name DB from paper range so collaborators' files don't overwrite each other.
    if args.db is None:
        start = args.skip + 1
        end   = args.skip + args.limit if args.limit else "end"
        args.db = f"results/openai/results_{start}_{end}.duckdb"
    Path(args.db).parent.mkdir(parents=True, exist_ok=True)

    client = OpenAI(api_key=OPENAI_API_KEY)
    con    = init_db(args.db)

    # Load papers from parquet via DuckDB — no pandas dependency
    # OFFSET skips the first N papers; LIMIT caps the total loaded after the skip.
    limit_clause  = f" LIMIT {args.limit}"   if args.limit else ""
    offset_clause = f" OFFSET {args.skip}"   if args.skip  else ""
    rows   = duckdb.sql(
        f"SELECT paper_id, title, abstract FROM read_parquet('{args.parquet}') "
        f"ORDER BY paper_id" + limit_clause + offset_clause
    ).fetchall()
    papers = [{"paper_id": r[0], "title": r[1], "abstract": r[2]} for r in rows]

    if not papers:
        print("No papers found in parquet.")
        con.close()
        return

    print(
        f"Papers loaded    : {len(papers)}\n"
        f"Model            : {OPENAI_MODEL}\n"
        f"Batch size       : {args.batch_size}\n"
        f"Output DB        : {args.db}\n",
        flush=True,
    )

    perms_map = {
        "durc":       _DURC_PERMS,
        "pepp":       _PEPP_PERMS,
        "governance": _GOV_PERMS,
    }

    # Process papers in chunks of --batch-size.
    # Per chunk: submit up to 9 batch jobs (3 assessments × 3 runs).
    # Already-completed (paper_id, run) pairs are skipped — safe to resume
    # after a crash without re-spending tokens on finished papers.
    for chunk_start in range(0, len(papers), args.batch_size):
        chunk     = papers[chunk_start: chunk_start + args.batch_size]
        chunk_ids = [p["paper_id"] for p in chunk]
        print(
            f"\n── Chunk {chunk_start // args.batch_size + 1}  "
            f"papers {chunk_ids[0]}–{chunk_ids[-1]} ({len(chunk)} papers) ──",
            flush=True,
        )

        # Submit batch jobs, skipping (assessment, run) pairs already completed.
        batch_meta: list[tuple[str, str, int, list[str]]] = []
        for assessment, perms in perms_map.items():
            for run_idx, perm in enumerate(perms, start=1):
                done = _already_done(con, assessment, chunk_ids, run_idx)
                remaining = [p for p in chunk if p["paper_id"] not in done]
                if not remaining:
                    print(f"  [{assessment} run {run_idx}] all done, skipping", flush=True)
                    continue

                desc     = f"{assessment} run {run_idx} | papers {chunk_ids[0]}–{chunk_ids[-1]}"
                batch_id = submit_batch(client, build_batch_jsonl(remaining, assessment, run_idx), desc)

                # Quick early-failure check — if the batch fails within 2s it's
                # almost certainly a bad parameter (model name, schema, API key).
                time.sleep(2)
                quick = client.batches.retrieve(batch_id)
                if quick.status == "failed":
                    raise RuntimeError(
                        f"Batch {batch_id} failed immediately — "
                        f"check model name, schema, and API key quota"
                    )

                batch_meta.append((batch_id, assessment, run_idx, perm))

        if not batch_meta:
            print("  All jobs already complete for this chunk.", flush=True)
            continue

        # Poll each job to completion, then parse and write results immediately.
        print(f"\n  Polling {len(batch_meta)} batch jobs...", flush=True)
        for batch_id, assessment, run_idx, perm in batch_meta:
            raw_rows        = download_results(client, poll_batch(client, batch_id))
            written = errors = 0

            for row in raw_rows:
                pid, asmnt, run, result, error, in_tok, out_tok, cached_tok, lp_json = \
                    parse_result_line(row)
                _INSERT_FN[asmnt](con, pid, str(pid), run, perm,
                                  result, error, in_tok, out_tok, cached_tok, lp_json)
                if error:
                    errors += 1
                    print(f"    [error] paper {pid} {asmnt} run {run}: {error}", flush=True)
                else:
                    written += 1

            print(f"    [{assessment} run {run_idx}] written={written}  errors={errors}",
                  flush=True)

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n── Final DB summary ──────────────────────────────────────────────", flush=True)
    for tbl in ("durc_results", "pepp_results", "governance_results"):
        n_papers     = con.execute(f"SELECT COUNT(DISTINCT paper_id) FROM {tbl}").fetchone()[0]
        n_rows       = con.execute(f"SELECT COUNT(*)                FROM {tbl}").fetchone()[0]
        total_in     = con.execute(f"SELECT SUM(in_tokens)          FROM {tbl}").fetchone()[0] or 0
        total_cached = con.execute(f"SELECT SUM(cached_tokens)      FROM {tbl}").fetchone()[0] or 0
        print(
            f"  {tbl:<25}: {n_papers} papers  {n_rows} rows  "
            f"in={total_in:,}  cached={total_cached:,}  "
            f"cache_rate={total_cached / max(total_in, 1) * 100:.1f}%",
            flush=True,
        )

    con.close()


if __name__ == "__main__":
    main()
