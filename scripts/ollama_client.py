"""
Stage 2 biosecurity assessment pipeline.

Reads papers already flagged by the manual Stage 1 screener (fuzzy / string
matching done externally) and runs three independent LLM assessments per paper:

  DURC       — 3 runs, D1–D9 categories in a different order each run
  PEPP       — 3 runs, P1–P3 criteria in a different order each run (Latin Square)
  Governance — 3 runs, G1–G5 categories in a different order each run

Permuting the presentation order across runs reduces anchoring / ordering bias.
All 9 calls per paper are fully independent and submitted together to a shared
ThreadPoolExecutor. A token-bucket rate limiter caps global API calls/sec.

Results are written to three DuckDB tables that join to the papers table on
paper_id. The id field (str(paper_id)) is passed to the LLM and must be
returned character-for-character unchanged — any mismatch is treated as an
error and the row is discarded.

Input — flagged papers come from one of two sources:
  1. DuckDB (default): papers table with a flagged BOOLEAN column added by
     the Stage 1 screener. Papers with all runs already stored are skipped
     automatically (resume-safe).
  2. JSONL (--input path): one JSON object per line, each with at minimum
     paper_id, title, abstract. Produced by the Stage 1 screener.

Usage:
    python client.py --db papers.duckdb
    python client.py --db papers.duckdb --limit 50 --workers 12 --rps 6
    python client.py --input flagged.jsonl --db results.duckdb
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import duckdb
import tiktoken
from dotenv import load_dotenv
from typing import Annotated

from pydantic import BaseModel, Field

load_dotenv()

# ── Tokeniser ─────────────────────────────────────────────────────────────────
# Used only as a fallback when the provider does not return token counts.
_TOKENIZER = tiktoken.get_encoding("cl100k_base")

def _count_tokens(text: str) -> int:
    return len(_TOKENIZER.encode(text))


# ── Pydantic output schemas ───────────────────────────────────────────────────
# Score is range-constrained 0–5 so hallucinated values are caught at
# validation time rather than silently written to the DB.

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
_RESULT_TYPES: dict[str, type[BaseModel]] = {
    "durc": DURCResult, "pepp": PEPPResult, "governance": GovernanceResult,
}
_SCORE_CLS: dict[str, type[BaseModel]] = {
    "durc": DURCScore, "pepp": PEPPScore, "governance": GovernanceScore,
}


# ── Prompt loading ─────────────────────────────────────────────────────────────
# Each prompt file has three fields: instruction (with placeholders), examples
# (array), and categories / criteria (dict). Loaded once at startup.

_PROMPTS_DIR = Path(__file__).parent / "prompts"

with open(_PROMPTS_DIR / "durc.json") as _f:
    _DURC = json.load(_f)

with open(_PROMPTS_DIR / "pepp.json") as _f:
    _PEPP = json.load(_f)

with open(_PROMPTS_DIR / "governance.json") as _f:
    _GOV = json.load(_f)

# ── Label ↔ ID mappings ───────────────────────────────────────────────────────
_ID_TO_LABEL: dict[str, dict[str, str]] = {
    "durc":       {k: v["name"].lower() for k, v in _DURC["categories"].items()},
    "pepp":       {k: v["name"].lower() for k, v in _PEPP["criteria"].items()},
    "governance": {k: v["name"].lower() for k, v in _GOV["categories"].items()},
}
_LABEL_TO_ID: dict[str, dict[str, str]] = {
    a: {v: k for k, v in m.items()} for a, m in _ID_TO_LABEL.items()
}
_LABEL_ORDER: dict[str, list[str]] = {
    a: list(m.values()) for a, m in _ID_TO_LABEL.items()
}
_CATS_BY_LABEL: dict[str, dict] = {
    "durc":       {v["name"].lower(): v for v in _DURC["categories"].values()},
    "pepp":       {v["name"].lower(): v for v in _PEPP["criteria"].values()},
    "governance": {v["name"].lower(): v for v in _GOV["categories"].values()},
}


def _translate_to_ids(assessment: str, raw: dict) -> dict:
    """Rename label keys ("enhanced virulence") to ID keys ("D1") in-place."""
    mapping = _LABEL_TO_ID[assessment]
    return {mapping.get(k, k): v for k, v in raw.items()}


# ── Sparse schema helper ──────────────────────────────────────────────────────
# For Ollama structured output: makes all category fields optional so the LLM
# can emit sparse JSON (only categories with evidence). The full Pydantic models
# (all fields required) are used for DB validation after zero-fill.

def _sparse_schema(result_type: type[BaseModel], assessment: str) -> dict:
    schema = copy.deepcopy(result_type.model_json_schema())
    schema["required"] = ["id"]
    # Rename property keys from IDs (D1 …) to natural-language labels.
    id_to_label = _ID_TO_LABEL[assessment]
    schema["properties"] = {id_to_label.get(k, k): v for k, v in schema["properties"].items()}
    return schema


# ── Zero-fill helpers ─────────────────────────────────────────────────────────

_ZERO_SCORE = {"score": 0, "evidence": ""}
_ID_CATS: dict[str, list[str]] = {
    "durc":       list(_ID_TO_LABEL["durc"].keys()),    # ["D1","D2",...]
    "pepp":       list(_ID_TO_LABEL["pepp"].keys()),    # ["P1","P2","P3"]
    "governance": list(_ID_TO_LABEL["governance"].keys()),  # ["G1",...,"G5"]
}


def _make_zero_result(assessment: str, result_id: str) -> BaseModel:
    """All-zero result — used when LLM returns the NONE FOUND sentinel."""
    return _RESULT_TYPES[assessment](
        id=result_id,
        **{k: _SCORE_CLS[assessment](**_ZERO_SCORE) for k in _ID_CATS[assessment]},
    )


def _normalize_category(cat_dict: dict) -> dict:
    """Rename any non-'score' string field to 'evidence'.
    Handles LLM field name drift: observation, rationale, reasoning, etc."""
    if "evidence" in cat_dict:
        return cat_dict
    for key, val in cat_dict.items():
        if key != "score" and isinstance(val, str):
            cat_dict["evidence"] = val
            del cat_dict[key]
            return cat_dict
    cat_dict.setdefault("evidence", "")
    return cat_dict


def _fill_zeros(assessment: str, raw: dict) -> dict:
    # Called after _translate_to_ids — raw keys are IDs (D1, P1 …) at this point.
    # Treat both absent keys and explicit null the same way.
    ids = _ID_CATS[assessment]
    for k in ids:
        if raw.get(k) is None:
            raw[k] = _ZERO_SCORE
    for k in ids:
        if isinstance(raw.get(k), dict):
            raw[k] = _normalize_category(raw[k])
    return raw


# ── Permutation sets ──────────────────────────────────────────────────────────
# Pre-computed with a fixed seed so every pipeline run uses identical orderings.
# This makes multi-run aggregation reproducible.

_RNG = random.Random(42)


def _generate_permutations(keys: list[str], n: int) -> list[list[str]]:
    """
    Select n permutations of keys. Tries to ensure no key appears in the same
    position more than twice across the selected set to spread positional exposure.
    Falls back to any permutation if the constraint cannot be fully satisfied.
    """
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

    # Safety: top up with any remaining permutation if constraint was too tight
    idx = 0
    while len(selected) < n:
        selected.append(list(pool[idx % len(pool)]))
        idx += 1

    return selected


# DURC: 9 categories, 3 runs
_DURC_PERMS: list[list[str]] = _generate_permutations(_LABEL_ORDER["durc"], n=3)

# PEPP: 3 criteria, 3 runs — exact Latin Square
_PEPP_PERMS: list[list[str]] = [
    _LABEL_ORDER["pepp"],
    _LABEL_ORDER["pepp"][1:] + _LABEL_ORDER["pepp"][:1],
    _LABEL_ORDER["pepp"][2:] + _LABEL_ORDER["pepp"][:2],
]

# Governance: 5 categories, 3 runs
_GOV_PERMS: list[list[str]] = _generate_permutations(_LABEL_ORDER["governance"], n=3)


# ── Prompt assembly ───────────────────────────────────────────────────────────
# Assembles the full system prompt for one run given a specific category order.
# Uses str.replace (not str.format) so that JSON curly braces in the instruction
# string are not misinterpreted as Python format placeholders.

def _fmt_examples(examples: list) -> str:
    parts = [
        f"[{ex['label']}]\nInput: {json.dumps(ex['input'])}\n"
        f"Output: {json.dumps(ex['output']) if isinstance(ex['output'], dict) else ex['output']}"
        for ex in examples
    ]
    return "\n\n".join(parts)


def _fmt_section(items_by_label: dict, labels: list[str]) -> str:
    # [label] is the exact JSON key the LLM must use in its output.
    return "\n\n".join(
        f"[{lbl}] {items_by_label[lbl]['name']}\n{items_by_label[lbl]['description']}\n"
        f"Keywords: {items_by_label[lbl]['keywords']}"
        for lbl in labels
    )


def _build_durc_prompt(perm: list[str]) -> str:
    return (
        _DURC["instruction"]
        .replace("{ordered_categories}", _fmt_section(_CATS_BY_LABEL["durc"], perm))
        .replace("{examples}",           _fmt_examples(_DURC["examples"]))
    )


def _build_pepp_prompt(perm: list[str]) -> str:
    return (
        _PEPP["instruction"]
        .replace("{ordered_criteria}", _fmt_section(_CATS_BY_LABEL["pepp"], perm))
        .replace("{examples}",         _fmt_examples(_PEPP["examples"]))
    )


def _build_gov_prompt(perm: list[str]) -> str:
    return (
        _GOV["instruction"]
        .replace("{ordered_categories}", _fmt_section(_CATS_BY_LABEL["governance"], perm))
        .replace("{examples}",           _fmt_examples(_GOV["examples"]))
    )


# Pre-built prompts — all 9 possible prompts computed once at startup.
_DURC_PROMPTS: list[str] = [_build_durc_prompt(p) for p in _DURC_PERMS]
_PEPP_PROMPTS: list[str] = [_build_pepp_prompt(p) for p in _PEPP_PERMS]
_GOV_PROMPTS:  list[str] = [_build_gov_prompt(p)  for p in _GOV_PERMS]

# Keyed by (assessment, run) — same structure as openai_client.py for
# cross-pipeline consistency.
_SYSTEM_PROMPTS: dict[tuple[str, int], str] = {
    ("durc",       1): _DURC_PROMPTS[0],
    ("durc",       2): _DURC_PROMPTS[1],
    ("durc",       3): _DURC_PROMPTS[2],
    ("pepp",       1): _PEPP_PROMPTS[0],
    ("pepp",       2): _PEPP_PROMPTS[1],
    ("pepp",       3): _PEPP_PROMPTS[2],
    ("governance", 1): _GOV_PROMPTS[0],
    ("governance", 2): _GOV_PROMPTS[1],
    ("governance", 3): _GOV_PROMPTS[2],
}


# ── Provider config ───────────────────────────────────────────────────────────

PROVIDER     = os.getenv("LLM_PROVIDER", "ollama")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:120b")
OLLAMA_HOST  = os.getenv("OLLAMA_HOST", "https://ollama.com")


# ── Ollama multi-key rotation ─────────────────────────────────────────────────
# Reads OLLAMA_CLOUD_KEY_1, OLLAMA_CLOUD_KEY_2, OLLAMA_CLOUD_KEY_3 … from .env.
# Workers round-robin across keys — each key has its own rate limit so N keys
# gives N× throughput ceiling.

def _load_ollama_keys() -> list[str]:
    keys, i = [], 1
    while k := os.getenv(f"OLLAMA_CLOUD_KEY_{i}"):
        keys.append(k)
        i += 1
    return keys

_OLLAMA_KEYS = _load_ollama_keys()
_KEY_CYCLE   = itertools.cycle(_OLLAMA_KEYS) if _OLLAMA_KEYS else itertools.cycle([""])
_KEY_LOCK    = threading.Lock()

def _next_ollama_key() -> str:
    with _KEY_LOCK:
        return next(_KEY_CYCLE)


# ── Rate limiter ──────────────────────────────────────────────────────────────
# Token bucket shared across all worker threads. Workers block here rather than
# firing bursts that would trigger 429s. Jitter in retry prevents thundering herd.

class RateLimiter:
    def __init__(self, rps: float):
        self._interval = 1.0 / rps
        self._lock     = threading.Lock()
        self._next_ok  = time.monotonic()

    def acquire(self):
        with self._lock:
            now  = time.monotonic()
            wait = self._next_ok - now
            if wait > 0:
                time.sleep(wait)
            self._next_ok = max(time.monotonic(), self._next_ok) + self._interval


_RATE_LIMITER: RateLimiter | None = None   # initialised in main() before workers start


# ── LLM response container ────────────────────────────────────────────────────

@dataclass
class LLMResponse:
    content: str
    in_tokens: int
    out_tokens: int


# ── Provider implementations ──────────────────────────────────────────────────
# All providers share the same signature:
#   (system_prompt, user_content, result_type) -> LLMResponse
# result_type is the Pydantic model the output will be validated against.

def call_ollama(system_prompt: str, user_content: str,
                result_type: type[BaseModel], assessment: str) -> LLMResponse:
    from ollama import Client

    if not _OLLAMA_KEYS:
        raise EnvironmentError("No OLLAMA_CLOUD_KEY_* found in environment")

    client = Client(
        host=OLLAMA_HOST,
        headers={"Authorization": f"Bearer {_next_ollama_key()}"},
    )
    response = client.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content},
        ],
        format=_sparse_schema(result_type, assessment),   # sparse schema: only scored categories required
    )

    in_tok  = getattr(response, "prompt_eval_count", 0) or 0
    out_tok = getattr(response, "eval_count", 0) or 0
    if in_tok  == 0: in_tok  = _count_tokens(system_prompt + user_content)
    if out_tok == 0: out_tok = _count_tokens(response.message.content)

    return LLMResponse(response.message.content, in_tok, out_tok)


def call_claude(system_prompt: str, user_content: str,
                result_type: type[BaseModel], assessment: str) -> LLMResponse:
    import anthropic
    client = anthropic.Anthropic()

    # cache_control pins the static system prompt in Anthropic's KV cache for
    # ~5 min — avoids re-charging the bulk of input tokens on every call.
    system = [{"type": "text", "text": system_prompt,
               "cache_control": {"type": "ephemeral"}}]
    resp = client.messages.create(
        model=os.getenv("CLAUDE_MODEL", "claude-haiku-4-5-20251001"),
        max_tokens=512,
        system=system,
        messages=[{"role": "user", "content": user_content}],
    )
    return LLMResponse(
        resp.content[0].text,
        resp.usage.input_tokens,
        resp.usage.output_tokens,
    )


def call_openai(system_prompt: str, user_content: str,
                result_type: type[BaseModel], assessment: str) -> LLMResponse:
    from openai import OpenAI
    client = OpenAI()

    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        max_tokens=512,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content},
        ],
    )
    return LLMResponse(
        resp.choices[0].message.content,
        resp.usage.prompt_tokens,
        resp.usage.completion_tokens,
    )


def call_gemini(system_prompt: str, user_content: str,
                result_type: type[BaseModel], assessment: str) -> LLMResponse:
    import google.generativeai as genai
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

    model = genai.GenerativeModel(
        model_name=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
        system_instruction=system_prompt,
    )
    resp = model.generate_content(
        user_content,
        generation_config={"max_output_tokens": 512, "response_mime_type": "application/json"},
    )
    in_tok  = (resp.usage_metadata.prompt_token_count
               if resp.usage_metadata else _count_tokens(system_prompt + user_content))
    out_tok = (resp.usage_metadata.candidates_token_count
               if resp.usage_metadata else _count_tokens(resp.text))
    return LLMResponse(resp.text, in_tok, out_tok)


_PROVIDERS = {
    "ollama": call_ollama,
    "claude": call_claude,
    "openai": call_openai,
    "gemini": call_gemini,
}


def call_llm(system_prompt: str, user_content: str,
             result_type: type[BaseModel], assessment: str) -> LLMResponse:
    fn = _PROVIDERS.get(PROVIDER)
    if not fn:
        raise ValueError(f"Unknown provider '{PROVIDER}'. Choose from: {list(_PROVIDERS)}")
    return fn(system_prompt, user_content, result_type, assessment)


def _call_with_retry(fn, max_retries: int = 4) -> LLMResponse:
    """Acquire rate limiter slot, call fn(), retry on 429 with exponential backoff."""
    for attempt in range(max_retries + 1):
        if attempt > 0:
            # Exponential backoff + full jitter before re-acquiring a slot.
            # Full jitter (random * ceiling) spreads retrying workers apart
            # better than additive jitter on a fixed base.
            ceiling = 2 ** attempt
            time.sleep(random.uniform(0, ceiling))
        if _RATE_LIMITER:
            _RATE_LIMITER.acquire()
        try:
            return fn()
        except Exception as e:
            err = str(e).lower()
            is_rate_limit = "429" in err or "rate limit" in err or "too many" in err
            if attempt < max_retries and is_rate_limit:
                print(f"  [rate limit] retry {attempt+1}/{max_retries}", flush=True)
            else:
                raise


# ── Database ──────────────────────────────────────────────────────────────────

def init_db(db_path: str) -> duckdb.DuckDBPyConnection:
    """
    Open the DuckDB file and ensure the three result tables exist.
    The papers table (source of flagged papers) is assumed to already exist
    and to have a flagged BOOLEAN column added by the Stage 1 screener.
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(db_path)

    con.execute("""
        CREATE TABLE IF NOT EXISTS durc_results (
            paper_id           INTEGER,
            id                 TEXT,
            doi                TEXT,
            run                INTEGER,            -- 1–3
            presentation_order TEXT,               -- e.g. "D3,D7,D1,D5,D9,D2,D8,D4,D6"
            D1_score INTEGER,  D1_evidence TEXT,
            D2_score INTEGER,  D2_evidence TEXT,
            D3_score INTEGER,  D3_evidence TEXT,
            D4_score INTEGER,  D4_evidence TEXT,
            D5_score INTEGER,  D5_evidence TEXT,
            D6_score INTEGER,  D6_evidence TEXT,
            D7_score INTEGER,  D7_evidence TEXT,
            D8_score INTEGER,  D8_evidence TEXT,
            D9_score INTEGER,  D9_evidence TEXT,
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
        CREATE TABLE IF NOT EXISTS pepp_results (
            paper_id           INTEGER,
            id                 TEXT,
            doi                TEXT,
            run                INTEGER,
            presentation_order TEXT,
            P1_score INTEGER,  P1_evidence TEXT,
            P2_score INTEGER,  P2_evidence TEXT,
            P3_score INTEGER,  P3_evidence TEXT,
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
            doi                TEXT,
            run                INTEGER,
            presentation_order TEXT,
            G1_score INTEGER,  G1_evidence TEXT,
            G2_score INTEGER,  G2_evidence TEXT,
            G3_score INTEGER,  G3_evidence TEXT,
            G4_score INTEGER,  G4_evidence TEXT,
            G5_score INTEGER,  G5_evidence TEXT,
            error          TEXT,
            in_tokens      INTEGER,
            out_tokens     INTEGER,
            cached_tokens  INTEGER,
            score_logprobs TEXT,
            assessed_at    TIMESTAMP DEFAULT current_timestamp,
            PRIMARY KEY (paper_id, run)
        )
    """)

    # Migrate existing tables created before these columns were added.
    for tbl in ("durc_results", "pepp_results", "governance_results"):
        con.execute(f"ALTER TABLE {tbl} ADD COLUMN IF NOT EXISTS doi            TEXT")
        con.execute(f"ALTER TABLE {tbl} ADD COLUMN IF NOT EXISTS in_tokens      INTEGER")
        con.execute(f"ALTER TABLE {tbl} ADD COLUMN IF NOT EXISTS out_tokens     INTEGER")
        con.execute(f"ALTER TABLE {tbl} ADD COLUMN IF NOT EXISTS cached_tokens  INTEGER")
        con.execute(f"ALTER TABLE {tbl} ADD COLUMN IF NOT EXISTS score_logprobs TEXT")

    return con


def get_pending_papers(con: duckdb.DuckDBPyConnection,
                       limit: int | None = None) -> list[dict]:
    """
    Return flagged papers that have not yet completed all assessment runs.
    'Complete' means 3 DURC rows + 3 PEPP rows + 3 governance rows exist.
    Safe to re-run — already-complete papers are skipped automatically.

    Requires the papers table to have a flagged BOOLEAN column.
    """
    limit_clause = f"LIMIT {limit}" if limit else ""
    rows = con.execute(f"""
        SELECT p.paper_id,
               CAST(p.paper_id AS VARCHAR) AS id,
               COALESCE(p.doi, '') AS doi,
               p.title,
               p.abstract
        FROM   papers p
        WHERE  p.flagged = true
          AND (
                (SELECT COUNT(*) FROM durc_results       d WHERE d.paper_id = p.paper_id) < 3
             OR (SELECT COUNT(*) FROM pepp_results        e WHERE e.paper_id = p.paper_id) < 3
             OR (SELECT COUNT(*) FROM governance_results  g WHERE g.paper_id = p.paper_id) < 3
          )
        ORDER BY p.paper_id ASC
        {limit_clause}
    """).fetchall()

    return [{"paper_id": r[0], "id": r[1], "doi": r[2], "title": r[3], "abstract": r[4]}
            for r in rows]


def load_flagged_from_screen(screen_path: str, source_path: str) -> list[dict]:
    con = duckdb.connect()
    con.execute(f"ATTACH '{screen_path}' AS screen (READ_ONLY)")
    rows = con.execute(f"""
        SELECT p.paper_id,
               CAST(p.paper_id AS VARCHAR) AS id,
               COALESCE(p.doi, '') AS doi,
               p.title,
               p.abstract
        FROM   read_parquet('{source_path}') AS p
        JOIN   screen.screen_results AS s USING (paper_id)
        WHERE  s.flag = true
        ORDER BY p.paper_id
    """).fetchall()
    con.close()
    return [{"paper_id": r[0], "id": r[1], "doi": r[2], "title": r[3], "abstract": r[4]}
            for r in rows]


def load_flagged_jsonl(path: str) -> list[dict]:
    """
    Load flagged papers from a JSONL file produced by the Stage 1 screener.
    Each line must have at minimum: paper_id, title, abstract.
    """
    papers = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            r.setdefault("id", str(r["paper_id"]))
            r.setdefault("doi", "")
            papers.append(r)
    return papers


def load_flagged_parquet(path: str, limit: int | None = None) -> list[dict]:
    limit_clause = f"LIMIT {limit}" if limit else ""
    rows = duckdb.sql(f"""
        SELECT paper_id,
               CAST(paper_id AS VARCHAR) AS id,
               COALESCE(doi, '') AS doi,
               title,
               abstract
        FROM   read_parquet('{path}')
        WHERE  flag = true
        ORDER BY paper_id
        {limit_clause}
    """).fetchall()
    return [{"paper_id": r[0], "id": r[1], "doi": r[2], "title": r[3], "abstract": r[4]}
            for r in rows]


# ── Job definition ────────────────────────────────────────────────────────────

@dataclass
class Job:
    paper_id:   int
    id:         str          # string id sent to LLM — must be returned verbatim
    doi:        str
    title:      str
    abstract:   str
    assessment: str          # "durc" | "pepp" | "governance"
    run:        int          # 1-based run index
    perm:       list[str]    # category/criterion order for this run


def build_jobs(papers: list[dict]) -> list[Job]:
    jobs = []
    for p in papers:
        doi = p.get("doi", "")
        for run, perm in enumerate(_DURC_PERMS, start=1):
            jobs.append(Job(p["paper_id"], p["id"], doi, p["title"], p["abstract"],
                            "durc", run, perm))
        for run, perm in enumerate(_PEPP_PERMS, start=1):
            jobs.append(Job(p["paper_id"], p["id"], doi, p["title"], p["abstract"],
                            "pepp", run, perm))
        for run, perm in enumerate(_GOV_PERMS, start=1):
            jobs.append(Job(p["paper_id"], p["id"], doi, p["title"], p["abstract"],
                            "governance", run, perm))
    return jobs


# ── DB write helpers ──────────────────────────────────────────────────────────
# INSERT OR REPLACE so that re-runs overwrite previous results cleanly.

def _insert_durc(con: duckdb.DuckDBPyConnection, job: Job,
                 result: DURCResult | None, error: str | None,
                 in_tokens: int, out_tokens: int,
                 cached_tokens: int = 0, lp_json: str = "null") -> None:
    order = ",".join(_LABEL_TO_ID["durc"][l] for l in job.perm)
    if result:
        vals = [
            job.paper_id, job.id, job.doi, job.run, order,
            result.D1.score, result.D1.evidence,
            result.D2.score, result.D2.evidence,
            result.D3.score, result.D3.evidence,
            result.D4.score, result.D4.evidence,
            result.D5.score, result.D5.evidence,
            result.D6.score, result.D6.evidence,
            result.D7.score, result.D7.evidence,
            result.D8.score, result.D8.evidence,
            result.D9.score, result.D9.evidence,
            error, in_tokens, out_tokens, cached_tokens, lp_json,
        ]
    else:
        vals = [job.paper_id, job.id, job.doi, job.run, order,
                *([None] * 18), error, in_tokens, out_tokens, cached_tokens, lp_json]

    con.execute("""
        INSERT OR REPLACE INTO durc_results
        (paper_id, id, doi, run, presentation_order,
         D1_score, D1_evidence, D2_score, D2_evidence,
         D3_score, D3_evidence, D4_score, D4_evidence,
         D5_score, D5_evidence, D6_score, D6_evidence,
         D7_score, D7_evidence, D8_score, D8_evidence,
         D9_score, D9_evidence,
         error, in_tokens, out_tokens, cached_tokens, score_logprobs)
        VALUES (?,?,?,?,?, ?,?,?,?, ?,?,?,?, ?,?,?,?, ?,?,?,?, ?,?, ?,?,?,?,?)
    """, vals)


def _insert_pepp(con: duckdb.DuckDBPyConnection, job: Job,
                 result: PEPPResult | None, error: str | None,
                 in_tokens: int, out_tokens: int,
                 cached_tokens: int = 0, lp_json: str = "null") -> None:
    order = ",".join(_LABEL_TO_ID["pepp"][l] for l in job.perm)
    if result:
        vals = [
            job.paper_id, job.id, job.doi, job.run, order,
            result.P1.score, result.P1.evidence,
            result.P2.score, result.P2.evidence,
            result.P3.score, result.P3.evidence,
            error, in_tokens, out_tokens, cached_tokens, lp_json,
        ]
    else:
        vals = [job.paper_id, job.id, job.doi, job.run, order,
                *([None] * 6), error, in_tokens, out_tokens, cached_tokens, lp_json]

    con.execute("""
        INSERT OR REPLACE INTO pepp_results
        (paper_id, id, doi, run, presentation_order,
         P1_score, P1_evidence, P2_score, P2_evidence, P3_score, P3_evidence,
         error, in_tokens, out_tokens, cached_tokens, score_logprobs)
        VALUES (?,?,?,?,?, ?,?,?,?, ?,?, ?,?,?,?,?)
    """, vals)


def _insert_governance(con: duckdb.DuckDBPyConnection, job: Job,
                       result: GovernanceResult | None, error: str | None,
                       in_tokens: int, out_tokens: int,
                       cached_tokens: int = 0, lp_json: str = "null") -> None:
    order = ",".join(_LABEL_TO_ID["governance"][l] for l in job.perm)
    if result:
        vals = [
            job.paper_id, job.id, job.doi, job.run, order,
            result.G1.score, result.G1.evidence,
            result.G2.score, result.G2.evidence,
            result.G3.score, result.G3.evidence,
            result.G4.score, result.G4.evidence,
            result.G5.score, result.G5.evidence,
            error, in_tokens, out_tokens, cached_tokens, lp_json,
        ]
    else:
        vals = [job.paper_id, job.id, job.doi, job.run, order,
                *([None] * 10), error, in_tokens, out_tokens, cached_tokens, lp_json]

    con.execute("""
        INSERT OR REPLACE INTO governance_results
        (paper_id, id, doi, run, presentation_order,
         G1_score, G1_evidence, G2_score, G2_evidence,
         G3_score, G3_evidence, G4_score, G4_evidence, G5_score, G5_evidence,
         error, in_tokens, out_tokens, cached_tokens, score_logprobs)
        VALUES (?,?,?,?,?, ?,?,?,?, ?,?,?,?, ?,?, ?,?,?,?,?)
    """, vals)


# ── Batch export ─────────────────────────────────────────────────────────────

def _export_batch(con: duckdb.DuckDBPyConnection,
                  paper_ids: list[int], base_db_path: str) -> None:
    """Export one batch to a new DuckDB file as a single long-format 'results' table.
    Schema: (paper_id, id, run, assessment, category, score, evidence, error, assessed_at)
    Called with db_lock held so no concurrent writes occur during ATTACH."""
    min_pid  = min(paper_ids)
    max_pid  = max(paper_ids)
    out_dir  = Path(base_db_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(out_dir / f"{min_pid}_{max_pid}.duckdb")
    id_csv   = ",".join(map(str, paper_ids))   # ints only — safe in SQL

    def _sel(table: str, assessment: str, cat: str) -> str:
        return (
            f"SELECT paper_id, id, run, '{assessment}' AS assessment, "
            f"'{cat}' AS category, {cat}_score AS score, {cat}_evidence AS evidence, "
            f"error, in_tokens, out_tokens, assessed_at FROM {table} WHERE paper_id IN ({id_csv})"
        )

    parts = (
        [_sel("durc_results",       "durc",       f"D{i}") for i in range(1, 10)] +
        [_sel("pepp_results",       "pepp",       f"P{i}") for i in range(1, 4)]  +
        [_sel("governance_results", "governance", f"G{i}") for i in range(1, 6)]
    )
    union_sql = "\nUNION ALL\n".join(parts)

    con.execute(f"ATTACH '{out_path}' AS _batch")
    con.execute(
        f"CREATE OR REPLACE TABLE _batch.results AS\n{union_sql}\n"
        f"ORDER BY paper_id, assessment, run, category"
    )
    con.execute("DETACH _batch")

    print(f"  [batch] {len(paper_ids)} papers → {out_path}", flush=True)


# ── Worker ────────────────────────────────────────────────────────────────────

def _run_job(job: Job) -> tuple[Job, BaseModel | None, str | None, int, int]:
    """
    Execute one LLM assessment call.
    Handles sparse JSON output (missing categories filled with score=0) and
    the NONE FOUND sentinel (all categories set to score=0).
    Returns (job, parsed_result_or_None, error_or_None, in_tokens, out_tokens).
    """
    paper_input = json.dumps({
        "id":       job.id,
        "title":    job.title,
        "abstract": job.abstract,
    })

    system      = _SYSTEM_PROMPTS[(job.assessment, job.run)]
    result_type = _RESULT_TYPES[job.assessment]

    in_tok = out_tok = 0
    try:
        resp    = _call_with_retry(lambda: call_llm(system, paper_input, result_type, job.assessment))
        in_tok  = resp.in_tokens
        out_tok = resp.out_tokens
        content = resp.content.strip()

        if content == "NONE FOUND":
            result = _make_zero_result(job.assessment, job.id)
        else:
            raw    = json.loads(content)
            raw    = _translate_to_ids(job.assessment, raw)  # labels → IDs
            raw    = _fill_zeros(job.assessment, raw)
            result = result_type.model_validate(raw)

        # The id must come back verbatim — any coercion or re-generation is an error.
        if result.id != job.id:
            raise ValueError(f"id mismatch: sent {job.id!r}, got {result.id!r}")

        return job, result, None, in_tok, out_tok

    except Exception as e:
        return job, None, str(e), in_tok, out_tok


def _worker(job: Job, con: duckdb.DuckDBPyConnection,
            db_lock: threading.Lock) -> tuple[str, bool, str | None]:
    """
    Run one job and write the result to the appropriate table.
    db_lock serialises all writes — DuckDB allows only one writer at a time.
    Returns (assessment_type, success, error_str).
    """
    job_out, result, error, in_tok, out_tok = _run_job(job)

    with db_lock:
        if job.assessment == "durc":
            _insert_durc(con, job_out, result, error, in_tok, out_tok)
        elif job.assessment == "pepp":
            _insert_pepp(con, job_out, result, error, in_tok, out_tok)
        else:
            _insert_governance(con, job_out, result, error, in_tok, out_tok)

    if error:
        print(f"  [error] paper {job.paper_id} {job.assessment} run {job.run}: {error}",
              flush=True)

    return job.assessment, error is None, error


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Stage 2 biosecurity assessment: DURC, PEPP, and governance scoring."
    )
    parser.add_argument("--db",
                        default="papers.duckdb",
                        help="DuckDB file (source of flagged papers + results destination)")
    parser.add_argument("--input",
                        default=None,
                        help="JSONL file of flagged papers from Stage 1 screener "
                             "(overrides --db as paper source)")
    parser.add_argument("--parquet",
                        default=None,
                        help="Parquet file with paper_id, doi, title, abstract, flag columns "
                             "(reads flagged=true rows directly)")
    parser.add_argument("--screen",
                        default=None,
                        help="Stage 1 screen results duckdb (e.g. data/biorXiv24_25_screen.duckdb)")
    parser.add_argument("--source",
                        default=None,
                        help="Source parquet to join title/abstract from (used with --screen)")
    parser.add_argument("--limit",
                        type=int, default=None,
                        help="Cap number of papers — useful for test runs")
    parser.add_argument("--workers",
                        type=int, default=12,
                        help="Parallel worker threads (default 12)")
    parser.add_argument("--rps",
                        type=float, default=6.0,
                        help="Max API calls/sec across all workers "
                             "(default 6 = 2 rps/key × 3 keys)")
    args = parser.parse_args()

    global _RATE_LIMITER
    _RATE_LIMITER = RateLimiter(args.rps)

    con = init_db(args.db)

    if args.parquet:
        papers = load_flagged_parquet(args.parquet, limit=args.limit)

        # ── Resume filter ─────────────────────────────────────────────────────
        # When restarting after a crash, skip papers that already have all 3
        # runs complete in durc_results, pepp_results, AND governance_results.
        # A paper is complete only if it appears in all three intersections.
        # This makes --parquet restarts safe without re-spending tokens.
        complete_ids = set(
            row[0] for row in con.execute("""
                SELECT paper_id FROM durc_results      GROUP BY paper_id HAVING COUNT(*) >= 3
                INTERSECT
                SELECT paper_id FROM pepp_results       GROUP BY paper_id HAVING COUNT(*) >= 3
                INTERSECT
                SELECT paper_id FROM governance_results GROUP BY paper_id HAVING COUNT(*) >= 3
            """).fetchall()
        )
        before = len(papers)
        papers = [p for p in papers if p["paper_id"] not in complete_ids]
        print(f"Resume check: {before} flagged → {len(papers)} pending "
              f"({len(complete_ids)} already complete, skipped)")
        # ─────────────────────────────────────────────────────────────────────

    elif args.screen and args.source:
        papers = load_flagged_from_screen(args.screen, args.source)
        if args.limit:
            papers = papers[:args.limit]
    elif args.input:
        papers = load_flagged_jsonl(args.input)
        if args.limit:
            papers = papers[:args.limit]
    else:
        papers = get_pending_papers(con, limit=args.limit)

    total_papers = len(papers)
    if total_papers == 0:
        print("No pending papers found. All done or no flagged papers in source.")
        con.close()
        return

    jobs       = build_jobs(papers)
    total_jobs = len(jobs)   # total_papers × 9

    print(
        f"Papers to assess : {total_papers}\n"
        f"Total jobs       : {total_jobs}  (3 DURC + 3 PEPP + 3 governance per paper)\n"
        f"Provider         : {PROVIDER}\n"
        f"Workers          : {args.workers}\n"
        f"RPS limit        : {args.rps}\n"
        f"Ollama keys      : {len(_OLLAMA_KEYS) if PROVIDER == 'ollama' else 'n/a'}\n",
        flush=True,
    )

    paper_jobs_remaining = {p["paper_id"]: 9 for p in papers}
    current_batch: list[int] = []

    db_lock   = threading.Lock()
    done = errors = 0
    run_start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_worker, j, con, db_lock): j for j in jobs}

        for future in as_completed(futures):
            job = futures[future]
            _, success, _ = future.result()
            done   += 1
            errors += 0 if success else 1

            # Track per-paper completion; export once 100 papers are fully done.
            paper_jobs_remaining[job.paper_id] -= 1
            if paper_jobs_remaining[job.paper_id] == 0:
                current_batch.append(job.paper_id)
                if len(current_batch) == 100:
                    with db_lock:
                        _export_batch(con, sorted(current_batch), args.db)
                    current_batch.clear()

            if done % 100 == 0 or done == total_jobs:
                elapsed = time.time() - run_start
                print(
                    f"  done={done}/{total_jobs}  errors={errors}"
                    f"  elapsed={elapsed:.0f}s  avg={elapsed/done:.1f}s/job",
                    flush=True,
                )

    # Export any tail batch (< 100 papers) after all workers finish.
    if current_batch:
        _export_batch(con, sorted(current_batch), args.db)

    elapsed = time.time() - run_start
    con.close()

    print(f"""
── Run summary ──────────────────────────────────
  Provider  : {PROVIDER}
  Workers   : {args.workers}
  RPS limit : {args.rps}
  Papers    : {total_papers}
  Jobs      : {total_jobs}
  Errors    : {errors}
  Time      : {elapsed:.1f}s
  Avg/job   : {elapsed / total_jobs:.2f}s
────────────────────────────────────────────────""")


if __name__ == "__main__":
    main()