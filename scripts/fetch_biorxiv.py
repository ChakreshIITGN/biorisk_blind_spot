# Licensed under CC BY-NC-ND 4.0 — https://creativecommons.org/licenses/by-nc-nd/4.0/
"""
Fetch all biorXiv and medRxiv paper metadata via the official API.

API reference : https://api.biorxiv.org/
Endpoint      : https://api.biorxiv.org/details/{server}/{start}/{end}/{cursor}

Parameters
----------
server  : biorxiv | medrxiv
start   : YYYY-MM-DD  (earliest publication date to include)
end     : YYYY-MM-DD  (latest  publication date to include)
cursor  : page offset — 0, 100, 200 …  (API returns exactly 100 records per call)

How pagination guarantees zero duplicates
-----------------------------------------
The API returns records in a fixed, stable order: pub_date ASC, then doi ASC.
The cursor is a simple positional offset into that sorted list:

    cursor=0   → positions   0 –  99
    cursor=100 → positions 100 – 199
    cursor=200 → positions 200 – 299   … and so on.

Adjacent pages never overlap. A single pass from cursor=0, incrementing by 100
after each successful call, visits every record exactly once.

Duplicates only arise when results are appended across multiple runs. This script
always opens the output file in write mode ('w'), so every run produces a fresh,
complete file. For incremental updates (new papers only), pass a later --start
date and write to a separate output file — never append to an existing one.

Usage
-----
    # Full corpus, both servers, 2013 to today (default)
    uv run python fetch_biorxiv.py

    # biorXiv only
    uv run python fetch_biorxiv.py --server biorxiv

    # Incremental update — new papers since last fetch, separate output file
    uv run python fetch_biorxiv.py --start 2026-04-28 --out data/biorxiv_update.jsonl

    # Run in background, log to file
    nohup uv run python fetch_biorxiv.py >> logs/fetch_v2.log 2>&1 &
"""

import argparse
import json
import time
from datetime import date
from pathlib import Path

import requests

# ── Constants ─────────────────────────────────────────────────────────────────

BASE_URL    = "https://api.biorxiv.org/details/{server}/{start}/{end}/{cursor}"
PAGE_SIZE   = 100   # fixed API limit — 100 records per call, no way to change this
SLEEP_SEC   = 1.0   # 1 request/sec; conservative and within API guidelines
MAX_RETRIES = 5     # exponential back-off on transient network or HTTP errors

# ── API call ──────────────────────────────────────────────────────────────────

def fetch_page(session: requests.Session,
               server: str, start: str, end: str, cursor: int) -> dict:
    """
    Fetch one page (up to 100 records) from the API.

    Retries up to MAX_RETRIES times with exponential back-off on any
    network or HTTP error. Raises RuntimeError if all retries are exhausted.
    """
    url = BASE_URL.format(server=server, start=start, end=end, cursor=cursor)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            wait = 2 ** attempt
            print(f"  [retry {attempt}/{MAX_RETRIES}] {exc} — retrying in {wait}s",
                  flush=True)
            time.sleep(wait)
    raise RuntimeError(f"All {MAX_RETRIES} retries failed for: {url}")


def parse_record(raw: dict, server: str) -> dict:
    """
    Extract and normalise the fields we keep from a raw API record.

    'version' is stored so downstream code can select the latest revision of
    each preprint (max version per doi) when converting to Parquet.
    """
    authors = raw.get("authors", "")
    return {
        "doi":                              raw.get("doi", ""),
        "title":                            (raw.get("title",    "") or "").strip(),
        "abstract":                         (raw.get("abstract", "") or "").strip(),
        "authors":                          authors,
        "author_corresponding":             (raw.get("author_corresponding", "") or "").strip(),
        "author_corresponding_institution": (raw.get("author_corresponding_institution", "") or "").strip(),
        "author_count":                     len([a.strip() for a in authors.split(";") if a.strip()])
                                            if authors else 0,
        "pub_date":                         raw.get("date")    or None,
        "version":                          raw.get("version") or None,
        "category":                         raw.get("category", ""),
        "server":                           server,
    }

# ── Per-server fetch loop ─────────────────────────────────────────────────────

def fetch_server(session: requests.Session, fh,
                 server: str, start: str, end: str) -> int:
    """
    Fetch every record for one server within [start, end] and write to fh.

    Steps through cursor=0, 100, 200 … until the API returns count=0,
    which signals that all records have been returned. Each record is
    written as one JSON line. Returns the total number of records written.
    """
    print(f"\n{'='*60}", flush=True)
    print(f"Server : {server}", flush=True)
    print(f"Range  : {start} → {end}", flush=True)
    print(f"{'='*60}", flush=True)

    cursor = 0      # always start at the first record
    saved  = 0
    total  = None   # populated from the first API response

    while True:
        data  = fetch_page(session, server, start, end, cursor)
        msg   = (data.get("messages") or [{}])[0]
        count = int(msg.get("count", 0))

        # The API reports the full result-set size in every response.
        if total is None:
            total = int(msg.get("total", 0))
            print(f"Total  : {total:,} records", flush=True)

        # count=0 means we have stepped past the last record — fetch is complete.
        if count == 0:
            break

        for raw in data.get("collection", []):
            fh.write(json.dumps(parse_record(raw, server), default=str) + "\n")

        saved  += count
        cursor += PAGE_SIZE          # advance to the next non-overlapping page
        pct     = saved / total * 100 if total else 0
        print(f"  cursor={cursor:>7,}  saved={saved:>7,}/{total:,}  ({pct:.1f}%)",
              flush=True)

        # Safety stop: if cursor has passed the reported total we are done.
        if cursor >= total:
            break

        time.sleep(SLEEP_SEC)

    print(f"Done — {saved:,} records for {server}", flush=True)
    return saved

# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch biorXiv / medRxiv metadata to a JSONL file"
    )
    parser.add_argument(
        "--server",
        choices=["biorxiv", "medrxiv", "both"],
        default="both",
        help="Which preprint server to fetch (default: both)",
    )
    parser.add_argument(
        "--start",
        default="2013-01-01",
        help="Start date YYYY-MM-DD (default: 2013-01-01)",
    )
    parser.add_argument(
        "--end",
        default=str(date.today()),
        help="End date YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--out",
        default="data/biorxiv_all_fetchv2.jsonl",
        help="Output JSONL path (default: data/biorxiv_all_fetchv2.jsonl)",
    )
    args = parser.parse_args()

    servers = ["biorxiv", "medrxiv"] if args.server == "both" else [args.server]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": "biorxiv-fetcher/2.0 (research)"})

    grand_total = 0

    # Write mode ('w') — always produces a fresh, complete file.
    # Re-running replaces the file rather than appending duplicates.
    with open(out_path, "w") as fh:
        for server in servers:
            grand_total += fetch_server(session, fh, server, args.start, args.end)

    print(f"\nAll done. Grand total: {grand_total:,} records → {out_path}", flush=True)


if __name__ == "__main__":
    main()
