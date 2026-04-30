"""
fetch_arxiv.py — fetch 500 random papers from arXiv non-biology categories.

Used to build a true negative-control dataset (baseline_data2.parquet) for
LLM calibration: papers with no conceivable DURC/PEPP/Governance relevance.

Categories chosen: CS, mathematics, physics (non-bio), economics, statistics.
Papers are fetched in batches of 100 from random start offsets per category,
parsed from the arXiv Atom XML API, and saved as a parquet file.

Output: data/baseline_data2.parquet
Columns: paper_id (sequential int), doi, title, abstract

arXiv API docs: https://info.arxiv.org/help/api/index.html
Rate limit: 1 request per second (polite use).

Usage:
    uv run python fetch_arxiv.py
"""

import random
import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import duckdb

OUTPUT    = "data/baseline_data2.parquet"
N_TARGET  = 500
BATCH     = 100      # arXiv API max per request
SEED      = 42
DELAY     = 1.5      # seconds between requests (arXiv rate limit)

# Non-biology arXiv categories — clearly unrelated to DURC/PEPP/Governance
CATEGORIES = [
    "cs.AI",        # Artificial Intelligence
    "cs.LG",        # Machine Learning
    "cs.CV",        # Computer Vision
    "math.CO",      # Combinatorics
    "math.ST",      # Statistics Theory
    "physics.hep-ph",   # High Energy Physics
    "physics.cond-mat", # Condensed Matter
    "econ.GN",      # General Economics
    "stat.ML",      # Machine Learning (Statistics)
    "astro-ph.GA",  # Astrophysics: Galaxies
]

ARXIV_API = "http://export.arxiv.org/api/query"
NS        = {"atom": "http://www.w3.org/2005/Atom"}

random.seed(SEED)


def fetch_batch(category: str, start: int, max_results: int = BATCH) -> list[dict]:
    params = urllib.parse.urlencode({
        "search_query": f"cat:{category}",
        "start":        start,
        "max_results":  max_results,
        "sortBy":       "submittedDate",
        "sortOrder":    "descending",
    })
    url = f"{ARXIV_API}?{params}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        xml = resp.read()

    root    = ET.fromstring(xml)
    entries = root.findall("atom:entry", NS)
    papers  = []
    for entry in entries:
        title    = entry.findtext("atom:title",   "", NS).replace("\n", " ").strip()
        abstract = entry.findtext("atom:summary", "", NS).replace("\n", " ").strip()
        arxiv_id = entry.findtext("atom:id", "", NS).strip()  # full URL e.g. http://arxiv.org/abs/2301.12345v1
        doi      = f"10.48550/{arxiv_id.split('abs/')[-1].split('v')[0]}"  # canonical arXiv DOI
        if title and abstract:
            papers.append({"doi": doi, "title": title, "abstract": abstract})
    return papers


def main():
    all_papers: list[dict] = []
    # Distribute N_TARGET across categories; random start offset per category
    per_cat = (N_TARGET // len(CATEGORIES)) + BATCH  # fetch a bit more than needed
    batches_per_cat = max(1, per_cat // BATCH)

    for cat in CATEGORIES:
        if len(all_papers) >= N_TARGET:
            break
        start_offset = random.randint(0, 5000)  # random offset into the category's paper list
        for b in range(batches_per_cat):
            start = start_offset + b * BATCH
            print(f"  {cat}  start={start} ...", end=" ", flush=True)
            try:
                papers = fetch_batch(cat, start, BATCH)
                all_papers.extend(papers)
                print(f"{len(papers)} fetched  (total {len(all_papers)})")
            except Exception as exc:
                print(f"ERROR: {exc}")
            time.sleep(DELAY)
            if len(all_papers) >= N_TARGET:
                break

    # Trim, deduplicate by doi, assign sequential paper_id
    seen = set()
    unique = []
    for p in all_papers:
        if p["doi"] not in seen:
            seen.add(p["doi"])
            unique.append(p)

    sample = unique[:N_TARGET]
    for i, p in enumerate(sample, start=1):
        p["paper_id"] = i

    print(f"\nTotal unique papers collected: {len(sample)}")

    # Write to parquet via DuckDB
    con = duckdb.connect()
    con.execute("""
        CREATE TABLE arxiv (
            paper_id INTEGER,
            doi      VARCHAR,
            title    VARCHAR,
            abstract VARCHAR
        )
    """)
    con.executemany(
        "INSERT INTO arxiv VALUES (?, ?, ?, ?)",
        [(p["paper_id"], p["doi"], p["title"], p["abstract"]) for p in sample]
    )
    con.execute(f"COPY (SELECT * FROM arxiv ORDER BY paper_id) TO '{OUTPUT}' (FORMAT PARQUET)")
    n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{OUTPUT}')").fetchone()[0]
    con.close()
    print(f"Saved {n} papers to {OUTPUT}")


if __name__ == "__main__":
    main()
