import argparse
import json
import os

import pandas as pd
from tqdm import tqdm

from string_matcher import StringAgentMatcher
from semantic_matcher import SemanticAgentMatcher
from configs.config import config
from chem_bio_agent_lists import load_agent_synonyms

class AgentDetectionPipeline:
    def __init__(self, config):
        self.config = config
        self.agent_dict = load_agent_synonyms()
        self.string_matcher = StringAgentMatcher(self.agent_dict, config["string_matching"])
        self.semantic_matcher = SemanticAgentMatcher(self.agent_dict, config["semantic"]) if config["semantic"]["enabled"] else None

    def run(self, title, abstract):
        text = f"{title} {abstract}".lower()
        # Stage 1: string matching
        matches,str_patterns = self.string_matcher.match(text)
        if matches:
            return {"flag": True, "agents": matches, "method": "string", "str_patterns":str_patterns}
        # Stage 2: semantic fallback (if enabled)
        if self.semantic_matcher:
            matches = self.semantic_matcher.match(text)
            if matches:
                return {"flag": True, "agents": matches, "method": "semantic"}
        return {"flag": False, "agents": [], "str_patterns":[]}
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="./data/biorXiv24_25.parquet", help="Input file (.parquet or .csv)")
    parser.add_argument("--output", choices=["jsonl", "duckdb"], default="jsonl")
    args = parser.parse_args()

    df = pd.read_csv(args.input) if args.input.endswith(".csv") else pd.read_parquet(args.input)
    pipeline = AgentDetectionPipeline(config)
    df_sample = df[["paper_id", "doi", "title", "abstract"]]
    all_res = []
    for idx, row in tqdm(df_sample.iterrows()):
        res = pipeline.run(row["title"], row["abstract"])
        # if res["flag"]:
            # print(f"DOI: {row['doi']} - Detected: {res.get('agent')} via {res.get('method')}")
        
        all_res.append({"paper_id": int(row["paper_id"]), "doi": row["doi"], "flag": res["flag"], "agent": (res.get("agents") or [None])[0], "method": res.get("method"), "str_patterns":(res.get("str_patterns") or [None])[0]})

    stem = os.path.splitext(os.path.basename(args.input))[0]  # e.g. "biorXiv24_25"
    os.makedirs("data", exist_ok=True)
    if args.output == "jsonl":
        out_path = f"data/{stem}_screenv2.jsonl"
        with open(out_path, "w") as f:
            for r in all_res:
                f.write(json.dumps(r) + "\n")
        print(f"Wrote {len(all_res)} rows → {out_path}")
    else:
        import duckdb
        out_path = f"data/{stem}_screen.duckdb"
        con = duckdb.connect(out_path)
        con.execute("CREATE OR REPLACE TABLE screen_results (paper_id INTEGER PRIMARY KEY, doi TEXT, flag BOOLEAN, agent TEXT, method TEXT, str_patterns TEXT)")
        con.executemany("INSERT OR REPLACE INTO screen_results VALUES (?, ?, ?, ?, ?, ?)",
                        [(r["paper_id"], r["doi"], r["flag"], r["agent"], r["method"], r["str_patterns"]) for r in all_res])
        con.close()
        print(f"Wrote {len(all_res)} rows → {out_path}")
