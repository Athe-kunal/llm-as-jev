"""Exports the appendix examples: decision, option probabilities and evidence-head token scores.

Needs the Try-it server (blogs/server.py, with the evidence head) running, e.g. on port 8767:

  .venv/bin/python paper/figures/export_examples.py [--port 8767]

Rule, fixed in advance: the evaluation rows of the evidence study (attribution_mlm.select_rows, seed 0,
50 rows per dataset); for each dataset, the first row in that order whose prompt has at most MAX_TOKENS
tokens. A dataset with no such row is left out. Writes paper/figures/examples.json.
"""
import argparse
import json
import pathlib
import urllib.request

from llm_jev.dataloader import storage
from llm_jev.evaluation import attribution_mlm

MAX_TOKENS = 1400
OUT = pathlib.Path(__file__).parent / "examples.json"


def get(port, path):
    return json.load(urllib.request.urlopen(f"http://localhost:{port}{path}"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8767)
    port = parser.parse_args().port
    rows = attribution_mlm.select_rows(storage.load_examples(source="data/splits/reason_long_nimble", split="eval"), 50, 0)
    chosen = {}
    for r in rows:
        if r.source in chosen:
            continue
        row = get(port, f"/api/row?id={r.id}")
        if row["tokens"] <= MAX_TOKENS:
            chosen[r.source] = {"row": row, "evidence": get(port, f"/api/attribute?id={r.id}")["evidence"]}
    info = get(port, "/api/info")
    OUT.write_text(json.dumps({"run": info["run"], "examples": chosen}))
    print({k: (v["row"]["tokens"], v["row"]["answer"] == v["row"]["gold"]) for k, v in chosen.items()})


if __name__ == "__main__":
    main()
