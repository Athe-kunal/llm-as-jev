"""ContextCite (full masking, FA2) on the rows with gold evidence spans.

Each row's per-sentence weights are scored by:
- spans: context_cite.span_metrics against the gold spans (precision, recall,
  f_beta of the top 3, average precision, top-1);
- drop@k: how much logit q(answer) falls when the k highest-weighted sources
  are removed (full masking), the ContextCite paper's top-k removal test;
- lds: Spearman correlation, over held-out random ablations, between the
  summed weights of the kept sources and the actual logit q(answer)
  (linear datamodeling score);
- seconds per row.
"""

import argparse
import dataclasses
import json
import pathlib
import time
from collections import defaultdict
from typing import Any

import datasets
from loguru import logger
import numpy as np
from scipy.stats import spearmanr
import torch

from llm_jev.dataloader import schema
from llm_jev.dataloader import storage
from llm_jev.evaluation import evaluate
from llm_jev.modeling import context_cite
from llm_jev.modeling import fa2
from llm_jev.modeling import masked_cite
from llm_jev.modeling import modeling

TOP_K = (1, 3, 5)
METRICS = ["ap", "top1", "precision", "recall", "lds", "drop@1", "drop@3", "drop@5", "seconds"]


def evaluate_row(model, tok, row: schema.Example, lds_masks: int = 32,
                 seed: int = 1_000_000) -> dict[str, Any]:
    """One row's ContextCite result and metrics. The model must use modeling.fa2."""
    torch.cuda.synchronize()
    t0 = time.time()
    res = masked_cite.attribute(model, tok, row, removal="full")
    torch.cuda.synchronize()
    seconds = time.time() - t0
    S, answer = len(res["sources"]), res["answer"]
    w = np.array([x["w"] for x in res["sources"]])
    order = np.argsort(-w, kind="stable")
    ks = [k for k in TOP_K if k < S]
    held = context_cite.ablation_masks(lds_masks, S, seed=seed)         # [L, S], unseen by the surrogate
    keep = np.ones((1 + len(held) + len(ks), S), bool)                  # full, held-out, top-k removals
    keep[1: 1 + len(held)] = held
    for i, k in enumerate(ks):
        keep[1 + len(held) + i, order[:k]] = False
    ids, owner, _ = masked_cite.token_owners(tok, row, res["sources"])
    s = masked_cite.score_keep(model, tok, ids, owner, keep, len(row.options))
    y = np.array([context_cite.logit_q(x, answer) for x in s])
    return {"id": row.id, "source": row.source, "sources": S, "answer": answer, "gold": row.gold,
            "q": res["q"], "seconds": seconds, **(context_cite.span_metrics(row, res) or {}),
            "lds": float(spearmanr(held @ w, y[1: 1 + len(held)]).statistic),
            **{f"drop@{k}": float(y[0] - y[1 + len(held) + i]) for i, k in enumerate(ks)},
            "w": w.tolist()}


def summarize(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Mean of each metric over all rows ("all") and per source dataset, with the row counts."""
    acc: dict = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for group in ("all", r["source"]):
            acc[group]["rows"].append(1)
            for key in METRICS:
                if not np.isnan(r.get(key, np.nan)):
                    acc[group][key].append(r[key])
    return {g: {k: float(np.sum(v) if k == "rows" else np.mean(v)) for k, v in m.items()}
            for g, m in acc.items()}


def table(summary: dict) -> str:
    """A plain-text table of the means, one line per group."""
    lines = [f"{'group':<14}{'rows':>6}" + "".join(f"{c:>10}" for c in METRICS)]
    for g in ["all"] + sorted(x for x in summary if x != "all"):
        m = summary[g]
        lines.append(f"{g:<14}{int(m['rows']):>6}" + "".join(f"{m[c]:>10.3f}" if c in m else f"{'-':>10}"
                                                        for c in METRICS))
    return "\n".join(lines)


def main() -> None:
    """Attributes --per seeded rows with gold spans per dataset and writes the metrics."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="Athekunal/llm-jev-qwen3-0.6b", help="merged model id or folder")
    parser.add_argument("--adapter", help="LoRA adapter on --model (merged before use)")
    parser.add_argument("--data", default="Athekunal/llm-jev-reason-long")
    parser.add_argument("--split", default="eval")
    parser.add_argument("--source", nargs="*", help="only these datasets")
    parser.add_argument("--per", type=int, default=50, help="rows per dataset")
    parser.add_argument("--spans", action="store_true",
                        help="gold spans from --data's generated `spans` config (train rows) instead of the rows' own")
    parser.add_argument("--lds-masks", type=int, default=32, help="held-out ablations for the LDS")
    parser.add_argument("--fuse", action=argparse.BooleanOptionalAction, default=True,
                        help="fused kernels: compile the decoder layers (modeling.fuse)")
    parser.add_argument("--out", required=True, help="folder for rows.jsonl and summary.json")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = modeling.load(args.model, args.adapter)
    fa2.use(model)
    if args.fuse:
        modeling.fuse(model)
    rows = storage.load_examples(args.data, args.split)
    if args.spans:
        generated = {r["id"]: r["spans"] for r in datasets.load_dataset(args.data, "spans", split="train")}
        rows = [dataclasses.replace(r, spans=tuple(schema.Span(**s) for s in generated[r.id]))
                for r in rows if generated.get(r.id)]
    rows = [r for r in rows
            if r.spans and (not args.source or r.source in args.source)
            and len(srcs := context_cite.split_sources(r)) >= 2
            and context_cite.gold_sentences(r, srcs)]                    # spans not between sentences
    chosen = evaluate.subset(rows, args.per, seed=0)
    if args.fuse and chosen:
        masked_cite.attribute(model, tok, chosen[0], removal="full")      # compiles outside the timings
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    done = []
    with open(out / "rows.jsonl", "w") as f:
        for i, row in enumerate(chosen):
            r = evaluate_row(model, tok, row, args.lds_masks)
            f.write(json.dumps(r) + "\n")
            f.flush()
            done.append(r)
            logger.info("{}/{} {} ({} sentences): ap {:.2f} lds {:.2f} {:.2f}s", i + 1, len(chosen), row.id,
                        r["sources"], r.get("ap", float("nan")), r["lds"], r["seconds"])
    summary = summarize(done)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("ContextCite\n{}", table(summary))
    logger.info("wrote {}", out)


if __name__ == "__main__":
    main()
