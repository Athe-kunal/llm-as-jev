"""ContextCite (full masking, FA2) against AttnLRP on the eval rows with gold evidence spans.

Every method gives one weight per source sentence for the model's decision;
each is scored on the same rows by:
- spans: context_cite.span_metrics against the gold spans (precision, recall,
  f_beta of the top 3, average precision, top-1);
- drop@k: how much logit q(answer) falls when the k highest-weighted sources
  are removed (full masking), the ContextCite paper's top-k removal test;
- lds: Spearman correlation, over held-out random ablations, between the
  summed weights of the kept sources and the actual logit q(answer)
  (linear datamodeling score);
- seconds per row.
Methods: contextcite, attnlrp_logit and attnlrp_contrast (attn_lrp.goal).
"""

import argparse
import dataclasses
import json
import pathlib
import random
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
from llm_jev.modeling import attn_lrp
from llm_jev.modeling import context_cite
from llm_jev.modeling import fa2
from llm_jev.modeling import masked_cite
from llm_jev.modeling import modeling

TOP_K = (1, 3, 5)


def timed(fn, *args, **kwargs) -> tuple[Any, float]:
    """fn's result and its wall time in seconds (CUDA synchronized)."""
    torch.cuda.synchronize()
    t0 = time.time()
    out = fn(*args, **kwargs)
    torch.cuda.synchronize()
    return out, time.time() - t0


def run_methods(model, tok, row: schema.Example) -> dict[str, tuple[dict, float]]:
    """Each method's (result, seconds) on one row."""
    return {
        "contextcite": timed(masked_cite.attribute, model, tok, row, removal="full"),
        "attnlrp_logit": timed(attn_lrp.attribute, model, tok, row, target="logit"),
        "attnlrp_contrast": timed(attn_lrp.attribute, model, tok, row, target="contrast"),
    }


def evaluate_row(model, tok, row: schema.Example, lds_masks: int = 32,
                 seed: int = 1_000_000) -> dict[str, Any]:
    """One row's metrics for every method (all explain ContextCite's answer)."""
    results = run_methods(model, tok, row)
    sources = results["contextcite"][0]["sources"]
    S = len(sources)
    answer = results["contextcite"][0]["answer"]
    ids, owner, _ = masked_cite.token_owners(tok, row, sources)
    held = context_cite.ablation_masks(lds_masks, S, seed=seed)         # [L, S], unseen by the surrogate
    removals = []                                                       # (method, k) per top-k mask
    for name, (res, _) in results.items():
        order = np.argsort([-s["w"] for s in res["sources"]], kind="stable")
        for k in TOP_K:
            if k < S:
                removals.append((name, k, order[:k]))
    keep = np.ones((1 + len(held) + len(removals), S), bool)            # full, held-out, top-k removals
    keep[1: 1 + len(held)] = held
    for i, (_, _, drop) in enumerate(removals):
        keep[1 + len(held) + i, drop] = False
    s = masked_cite.score_keep(model, tok, ids, owner, keep, len(row.options))
    y = np.array([context_cite.logit_q(x, answer) for x in s])
    out = {"id": row.id, "source": row.source, "sources": S, "answer": answer, "gold": row.gold,
           "q": results["contextcite"][0]["q"], "methods": {}}
    for name, (res, seconds) in results.items():
        w = np.array([x["w"] for x in res["sources"]])
        m = {"seconds": seconds, "agrees": res["answer"] == answer,
             **(context_cite.span_metrics(row, res) or {}),
             "lds": float(spearmanr(held @ w, y[1: 1 + len(held)]).statistic),
             "w": w.tolist()}
        for i, (n, k, _) in enumerate(removals):
            if n == name:
                m[f"drop@{k}"] = float(y[0] - y[1 + len(held) + i])
        if name != "contextcite":
            m["goal"], m["relevance"], m["context"] = res["goal"], res["relevance"], res["context"]
            m["rank_vs_cc"] = float(spearmanr(w, out["methods"]["contextcite"]["w"]).statistic)
        out["methods"][name] = m
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    """Mean of each numeric metric per method, over all rows ("all") and per source dataset."""
    acc: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        for name, m in r["methods"].items():
            for key, v in m.items():
                if isinstance(v, (int, float, bool)) and not (isinstance(v, float) and np.isnan(v)):
                    for group in ("all", r["source"]):
                        acc[group][name][key].append(float(v))
    return {g: {n: {k: float(np.mean(v)) for k, v in ms.items()} for n, ms in by.items()}
            for g, by in acc.items()}


def table(summary: dict, group: str = "all") -> str:
    """A plain-text table of one group's means."""
    cols = ["ap", "top1", "precision", "recall", "lds", "drop@1", "drop@3", "drop@5", "rank_vs_cc", "seconds"]
    lines = [f"{'method':<18}" + "".join(f"{c:>11}" for c in cols)]
    for name, m in summary[group].items():
        lines.append(f"{name:<18}" + "".join(f"{m[c]:>11.3f}" if c in m else f"{'-':>11}" for c in cols))
    return "\n".join(lines)


def main() -> None:
    """Attributes --per seeded rows with gold spans per dataset and writes the comparison."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="Athekunal/llm-jev-qwen3-0.6b", help="merged model id or folder")
    parser.add_argument("--adapter", help="LoRA adapter on --model (merged before use)")
    parser.add_argument("--data", default="Athekunal/llm-jev-reason-long")
    parser.add_argument("--split", default="eval")
    parser.add_argument("--source", nargs="*", help="only these datasets")
    parser.add_argument("--per", type=int, default=50, help="rows per dataset")
    parser.add_argument("--limit", type=int, help="at most this many rows in all, spread over the datasets")
    parser.add_argument("--spans", action="store_true",
                        help="gold spans from --data's generated `spans` config (train rows) instead of the rows' own")
    parser.add_argument("--lds-masks", type=int, default=32, help="held-out ablations for the LDS")
    parser.add_argument("--out", required=True, help="folder for rows.jsonl and summary.json")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = modeling.load(args.model, args.adapter)
    fa2.use(model)
    rows = storage.load_examples(args.data, args.split)
    if args.spans:
        generated = {r["id"]: r["spans"] for r in datasets.load_dataset(args.data, "spans", split="train")}
        rows = [dataclasses.replace(r, spans=tuple(schema.Span(**s) for s in generated[r.id]))
                for r in rows if generated.get(r.id)]
    rows = [r for r in rows
            if r.spans and (not args.source or r.source in args.source)
            and len(context_cite.split_sources(r)) >= 2
            and context_cite.gold_sentences(r, context_cite.split_sources(r))]  # spans not between sentences
    by: dict[str, list] = {}
    for r in rows:
        by.setdefault(r.source, []).append(r)
    picks = [random.Random(0).sample(rs, min(args.per, len(rs))) for _, rs in sorted(by.items())]
    chosen = [p[j] for j in range(args.per) for p in picks if j < len(p)][: args.limit]   # round-robin over datasets
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    done = []
    with open(out / "rows.jsonl", "w") as f:
        for i, row in enumerate(chosen):
            r = evaluate_row(model, tok, row, args.lds_masks)
            f.write(json.dumps(r) + "\n")
            f.flush()
            done.append(r)
            logger.info("{}/{} {} ({} sentences): {}", i + 1, len(chosen), row.id, r["sources"],
                        {n: (round(m.get("ap", float("nan")), 2), round(m["lds"], 2), round(m["seconds"], 2))
                         for n, m in r["methods"].items()})
    summary = summarize(done)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    for group in ["all"] + sorted(g for g in summary if g != "all"):
        logger.info("{} ({} rows)\n{}", group, sum(r["source"] == group or group == "all" for r in done),
                    table(summary, group))
    logger.info("wrote {}", out)


if __name__ == "__main__":
    main()
