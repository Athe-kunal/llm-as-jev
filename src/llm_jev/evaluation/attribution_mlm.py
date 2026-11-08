"""Citation for an encoder: AT2, average attention and ContextCite on rows with gold evidence.

Usage:
  CUDA_VISIBLE_DEVICES=3 .venv/bin/python -m llm_jev.evaluation.attribution_mlm \
      --at2 runs/at2_ultrabert_380m [--per 50]

The encoder is the AT2 run's (its config.yaml); theta is <at2>/theta.pt. Each
method gives per-sentence weights for the model's own answer; each is scored
as evaluation.attribution scores the decoder's ContextCite:
- spans: context_cite.span_metrics against the gold spans (ap, top1, and
  precision / recall of the top 3);
- lds: Spearman correlation, over held-out random ablations, between the
  summed weights of the kept sentences and the actual logit q(answer);
- drop@k: how much logit q(answer) falls when the method's k highest-weighted
  sentences are deleted;
- seconds per row.
All methods share the held-out ablations, and one batched ablation call per
row scores them and every method's top-k removals.
"""

import argparse
import collections
import json
import pathlib
import random
import time

from loguru import logger
import numpy as np
from scipy.stats import spearmanr
import torch

from llm_jev.config import at2_config
from llm_jev.dataloader import storage
from llm_jev.evaluation import evaluate_mlm
from llm_jev.modeling import context_cite
from llm_jev.modeling import encoder_at2
from llm_jev.modeling import encoder_cite

TOP_K = (1, 3, 5)
METRICS = ["ap", "top1", "precision", "recall", "lds", "drop@1", "drop@3", "drop@5", "seconds"]


def load(at2_dir: str | pathlib.Path):
    """(scorer, tokenizer, format, theta, budget) of an AT2 run directory."""
    at2_dir = pathlib.Path(at2_dir)
    cfg = at2_config.resolve(at2_config.load(at2_dir / "config.yaml"))
    scorer, tok, fmt = evaluate_mlm.load_run(cfg.encoder.run, attention=cfg.encoder.attention)
    theta = torch.load(at2_dir / "theta.pt").cuda()
    return scorer, tok, fmt, theta, evaluate_mlm.run_config(cfg.encoder.run).model.pack_tokens


def methods(scorer, tok, fmt, theta: torch.Tensor, budget: int) -> dict:
    """name -> row -> context_cite-style result."""
    uniform = torch.full_like(theta, 1 / len(theta))
    return {
        "at2": lambda r: encoder_at2.attribute(scorer, tok, fmt, r, theta),
        "average_attention": lambda r: encoder_at2.attribute(scorer, tok, fmt, r, uniform),
        "contextcite": lambda r: encoder_cite.attribute(scorer, tok, fmt, r, budget=budget),
    }


def select_rows(examples: list, per: int = 50, seed: int = 0) -> list:
    """The evaluation rows: `per` rows per dataset with gold spans and >= 2 sentences (seeded)."""
    by = collections.defaultdict(list)
    for r in examples:
        if r.has_spans and len(context_cite.split_sources(r)) >= 2:
            by[r.source].append(r)
    rng = random.Random(seed)
    return [r for s in sorted(by) for r in rng.sample(by[s], min(per, len(by[s])))]


def evaluate_row(scorer, tok, fmt, row, fns: dict, lds_masks: int = 32, seed: int = 1_000_000,
                 budget: int = 32768) -> dict:
    """Every method's result and metrics on one row."""
    res, seconds = {}, {}
    for name, fn in fns.items():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        res[name] = fn(row)
        torch.cuda.synchronize()
        seconds[name] = time.perf_counter() - t0
    answer = res["contextcite"]["answer"]
    sources = res["contextcite"]["sources"]
    S = len(sources)
    held = context_cite.ablation_masks(lds_masks, S, seed=seed)          # [L, S], unseen by ContextCite
    ks = [k for k in TOP_K if k < S]
    keep = [np.ones(S, bool), *held]
    where = {}
    for name, r in res.items():
        order = np.argsort(-np.array([x["w"] for x in r["sources"]]), kind="stable")
        for k in ks:
            m = np.ones(S, bool)
            m[order[:k]] = False
            where[name, k] = len(keep)
            keep.append(m)
    y = encoder_at2.ablation_targets(scorer, tok, fmt, row, sources, np.stack(keep), answer, budget=budget)
    out = {"id": row.id, "source": row.source, "sources": S, "answer": answer, "gold": row.gold,
           "answers_agree": len({r["answer"] for r in res.values()}) == 1, "methods": {}}
    for name, r in res.items():
        w = np.array([x["w"] for x in r["sources"]])
        lds = spearmanr(held @ w, y[1: 1 + len(held)]).statistic if w.std() > 0 else 0.0
        out["methods"][name] = {**(context_cite.span_metrics(row, r) or {}), "lds": float(lds),
                                **{f"drop@{k}": float(y[0] - y[where[name, k]]) for k in ks},
                                "seconds": seconds[name], "w": w.tolist()}
    return out


def summarize(rows: list[dict]) -> dict:
    """Mean of each metric per dataset and method; "ALL" averages the datasets."""
    acc: dict = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        for name, m in r["methods"].items():
            for k in METRICS:
                if k in m and m[k] == m[k]:  # skip NaN
                    acc[(r["source"], name)][k].append(m[k])
    out: dict = collections.defaultdict(dict)
    for (src, name), ms in acc.items():
        out[src][name] = {k: float(np.mean(v)) for k, v in ms.items()} | {"rows": len(ms["seconds"])}
    names = {n for d in out.values() for n in d}
    out["ALL"] = {n: {k: float(np.mean([out[s][n][k] for s in out if s != "ALL" and k in out[s][n]]))
                      for k in METRICS} for n in names}
    return dict(out)


def table(summary: dict) -> str:
    """The summary as a fixed-width table."""
    lines = [f"{'dataset':12s} {'method':18s} " + " ".join(f"{k:>8s}" for k in METRICS)]
    for src in sorted(summary, key=lambda s: (s == "ALL", s)):
        for name in ("at2", "average_attention", "contextcite"):
            if name in summary[src]:
                m = summary[src][name]
                lines.append(f"{src:12s} {name:18s} " + " ".join(
                    f"{m.get(k, float('nan')):8.3f}" for k in METRICS))
    return "\n".join(lines)


def main() -> None:
    """Writes <at2>/attribution_<split>.json and logs the table."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--at2", required=True, help="AT2 run directory (train.train_at2)")
    parser.add_argument("--data", default="data/splits/reason_long_nimble")
    parser.add_argument("--split", default="eval")
    parser.add_argument("--per", type=int, default=50, help="rows with gold evidence per dataset")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    at2_dir = pathlib.Path(args.at2)
    logger.add(at2_dir / f"attribution_{args.split}.log")
    scorer, tok, fmt, theta, budget = load(at2_dir)
    rows = select_rows(storage.load_examples(args.data, args.split), args.per, args.seed)
    logger.info("{} rows: {}", len(rows), dict(collections.Counter(r.source for r in rows)))
    fns = methods(scorer, tok, fmt, theta, budget)
    results = []
    for i, r in enumerate(rows, 1):
        results.append(evaluate_row(scorer, tok, fmt, r, fns, budget=budget))
        if i % 25 == 0 or i == len(rows):
            logger.info("{}/{} rows", i, len(rows))
    summary = summarize(results)
    (at2_dir / f"attribution_{args.split}.json").write_text(json.dumps({"summary": summary, "rows": results}))
    logger.info("\n{}", table(summary))
    logger.info("answers agree across methods on {}/{} rows", sum(r["answers_agree"] for r in results), len(results))


if __name__ == "__main__":
    main()
