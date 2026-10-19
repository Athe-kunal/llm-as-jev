"""Evidence attribution scored against gold spans: AT2 (trained and uniform theta), optionally ContextCite.

Usage:
  CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src .venv/bin/python -m evaluation.attribution \
      --adapter runs/qwen3_0.6b_lam0.01/adapter --theta runs/at2_qwen3_0.6b/theta.pt \
      --out runs/at2_qwen3_0.6b/eval_attribution.json [--contextcite]

The rows are --per eval rows of each dataset with gold spans (seeded as in
docs/architecture.md §7 step 5b); every method is scored with
context_cite.span_metrics (top-k selection, AP, top-1).
"""

import argparse
import collections
import json
import pathlib
import random
import time

from loguru import logger
import numpy as np
import torch

from dataloader import data
from modeling import at2, context_cite, modeling


def main() -> None:
    """Writes per-dataset mean metrics for each method to --out."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--adapter")
    parser.add_argument("--theta", required=True)
    parser.add_argument("--data", default=data.TRAIN_EVAL)
    parser.add_argument("--per", type=int, default=20)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--contextcite", action="store_true", help="also run ContextCite (64 ablations per row)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = (modeling.load_trained(args.model, args.adapter).merge_and_unload() if args.adapter
             else modeling.load_base(args.model).to("cuda").eval())
    theta = torch.load(args.theta, map_location="cpu").float()
    methods = {"at2": lambda r: at2.attribute(model, tok, r, theta),
               "average_attention": lambda r: at2.attribute(model, tok, r, torch.ones_like(theta) / len(theta))}
    if args.contextcite:
        methods["contextcite"] = lambda r: context_cite.attribute(model, tok, r)
    by = collections.defaultdict(list)
    for r in data.load_rows(args.data, "eval"):
        if r["spans"]:
            by[r["source"]].append(r)
    results = {}
    for src, rs in sorted(by.items()):
        sample = random.Random(0).sample(rs, min(args.per, len(rs)))
        for name, fn in methods.items():
            m, t0 = collections.defaultdict(list), time.time()
            for r in sample:
                res = fn(r)
                for key, v in context_cite.span_metrics(r, res, k=args.k).items():
                    m[key].append(v)
            results.setdefault(src, {})[name] = {**{k: float(np.mean(v)) for k, v in m.items()},
                                                 "seconds_per_row": (time.time() - t0) / len(sample)}
            logger.info("{} {}: {}", src, name, {k: round(v, 3) for k, v in results[src][name].items()})
    pathlib.Path(args.out).write_text(json.dumps(results, indent=2))
    logger.info("wrote {}", args.out)


if __name__ == "__main__":
    main()
