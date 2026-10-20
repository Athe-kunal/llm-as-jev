"""Builds AT2's training data: per-row ablation features and targets.

Usage:
  CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src .venv/bin/python -m dataloader.at2_data \
      --model Qwen/Qwen3-0.6B --adapter runs/qwen3_0.6b_lam0.01/adapter \
      --out runs/at2_qwen3_0.6b/data.npz

Samples --rows training rows evenly over the datasets whose rows have at
least two sentences (the paper uses 2,000) and stores, per row, the
features of --masks ablations [masks, L*H] (float16) and their targets
(logit q of the predicted answer) from modeling.at2.example_data.
"""

import argparse
import collections
import pathlib
import random

from loguru import logger
import numpy as np

from dataloader import data
from modeling import at2, context_cite, modeling


def sample_rows(rows: list[dict], n: int, seed: int) -> list[dict]:
    """n rows with at least two sentences, as even across datasets as possible."""
    by: dict[str, list] = collections.defaultdict(list)
    for r in rows:
        if len(context_cite.split_sources(r)) >= 2:
            by[r["source"]].append(r)
    rng = random.Random(seed)
    for pool in by.values():
        rng.shuffle(pool)
    out, i = [], 0
    while len(out) < n and any(i < len(p) for p in by.values()):
        out += [p[i] for p in by.values() if i < len(p)][: n - len(out)]
        i += 1
    return out


def main() -> None:
    """Writes the features, targets and row ids to --out."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--adapter", help="a saved LoRA adapter (else the base model)")
    parser.add_argument("--data", default=data.TRAIN_EVAL)
    parser.add_argument("--split", default="train")
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--masks", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    logger.add(out.with_suffix(".log"))
    tok = AutoTokenizer.from_pretrained(args.model)
    model = modeling.load(args.model, args.adapter)
    rows = sample_rows(data.load_rows(args.data, args.split), args.rows, args.seed)
    logger.info("{} rows: {}", len(rows), dict(collections.Counter(r["source"] for r in rows)))
    feats, targets = [], []
    for i, row in enumerate(rows, 1):
        x, y = at2.example_data(model, tok, row, args.masks, seed=args.seed)
        feats.append(x.cpu().numpy().astype(np.float16))
        targets.append(y.astype(np.float32))
        if i % 100 == 0 or i == len(rows):
            logger.info("{}/{} rows", i, len(rows))
    np.savez(out, features=np.stack(feats), targets=np.stack(targets),
             ids=np.array([r["id"] for r in rows]), sources=np.array([r["source"] for r in rows]))
    logger.info("wrote {}: features {}, targets {}", out, np.stack(feats).shape, np.stack(targets).shape)


if __name__ == "__main__":
    main()
