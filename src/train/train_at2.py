"""Trains AT2's score estimator on precomputed ablation data (dataloader.at2_data).

Usage:
  CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src .venv/bin/python -m train.train_at2 \
      --data runs/at2_qwen3_0.6b/data.npz --out runs/at2_qwen3_0.6b

As the reference's LinearScoreEstimatorTrainer.train: batches of rows
sampled without replacement, loss = losses.pearson_loss between features @
theta and the targets, Adam with a cosine schedule, gradient clipping;
theta is L1-normalised at the end and saved to <out>/theta.pt. --val-frac
rows are held out to report the Pearson correlation on unseen rows (the
reference trains on all rows).
"""

import argparse
import pathlib

from loguru import logger
import numpy as np
import torch

from modeling import at2
from train import losses


def split(Y: torch.Tensor, val_frac: float, seed: int):
    """The rows kept (targets that vary across masks: Pearson needs it) and a
    seeded train / held-out split of them, shared with train.train_citation so
    both are scored on the same rows. Returns (keep [R] bool, train indices,
    held-out indices, the generator for later shuffling)."""
    keep = Y.std(1) > 0
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(int(keep.sum()), generator=gen)
    n_val = int(len(perm) * val_frac)
    return keep, perm[n_val:], perm[:n_val], gen


def main() -> None:
    """Writes <out>/theta.pt and <out>/train_at2.log."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logger.add(out / "train_at2.log")
    d = np.load(args.data)
    X = torch.tensor(d["features"], dtype=torch.float32)                      # [R, M, F]
    Y = torch.tensor(d["targets"], dtype=torch.float32)                       # [R, M]
    keep, tr, val, gen = split(Y, args.val_frac, args.seed)
    X, Y, sources = X[keep], Y[keep], d["sources"][keep.numpy()]
    logger.info("{} rows ({} without target variance dropped): {} train, {} val; {} masks, {} features",
                len(Y), int((~keep).sum()), len(tr), len(val), X.shape[1], X.shape[2])
    X, Y = X.to(args.device), Y.to(args.device)
    est = at2.ScoreEstimator(X.shape[2]).to(args.device)

    def corr(idx):
        with torch.no_grad():
            return -losses.pearson_loss(est(X[idx]), Y[idx]).item()

    logger.info("before training: Pearson train {:.4f} val {:.4f}", corr(tr), corr(val))
    opt = torch.optim.Adam(est.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    bs = min(args.batch_size, len(tr))
    for step in range(args.steps):
        batch = tr[torch.randperm(len(tr), generator=gen)[:bs]]
        loss = losses.pearson_loss(est(X[batch]), Y[batch])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(est.parameters(), args.max_grad_norm)
        opt.step()
        sched.step()
        if step == 0 or (step + 1) % 100 == 0:
            logger.info("step {}/{}: loss {:.4f}, Pearson val {:.4f}", step + 1, args.steps, loss.item(), corr(val))
    theta = est.theta().cpu()
    torch.save(theta, out / "theta.pt")
    per_source = {s: -losses.pearson_loss(est(X[val][torch.tensor(sources[val.numpy()] == s)]),
                                          Y[val][torch.tensor(sources[val.numpy()] == s)]).item()
                  for s in np.unique(sources[val.numpy()])}
    logger.info("after training: Pearson train {:.4f} val {:.4f}; val by dataset {}", corr(tr), corr(val),
                {k: round(v, 3) for k, v in per_source.items()})
    logger.info("wrote {}", out / "theta.pt")


if __name__ == "__main__":
    main()
