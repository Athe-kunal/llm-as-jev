"""Trains AT2 for an encoder run: ablation data (stage 1), then theta (stage 2).

Usage:
  CUDA_VISIBLE_DEVICES=3 .venv/bin/python -m llm_jev.train.train_at2 \
      --config configs/at2/ultrabert_380m.yaml [data.rows=500 out=runs/x ...]

Stage 1 builds <out>/data.npz (skipped when it exists, so theta can be
re-fitted without recomputing): for data.rows training rows, balanced over
data.datasets, the features [masks, L*H] and targets [masks] of
modeling.encoder_at2.example_data. Stage 2 fits theta as the reference's
LinearScoreEstimatorTrainer (and the Qwen train_at2): batches of rows,
losses.pearson_loss, Adam with a cosine schedule, gradient clipping; train.
val_frac of the rows are held out, and the held-out Pearson per dataset is
logged before (uniform theta, the average-attention baseline) and after.
Writes <out>/config.yaml, theta.pt, metrics.json and train_at2.log.
"""

import argparse
import collections
import json
import pathlib
import random

from loguru import logger
import numpy as np
from omegaconf import OmegaConf
import torch

from llm_jev.dataloader import storage
from llm_jev.evaluation import evaluate_mlm
from llm_jev.modeling import context_cite
from llm_jev.modeling import encoder_at2
from llm_jev.train import at2_config
from llm_jev.train import losses


def sample_rows(rows: list, datasets: list[str] | None, n: int, min_sources: int, seed: int) -> list:
    """n rows with at least min_sources sentences, as even across the datasets as possible."""
    by: dict[str, list] = collections.defaultdict(list)
    for r in rows:
        if (datasets is None or r.source in datasets) and len(context_cite.split_sources(r)) >= min_sources:
            by[r.source].append(r)
    rng = random.Random(seed)
    for name in sorted(by):
        rng.shuffle(by[name])
    out, i = [], 0
    while len(out) < n and any(i < len(p) for p in by.values()):
        out += [by[name][i] for name in sorted(by) if i < len(by[name])][: n - len(out)]
        i += 1
    return out


def build_data(cfg: at2_config.Config, path: pathlib.Path) -> None:
    """Stage 1: writes the features, targets, answers, ids and datasets to `path`."""
    scorer, tok, fmt = evaluate_mlm.load_run(cfg.encoder.run, attention=cfg.encoder.attention)
    budget = evaluate_mlm.run_config(cfg.encoder.run).model.pack_tokens
    rows = sample_rows(storage.load_examples(cfg.data.source, cfg.data.split), cfg.data.datasets,
                       cfg.data.rows, cfg.data.min_sources, cfg.data.seed)
    logger.info("{} rows: {}", len(rows), dict(collections.Counter(r.source for r in rows)))
    feats, targets, answers = [], [], []
    for i, r in enumerate(rows, 1):
        x, y, a = encoder_at2.example_data(scorer, tok, fmt, r, cfg.data.masks, cfg.data.seed, budget=budget)
        feats.append(x.cpu().numpy().astype(np.float16))
        targets.append(y.astype(np.float32))
        answers.append(a)
        if i % 100 == 0 or i == len(rows):
            logger.info("{}/{} rows", i, len(rows))
    np.savez(path, features=np.stack(feats), targets=np.stack(targets), answers=np.array(answers),
             ids=np.array([r.id for r in rows]), sources=np.array([r.source for r in rows]),
             gold=np.array([r.gold for r in rows]))
    logger.info("wrote {}: features {}, targets {}", path, np.stack(feats).shape, np.stack(targets).shape)


def fit(cfg: at2_config.Config, path: pathlib.Path, out: pathlib.Path, device: str = "cuda") -> dict:
    """Stage 2: fits theta on `path`; writes theta.pt; returns the metrics."""
    d = np.load(path)
    X = torch.tensor(d["features"], dtype=torch.float32)                      # [R, M, F]
    Y = torch.tensor(d["targets"], dtype=torch.float32)                       # [R, M]
    keep = Y.std(1) > 0                                                        # Pearson needs varying targets
    X, Y, sources = X[keep], Y[keep], d["sources"][keep.numpy()]
    gen = torch.Generator().manual_seed(cfg.train.seed)
    perm = torch.randperm(len(Y), generator=gen)
    n_val = int(len(Y) * cfg.train.val_frac)
    val, tr = perm[:n_val], perm[n_val:]
    logger.info("{} rows ({} without target variance dropped): {} train, {} held out; {} masks, {} features",
                len(Y), int((~keep).sum()), len(tr), len(val), X.shape[1], X.shape[2])
    X, Y = X.to(device), Y.to(device)
    est = encoder_at2.ScoreEstimator(X.shape[2]).to(device)

    def corr(idx) -> float:
        with torch.no_grad():
            return -losses.pearson_loss(est(X[idx]), Y[idx]).item()

    def by_source(idx) -> dict:
        names = sources[idx.numpy()]
        return {s: corr(idx[torch.tensor(names == s)]) for s in sorted(set(names))}

    metrics = {"before": {"train": corr(tr), "val": corr(val), "val_by_dataset": by_source(val)}}
    logger.info("before (uniform theta = average attention): Pearson train {:.4f} held-out {:.4f}",
                metrics["before"]["train"], metrics["before"]["val"])
    opt = torch.optim.Adam(est.parameters(), lr=cfg.train.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.train.steps)
    bs = min(cfg.train.batch_size, len(tr))
    for step in range(cfg.train.steps):
        batch = tr[torch.randperm(len(tr), generator=gen)[:bs]]
        loss = losses.pearson_loss(est(X[batch]), Y[batch])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(est.parameters(), cfg.train.max_grad_norm)
        opt.step()
        sched.step()
        if step == 0 or (step + 1) % 100 == 0:
            logger.info("step {}/{}: loss {:.4f}, Pearson held-out {:.4f}", step + 1, cfg.train.steps,
                        loss.item(), corr(val))
    metrics["after"] = {"train": corr(tr), "val": corr(val), "val_by_dataset": by_source(val)}
    torch.save(est.theta().cpu(), out / "theta.pt")
    logger.info("after: Pearson train {:.4f} held-out {:.4f}", metrics["after"]["train"], metrics["after"]["val"])
    for s in metrics["after"]["val_by_dataset"]:
        logger.info("  held-out {:12s} {:.3f} -> {:.3f}", s, metrics["before"]["val_by_dataset"][s],
                    metrics["after"]["val_by_dataset"][s])
    return metrics


def main() -> None:
    """Builds the data if missing, fits theta, writes the outputs."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="YAML (train.at2_config)")
    parser.add_argument("overrides", nargs="*", help="dotlist, e.g. data.rows=500")
    args = parser.parse_args()
    raw = at2_config.load(args.config, args.overrides)
    cfg = at2_config.resolve(raw)
    out = pathlib.Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    logger.add(out / "train_at2.log")
    OmegaConf.save(raw, out / "config.yaml")
    data = out / "data.npz"
    if data.exists():
        logger.info("using existing {} (delete it to rebuild)", data)
    else:
        build_data(cfg, data)
        torch.cuda.empty_cache()
    metrics = fit(cfg, data, out)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    logger.info("wrote {}", out / "theta.pt")


if __name__ == "__main__":
    main()
