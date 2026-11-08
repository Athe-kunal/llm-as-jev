"""Fine-tuning a masked-language-model encoder as a decision model.

Usage:
  .venv/bin/accelerate launch --config_file configs/accelerate_ddp.yaml \
      -m llm_jev.train.train_mlm --config configs/train_mlm/ultrabert_380m.yaml \
      [optim.lr=1e-5 run.out=runs/x ...]

The encoder counterpart of train.py: one mask per option (dataloader.mlm),
cross-entropy over the K masks (modeling.encoder), full fine-tuning with fp32
weights under bf16 autocast. Every setting is in the YAML config
(config.mlm_config); dotlist arguments override it, and the resolved config
is saved as <run.out>/config.yaml. Evaluates before training, at the end and
every run.eval_every steps.
"""

import argparse
import contextlib
import json
import math
import pathlib
import random
import time
from typing import Any

from loguru import logger
from omegaconf import OmegaConf
import torch
from torch.optim.lr_scheduler import LinearLR
from torch.optim.lr_scheduler import SequentialLR

from llm_jev.config import mlm_config
from llm_jev.dataloader import mlm
from llm_jev.dataloader import schema
from llm_jev.dataloader import storage
from llm_jev.evaluation import evaluate
from llm_jev.evaluation import evaluate_mlm
from llm_jev.modeling import encoder


def row_rng(seed: int, epoch: int, i: int) -> random.Random:
    """The option-shuffling rng of training row i in an epoch."""
    return random.Random(f"{seed}:{epoch}:{i}")


def fitting_rows(acc, train: list[schema.Example], tok: Any, fmt: mlm.Format,
                 seed: int, max_tokens: int) -> list[int]:
    """Indices of the training rows whose encoding has at most max_tokens tokens."""
    from accelerate.utils import gather_object

    mine = [(i, len(mlm.encode(train[i], tok, fmt, row_rng(seed, 0, i))["input_ids"]))
            for i in range(acc.process_index, len(train), acc.num_processes)]
    lengths = dict(gather_object(mine))
    return [i for i in range(len(train)) if lengths[i] <= max_tokens]


def run_eval(acc, scorer, tok: Any, fmt: mlm.Format, rows: list[schema.Example],
             budget: int) -> dict[str, dict[str, float]]:
    """evaluation.evaluate_mlm metrics (the decoder's format: "ALL" averages
    the datasets), with the rows split over the processes and gathered."""
    from accelerate.utils import gather_object

    model = acc.unwrap_model(scorer)
    was_training = model.training
    model.eval()
    mine = evaluate_mlm.records(model, tok, fmt, rows[acc.process_index::acc.num_processes],
                                str(acc.device), log_every=float("inf"), budget=budget)
    model.train(was_training)
    return evaluate_mlm.summarize(gather_object(mine))


def main() -> None:
    """Trains run.epochs epochs and evaluates before and after."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="YAML (config.mlm_config)")
    parser.add_argument("overrides", nargs="*", help="dotlist, e.g. optim.lr=1e-5")
    args = parser.parse_args()
    raw = mlm_config.load(args.config, args.overrides)
    cfg = mlm_config.resolve(raw)

    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from transformers import AutoTokenizer

    out = pathlib.Path(cfg.run.out)
    out.mkdir(parents=True, exist_ok=True)
    acc = Accelerator(mixed_precision=cfg.run.mixed_precision,
                      log_with="tensorboard", project_dir=out)
    device = acc.device
    batch = cfg.optim.batch_size
    if batch % acc.num_processes:
        raise ValueError(f"optim.batch_size {batch} is not divisible by "
                         f"{acc.num_processes} processes")
    set_seed(cfg.run.seed)
    if acc.is_main_process:
        logger.add(out / "train.log")
        OmegaConf.save(raw, out / "config.yaml")
    else:
        logger.remove()
    acc.init_trackers("tb")

    tok = AutoTokenizer.from_pretrained(cfg.model.id, revision=cfg.model.revision)
    fmt = cfg.format
    train = storage.load_examples(cfg.data.source, cfg.data.train_split)
    rows = fitting_rows(acc, train, tok, fmt, cfg.run.seed, cfg.data.max_tokens)
    logger.info("{} of {} training rows are over {} tokens, dropped",
                len(train) - len(rows), len(train), cfg.data.max_tokens)
    evals = evaluate.subset(storage.load_examples(cfg.data.source, cfg.data.eval_split),
                            cfg.data.eval_per, cfg.data.eval_seed)

    model = encoder.load(cfg.model.id, cfg.model.revision, cfg.model.attention,
                         device=str(device), dtype=torch.float32)
    if cfg.model.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    scorer = encoder.Scorer(model, tok, fmt)
    logger.info("trainable parameters: {:,}",
                sum(p.numel() for p in scorer.parameters() if p.requires_grad))
    opt = torch.optim.AdamW(scorer.parameters(), lr=cfg.optim.lr,
                            betas=tuple(cfg.optim.betas), eps=cfg.optim.eps,
                            weight_decay=cfg.optim.weight_decay)
    per_epoch = len(rows) // batch
    total = cfg.run.epochs * per_epoch
    if cfg.run.max_steps > 0:
        total = min(total, cfg.run.max_steps)
    if total < 1:
        raise ValueError(f"{len(rows)} training rows are fewer than one batch of {batch}")
    warm = max(1, round(cfg.optim.warmup_ratio * total))
    sched = SequentialLR(opt, [
        LinearLR(opt, start_factor=1 / warm, end_factor=1.0, total_iters=max(1, warm - 1)),
        LinearLR(opt, start_factor=1.0, end_factor=cfg.optim.min_lr_ratio,
                 total_iters=max(1, total - warm)),
    ], milestones=[warm])
    scorer, opt = acc.prepare(scorer, opt)

    def evaluate_and_log(step: int, tag: str) -> None:
        t0 = time.time()
        metrics = run_eval(acc, scorer, tok, fmt, evals, cfg.model.pack_tokens)
        if acc.is_main_process:
            (out / f"eval_{tag}.json").write_text(json.dumps(metrics, indent=2))
        acc.log({f"eval/{k}/{m}": x for k, v in metrics.items()
                 for m, x in v.items()}, step=step)
        logger.info("eval {} at step {} ({:.1f}m): ALL acc {:.4f} log_p_gold {:.4f} | {}",
                    tag, step, (time.time() - t0) / 60, metrics["ALL"]["accuracy"],
                    metrics["ALL"]["log_p_gold"],
                    " ".join(f"{k} {v['accuracy']:.2f}" for k, v in metrics.items()
                             if k != "ALL"))

    if not cfg.run.skip_base_eval:
        evaluate_and_log(0, "base")
    scorer.train()
    order = []  # (epoch, row) in visiting order, reshuffled every epoch
    for epoch in range(cfg.run.epochs):
        shuffled = list(rows)
        random.Random(f"{cfg.run.seed}:order:{epoch}").shuffle(shuffled)
        order += [(epoch, i) for i in shuffled[: per_epoch * batch]]
    start, skipped = time.time(), 0
    for step in range(total):
        mine = order[step * batch: (step + 1) * batch][acc.process_index::acc.num_processes]
        sums = torch.zeros(2, device=device)  # loss, hits
        for j, (epoch, i) in enumerate(mine):
            ex = mlm.encode(train[i], tok, fmt, row_rng(cfg.run.seed, epoch, i))
            last = j == len(mine) - 1
            with contextlib.nullcontext() if last else acc.no_sync(scorer):
                loss, hit = encoder.loss(scorer, ex, device)
                acc.backward(loss / len(mine))
            sums += torch.stack([loss.detach(), hit.float()])
        sums = acc.reduce(sums, "sum") / batch
        norm = acc.clip_grad_norm_(scorer.parameters(), cfg.optim.max_grad_norm).item()
        if math.isfinite(norm):
            opt.step()
        else:
            skipped += 1
            logger.warning("step {}: skipped update, grad_norm {}", step + 1, norm)
        opt.zero_grad(set_to_none=True)
        sched.step()
        lr = sched.get_last_lr()[0]
        loss_mean, acc_mean = sums.tolist()
        acc.log({"train/loss": loss_mean, "train/accuracy": acc_mean,
                 "train/grad_norm": norm, "train/lr": lr}, step=step + 1)
        if (step + 1) % 5 == 0 or step == 0:
            el = time.time() - start
            logger.info(f"step {step + 1}/{total} loss {loss_mean:.4f} "
                        f"acc {acc_mean:.3f} grad_norm {norm:.3f} lr {lr:.2e} "
                        f"skipped {skipped} elapsed {el / 60:.1f}m "
                        f"eta {el / (step + 1) * (total - step - 1) / 60:.0f}m")
        if cfg.run.eval_every and (step + 1) % cfg.run.eval_every == 0 \
                and step + 1 < total:
            evaluate_and_log(step + 1, f"step{step + 1:04d}")
    acc.wait_for_everyone()
    if acc.is_main_process:
        acc.unwrap_model(scorer).model.save_pretrained(out / "model")
        tok.save_pretrained(out / "model")
    evaluate_and_log(total, "final")
    logger.info("done")
    acc.end_training()


if __name__ == "__main__":
    main()
