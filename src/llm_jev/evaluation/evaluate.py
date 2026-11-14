"""Evaluates any decision model on a held-out split.

Usage:
  llm-jev-evaluate --config configs/evaluate/encoder.yaml \
      model.path=runs/ultrabert_380m out=runs/ultrabert_380m tag=final

Metrics per dataset and their mean over datasets ("ALL"): accuracy of
argmax s and mean log q(gold), q = softmax(s). Writes eval_<tag>.json, and
with save_rows eval_<tag>_rows.json (per-row correctness, for bootstrap
intervals).
"""

import collections
from collections.abc import Sequence
import dataclasses
import json
import pathlib
import time
from typing import Any

from loguru import logger
import torch

from llm_jev import config
from llm_jev import models
from llm_jev.data import sampling
from llm_jev.data import schema
from llm_jev.data import storage
from llm_jev.models import base

Metrics = dict[str, dict[str, float]]


@dataclasses.dataclass(frozen=True)
class RowResult:
    """One scored row.

    Attributes:
      id: The row's id.
      source: Its dataset.
      correct: Whether argmax s is the gold option.
      log_p_gold: log softmax(s)[gold].
      prediction: argmax s (release order).
    """

    id: str
    source: str
    correct: bool
    log_p_gold: float
    prediction: int


def score_rows(
    model: base.DecisionModel,
    rows: Sequence[schema.Example],
    chunk: int = 256,
) -> list[RowResult]:
    """Scores rows in chunks (logging progress) and grades them."""
    out: list[RowResult] = []
    start = time.time()
    for i in range(0, len(rows), chunk):
        part = rows[i : i + chunk]
        for row, s in zip(part, model.score(part), strict=True):
            out.append(
                RowResult(
                    id=row.id,
                    source=row.source,
                    correct=int(s.argmax()) == row.gold,
                    log_p_gold=torch.log_softmax(s.float(), 0)[row.gold].item(),
                    prediction=int(s.argmax()),
                )
            )
        elapsed = time.time() - start
        logger.info(
            "scored {}/{} rows, {:.1f}m, eta {:.1f}m",
            len(out),
            len(rows),
            elapsed / 60,
            elapsed / len(out) * (len(rows) - len(out)) / 60,
        )
    return out


def summarize(results: Sequence[RowResult]) -> Metrics:
    """Per-dataset accuracy, log_p_gold and rows; "ALL" averages datasets."""
    by_source: dict[str, list[RowResult]] = collections.defaultdict(list)
    for r in results:
        by_source[r.source].append(r)
    metrics: Metrics = {
        name: {
            "accuracy": sum(r.correct for r in rs) / len(rs),
            "log_p_gold": sum(r.log_p_gold for r in rs) / len(rs),
            "rows": len(rs),
        }
        for name, rs in sorted(by_source.items())
    }
    metrics["ALL"] = {
        key: sum(m[key] for m in metrics.values()) / len(metrics)
        for key in ("accuracy", "log_p_gold")
    }
    return metrics


def evaluate_distributed(
    model: base.DecisionModel,
    rows: Sequence[schema.Example],
    accelerator: Any,
) -> Metrics:
    """`summarize(score_rows(...))` across the processes of a training run.

    Under FSDP every process must run the same forwards, so each scores all
    rows; otherwise the rows are split over the processes and gathered.
    """
    from accelerate.utils import DistributedType
    from accelerate.utils import gather_object

    was_training = model.network.training
    model.eval()
    if accelerator.distributed_type == DistributedType.FSDP:
        results = score_rows(model, rows)
    else:
        mine = list(
            rows[accelerator.process_index :: accelerator.num_processes]
        )
        results = gather_object(score_rows(model, mine))
    model.train(was_training)
    return summarize(results)


def save(out: pathlib.Path, tag: str, metrics: Metrics) -> None:
    """Writes <out>/eval_<tag>.json."""
    (out / f"eval_{tag}.json").write_text(json.dumps(metrics, indent=2))


def save_rows(
    out: pathlib.Path, tag: str, results: Sequence[RowResult]
) -> None:
    """Writes <out>/eval_<tag>_rows.json (id, dataset, correctness)."""
    rows = [
        {"id": r.id, "source": r.source, "correct": r.correct} for r in results
    ]
    (out / f"eval_{tag}_rows.json").write_text(json.dumps(rows))


def log_summary(tag: str, metrics: Metrics) -> None:
    """Logs the mean metrics, then accuracy per dataset."""
    logger.info(
        "eval {}: {}",
        tag,
        " ".join(f"{k} {v:.4f}" for k, v in metrics["ALL"].items()),
    )
    logger.info(
        "eval {} accuracy per dataset: {}",
        tag,
        " ".join(
            f"{name} {m['accuracy']:.3f}"
            for name, m in metrics.items()
            if name != "ALL" and "accuracy" in m
        ),
    )


def main() -> None:
    """Evaluates a model on one part of a split."""
    cfg, _ = config.from_command_line(config.EvaluateConfig, __doc__)
    rows = sampling.select_part(
        storage.load_examples(cfg.source, cfg.split),
        cfg.part,
        cfg.eval_per,
        cfg.seed,
        sampling.read_ids(cfg.rows_file),
    )
    model = models.load(cfg.model)
    results = score_rows(model, rows)
    metrics = summarize(results)
    out = pathlib.Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    save(out, cfg.tag, metrics)
    if cfg.save_rows:
        save_rows(out, cfg.tag, results)
    log_summary(cfg.tag, metrics)


if __name__ == "__main__":
    main()
