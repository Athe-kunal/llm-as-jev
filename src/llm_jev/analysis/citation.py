"""Compares citation baselines for an encoder on rows with gold evidence.

Usage:
  python -m llm_jev.analysis.citation at2=runs/at2_ultrabert_380m part=test

Methods: AT2 (theta from <at2>/theta.pt), average attention (uniform
theta) and ContextCite by deletion, all on the AT2 run's encoder. Each gives
per-sentence weights for the model's own answer, scored by:
- span metrics against the gold spans (ap, top1, precision / recall @ 3);
- lds: Spearman correlation, over held-out random ablations, between the
  summed weights of the kept sentences and the actual logit q(answer);
- drop@k: how much logit q(answer) falls when the k highest-weighted
  sentences are deleted;
- seconds per row.
Writes <at2>/attribution_<split>[_<part>].json and logs a table.
"""

import collections
from collections.abc import Callable
from collections.abc import Sequence
import json
import pathlib
import time
from typing import Any

from loguru import logger
import numpy as np
import torch

from llm_jev import config
from llm_jev.analysis import at2
from llm_jev.analysis import context_cite
from llm_jev.data import schema
from llm_jev.data import storage
from llm_jev.evidence import metrics as evidence_metrics
from llm_jev.evidence import sentences as sentences_lib
from llm_jev.models import encoder

TOP_K = (1, 3, 5)
METRICS = (
    "ap",
    "top1",
    "precision",
    "recall",
    "lds",
    "drop@1",
    "drop@3",
    "drop@5",
    "seconds",
)
Method = Callable[[schema.Example], context_cite.Attribution]


def methods(
    model: encoder.MaskedEncoder, theta: torch.Tensor
) -> dict[str, Method]:
    """Method name -> row -> attribution."""
    uniform = torch.full_like(theta, 1 / len(theta))
    return {
        "at2": lambda r: at2.attribute(model, r, theta),
        "average_attention": lambda r: at2.attribute(model, r, uniform),
        "contextcite": lambda r: context_cite.attribute(model, r),
    }


def _timed(
    model: encoder.MaskedEncoder, fn: Method, row: schema.Example
) -> tuple[Any, float]:
    if model.device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    result = fn(row)
    if model.device.type == "cuda":
        torch.cuda.synchronize()
    return result, time.perf_counter() - start


def evaluate_row(
    model: encoder.MaskedEncoder,
    row: schema.Example,
    fns: dict[str, Method],
    lds_masks: int = 32,
    seed: int = 1_000_000,
) -> dict[str, Any]:
    """Every method's weights and metrics on one row.

    All methods share the held-out ablations, and one batched call scores
    them together with every method's top-k removals.
    """
    results, seconds = {}, {}
    for name, fn in fns.items():
        results[name], seconds[name] = _timed(model, fn, row)
    answer = results["contextcite"].answer
    sentences = results["contextcite"].sentences
    n = len(sentences)
    held = sentences_lib.ablation_masks(lds_masks, n, seed=seed)  # [L, S]
    ks = [k for k in TOP_K if k < n]
    keep = [np.ones(n, bool), *held]
    where = {}
    for name, result in results.items():
        order = np.argsort(-result.weights, kind="stable")
        for k in ks:
            mask = np.ones(n, bool)
            mask[order[:k]] = False
            where[name, k] = len(keep)
            keep.append(mask)
    y = context_cite.ablation_targets(
        model, row, sentences, np.stack(keep), answer
    )
    out: dict[str, Any] = {
        "id": row.id,
        "source": row.source,
        "sources": n,
        "answer": answer,
        "gold": row.gold,
        "answers_agree": len({r.answer for r in results.values()}) == 1,
        "methods": {},
    }
    from scipy.stats import spearmanr

    for name, result in results.items():
        w = result.weights
        lds = (
            spearmanr(held @ w, y[1 : 1 + len(held)]).statistic
            if w.std() > 0
            else 0.0
        )
        out["methods"][name] = {
            **(evidence_metrics.span_metrics(row, sentences, w) or {}),
            "lds": float(lds),
            **{f"drop@{k}": float(y[0] - y[where[name, k]]) for k in ks},
            "seconds": seconds[name],
            "w": w.tolist(),
        }
    return out


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Mean of each metric per dataset and method; "ALL" averages datasets."""
    values: dict[tuple[str, str], dict[str, list[float]]] = (
        collections.defaultdict(lambda: collections.defaultdict(list))
    )
    for row in rows:
        for name, m in row["methods"].items():
            for k in METRICS:
                if k in m and m[k] == m[k]:  # Skips NaN.
                    values[(row["source"], name)][k].append(m[k])
    out: dict[str, dict[str, Any]] = collections.defaultdict(dict)
    for (source, name), ms in values.items():
        out[source][name] = {k: float(np.mean(v)) for k, v in ms.items()}
        out[source][name]["rows"] = len(ms["seconds"])
    names = {n for d in out.values() for n in d}
    out["ALL"] = {
        n: {
            k: float(
                np.mean(
                    [out[s][n][k] for s in out if s != "ALL" and k in out[s][n]]
                )
            )
            for k in METRICS
        }
        for n in names
    }
    return dict(out)


def table(summary: dict[str, Any]) -> str:
    """The summary as a fixed-width table."""
    lines = [
        f"{'dataset':12s} {'method':18s} "
        + " ".join(f"{k:>8s}" for k in METRICS)
    ]
    for source in sorted(summary, key=lambda s: (s == "ALL", s)):
        for name in ("at2", "average_attention", "contextcite"):
            if name in summary[source]:
                m = summary[source][name]
                cells = " ".join(
                    f"{m.get(k, float('nan')):8.3f}" for k in METRICS
                )
                lines.append(f"{source:12s} {name:18s} {cells}")
    return "\n".join(lines)


def main() -> None:
    """Scores every method on the rows and writes the JSON."""
    cfg, _ = config.from_command_line(config.CitationConfig, __doc__)
    at2_dir = pathlib.Path(cfg.at2)
    at2_cfg: config.AT2Config = config.resolve(
        config.load(config.AT2Config, at2_dir / "config.yaml")
    )  # type: ignore[assignment]
    name = f"attribution_{cfg.split}" + (f"_{cfg.part}" if cfg.part else "")
    logger.add(at2_dir / f"{name}.log")
    model = encoder.MaskedEncoder.from_run(
        at2_cfg.encoder.run, attention=at2_cfg.encoder.attention
    )
    theta = torch.load(at2_dir / "theta.pt").to(model.device)
    rows = sentences_lib.rows_with_evidence(
        storage.load_examples(cfg.source, cfg.split),
        cfg.per,
        cfg.seed,
        cfg.part,
    )
    logger.info(
        "{} rows: {}",
        len(rows),
        dict(collections.Counter(r.source for r in rows)),
    )
    fns = methods(model, theta)
    results = []
    for i, row in enumerate(rows, 1):
        results.append(evaluate_row(model, row, fns))
        if i % 25 == 0 or i == len(rows):
            logger.info("{}/{} rows", i, len(rows))
    summary = summarize(results)
    (at2_dir / f"{name}.json").write_text(
        json.dumps({"summary": summary, "rows": results})
    )
    logger.info("\n{}", table(summary))


if __name__ == "__main__":
    main()
