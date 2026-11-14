"""Trains and evaluates the evidence head on a frozen encoder run.

Usage:
  llm-jev-train-evidence --config configs/evidence/ultrabert_380m.yaml \
      [data.labels=gold out=runs/x ...]

Training rows are the training split's rows with evidence labels: gold
spans, plus (data.labels=gold+generated) the generated best-of-N spans where
a row has no gold. The head is conditioned on the gold option's mask and
trained with token-level binary cross-entropy over the document tokens;
data.val_frac of the rows pick the best epoch (token AP).

Evaluation (on `sentences.rows_with_evidence`, the rows the citation
baselines are scored on) is conditioned on the model's predicted answer:
- tokens: average precision against gold tokens, precision of the top 10,
  and the gold-token share (chance AP);
- sentences: a sentence's weight is its highest token probability, scored
  with `metrics.span_metrics`;
- seconds per row (the frozen forward plus the head).
Then one global word threshold is tuned (best word F1) on held-out rows that
are not report rows, and word precision / recall / F1 at it are reported.
Writes <out>/config.yaml, head.pt, metrics.json, threshold.json.
"""

import collections
from collections.abc import Sequence
import dataclasses
import json
import math
import pathlib
import random
import time
from typing import Any

from loguru import logger
import numpy as np
import torch

from llm_jev import config
from llm_jev.data import schema
from llm_jev.data import storage
from llm_jev.evidence import head as head_lib
from llm_jev.evidence import metrics
from llm_jev.evidence import sentences
from llm_jev.models import encoder


@dataclasses.dataclass(frozen=True)
class LabelledRow:
    """A training row with its evidence labels.

    Attributes:
      example: The row.
      tokens: Its document tokens, labelled against the evidence.
      kind: "gold" or "generated".
    """

    example: schema.Example
    tokens: head_lib.DocumentTokens
    kind: str


def generated_spans(
    cfg: config.EvidenceDataConfig,
) -> dict[str, list[schema.Span]]:
    """Row id -> generated evidence spans (the release's spans file)."""
    from huggingface_hub import hf_hub_download
    import pandas as pd

    path = hf_hub_download(
        cfg.generated_repo, cfg.generated_file, repo_type="dataset"
    )
    frame = pd.read_parquet(path)
    return {
        row_id: [
            schema.Span(int(s["part"]), int(s["start"]), int(s["end"]))
            for s in spans
        ]
        for row_id, spans in zip(frame["id"], frame["spans"], strict=True)
        if len(spans)
    }


def labelled_rows(
    cfg: config.EvidenceDataConfig, model: encoder.MaskedEncoder
) -> list[LabelledRow]:
    """Every training row with evidence, tokenised and labelled.

    Raises:
      ValueError: On an unknown label choice.
    """
    if cfg.labels not in ("gold", "gold+generated"):
        raise ValueError(
            f"data.labels is gold or gold+generated, not {cfg.labels!r}"
        )
    generated = {} if cfg.labels == "gold" else generated_spans(cfg)
    out = []
    for row in storage.load_examples(cfg.source, cfg.split):
        if cfg.datasets and row.source not in cfg.datasets:
            continue
        if row.has_spans:
            spans, kind = list(row.spans), "gold"
        elif row.id in generated:
            spans, kind = generated[row.id], "generated"
        else:
            continue
        tokens = head_lib.document_tokens(model, row, spans)
        if tokens.index and tokens.labels.sum() > 0:
            out.append(LabelledRow(row, tokens, kind))
    return out


def _gold_logits(
    model: encoder.MaskedEncoder, head: head_lib.EvidenceHead, row: LabelledRow
) -> torch.Tensor:
    """Head logits [T] for the gold option (the frozen forward has no grad)."""
    ex, hidden, _ = model.forward_row(row.example)
    return head(hidden[row.tokens.index], hidden[ex.masks[ex.gold]])


def validation_ap(
    model: encoder.MaskedEncoder,
    head: head_lib.EvidenceHead,
    rows: Sequence[LabelledRow],
) -> float:
    """Mean token AP over held-out training rows."""
    head.eval()
    with torch.no_grad():
        aps = [
            metrics.average_precision(
                r.tokens.labels.numpy(),
                _gold_logits(model, head, r).float().cpu().numpy(),
            )
            for r in rows
        ]
    head.train()
    return float(np.nanmean(aps))


def train(
    cfg: config.EvidenceConfig, model: encoder.MaskedEncoder, out: pathlib.Path
) -> head_lib.EvidenceHead:
    """Fits the head and saves the best epoch's weights as head.pt."""
    rows = labelled_rows(cfg.data, model)
    logger.info(
        "{} labelled rows: {}",
        len(rows),
        dict(collections.Counter((r.example.source, r.kind) for r in rows)),
    )
    rng = random.Random(cfg.train.seed)
    rng.shuffle(rows)
    n_val = max(1, int(len(rows) * cfg.data.val_frac))
    val, fit_rows = rows[:n_val], rows[n_val:]
    positives = sum(float(r.tokens.labels.sum()) for r in fit_rows)
    total = sum(len(r.tokens.labels) for r in fit_rows)
    pos_weight = cfg.train.pos_weight or (total - positives) / positives
    logger.info(
        "{} train / {} val rows; {:.1%} of training tokens are evidence; "
        "pos_weight {:.2f}",
        len(fit_rows),
        len(val),
        positives / total,
        pos_weight,
    )
    torch.manual_seed(cfg.train.seed)
    hidden = model.unwrapped.model.config.hidden_size
    head = head_lib.EvidenceHead(hidden, cfg.head.width, cfg.head.dropout).to(
        model.device
    )
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    step_rows = cfg.train.rows_per_step
    steps = cfg.train.epochs * math.ceil(len(fit_rows) / step_rows)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps
    )
    bce = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, device=model.device)
    )
    best: dict[str, torch.Tensor] = {}
    best_ap = -1.0
    logger.info(
        "val token AP before training: {:.4f}", validation_ap(model, head, val)
    )
    for epoch in range(cfg.train.epochs):
        rng.shuffle(fit_rows)
        start, running = time.time(), []
        for i in range(0, len(fit_rows), step_rows):
            optimizer.zero_grad(set_to_none=True)
            chunk = fit_rows[i : i + step_rows]
            for r in chunk:
                logits = _gold_logits(model, head, r)
                loss = bce(logits, r.tokens.labels.to(model.device)) / len(
                    chunk
                )
                loss.backward()
                running.append(loss.item() * len(chunk))
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            if (i // step_rows) % 100 == 0:
                logger.info(
                    "epoch {} step {}: loss {:.4f} ({:.1f}m)",
                    epoch + 1,
                    i // step_rows,
                    float(np.mean(running[-200:])),
                    (time.time() - start) / 60,
                )
        ap = validation_ap(model, head, val)
        logger.info("epoch {}: val token AP {:.4f}", epoch + 1, ap)
        if ap > best_ap:
            best_ap = ap
            best = {k: v.detach().clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best)
    torch.save(best, out / "head.pt")
    logger.info("kept the best epoch: val token AP {:.4f}", best_ap)
    return head.eval()


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def evaluate(
    cfg: config.EvidenceConfig,
    model: encoder.MaskedEncoder,
    head: head_lib.EvidenceHead,
) -> dict[str, dict[str, float]]:
    """Token and sentence metrics per dataset, and their mean ("ALL")."""
    data = cfg.data
    rows = sentences.rows_with_evidence(
        storage.load_examples(data.source, data.eval_split),
        data.eval_per,
        data.eval_seed,
        data.eval_part,
    )
    per: dict[str, dict[str, list[float]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    for r in rows:
        _sync(model.device)
        start = time.perf_counter()
        pred = head_lib.predict(model, head, r)
        _sync(model.device)
        seconds = time.perf_counter() - start
        p = pred.probabilities.numpy()
        gold = np.array(
            [
                any(
                    s.part == part and a < s.end and b > s.start
                    for s in r.spans
                )
                for part, a, b in pred.tokens.where
            ]
        )
        top = np.argsort(-p)[:10]
        sents = sentences.split(r)
        _, ranges = encoder.token_ranges(
            r, model.tokenizer, model.fmt, [s.char_range for s in sents]
        )
        by_index = dict(zip(pred.tokens.index, p.tolist(), strict=True))
        weights = [
            max((by_index.get(i, 0.0) for i in range(a, b)), default=0.0)
            for a, b in ranges
        ]
        span = metrics.span_metrics(r, sents, weights) or {}
        m = per[r.source]
        m["token_ap"].append(metrics.average_precision(gold, p))
        m["token_p@10"].append(float(gold[top].mean()))
        m["gold_share"].append(float(gold.mean()))
        for k in ("ap", "top1", "precision", "recall"):
            m[f"sentence_{k}"].append(span.get(k, float("nan")))
        m["seconds"].append(seconds)
        m["answer_correct"].append(float(pred.answer == r.gold))
    out = {
        source: {k: float(np.nanmean(v)) for k, v in ms.items()}
        | {"rows": len(ms["seconds"])}
        for source, ms in sorted(per.items())
    }
    keys = [k for k in next(iter(out.values())) if k != "rows"]
    out["ALL"] = {k: float(np.mean([out[s][k] for s in out])) for k in keys}
    return out


@torch.no_grad()
def _word_scores(
    model: encoder.MaskedEncoder,
    head: head_lib.EvidenceHead,
    rows: Sequence[schema.Example],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Word scores, gold labels and dataset per word (predicted answer)."""
    scores, labels, sources = [], [], []
    for r in rows:
        pred = head_lib.predict(model, head, r)
        for p, part in enumerate(r.parts):
            tokens = [
                (a, b, float(v))
                for (q, a, b), v in zip(
                    pred.tokens.where, pred.probabilities, strict=True
                )
                if q == p
            ]
            for a, b, w in metrics.word_scores(part.text, tokens):
                scores.append(w)
                labels.append(
                    any(
                        s.part == p and a < s.end and b > s.start
                        for s in r.spans
                    )
                )
                sources.append(r.source)
    return np.array(scores), np.array(labels), sources


def tune_threshold(
    cfg: config.EvidenceConfig,
    model: encoder.MaskedEncoder,
    head: head_lib.EvidenceHead,
) -> dict[str, Any]:
    """Tunes one word threshold on non-report rows; reports P / R / F1.

    Raises:
      AssertionError: If tuning and report rows overlap.
    """
    data = cfg.data
    examples = storage.load_examples(data.source, data.eval_split)
    report = sentences.rows_with_evidence(
        examples, data.eval_per, data.eval_seed, data.eval_part
    )
    report_ids = {r.id for r in report}
    if data.tune_part is not None:
        held = sentences.rows_with_evidence(
            examples, 0, data.eval_seed, data.tune_part
        )
        assert not {r.id for r in held} & report_ids, (
            "tuning rows overlap reports"
        )
    else:
        held = [r for r in examples if r.has_spans and r.id not in report_ids]
    s, y, _ = _word_scores(model, head, held)
    grid = np.unique(np.quantile(s, np.linspace(0, 1, 401)))
    f1 = [metrics.precision_recall_f1(s, y, t)[2] for t in grid]
    threshold = float(grid[int(np.argmax(f1))])
    logger.info(
        "threshold tuned on {} rows ({} words): {:.4f} (word F1 {:.3f})",
        len(held),
        len(s),
        threshold,
        max(f1),
    )
    s, y, src = _word_scores(model, head, report)
    out: dict[str, Any] = {
        "threshold": threshold,
        "tuning_rows": len(held),
        "tuning_f1": float(max(f1)),
        "report": {},
    }
    for name in [*sorted(set(src)), "ALL"]:
        keep = (
            np.ones(len(s), bool)
            if name == "ALL"
            else np.array([x == name for x in src])
        )
        p, r, f = metrics.precision_recall_f1(s[keep], y[keep], threshold)
        out["report"][name] = {
            "precision": p,
            "recall": r,
            "f1": f,
            "selected_share": float((s[keep] >= threshold).mean()),
            "gold_share": float(y[keep].mean()),
        }
    return out


def main() -> None:
    """Trains the head (unless <out>/head.pt exists) and evaluates it."""
    cfg, raw = config.from_command_line(config.EvidenceConfig, __doc__)
    out = pathlib.Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    logger.add(out / "train_evidence.log")
    tag = f"_{cfg.data.eval_part}" if cfg.data.eval_part else ""
    config.save(raw, out / f"config{tag}.yaml")
    model = encoder.MaskedEncoder.from_run(
        cfg.encoder.run, attention=cfg.encoder.attention
    )
    model.unwrapped.requires_grad_(False)
    if (out / "head.pt").exists():
        logger.info("using existing {} (delete it to retrain)", out / "head.pt")
        hidden = model.unwrapped.model.config.hidden_size
        head = head_lib.EvidenceHead(hidden, cfg.head.width, cfg.head.dropout)
        head.load_state_dict(torch.load(out / "head.pt"))
        head = head.to(model.device).eval()
    else:
        head = train(cfg, model, out)
    results = evaluate(cfg, model, head)
    (out / f"metrics{tag}.json").write_text(json.dumps(results, indent=2))
    threshold = tune_threshold(cfg, model, head)
    (out / f"threshold{tag}.json").write_text(json.dumps(threshold, indent=2))
    for name, m in results.items():
        logger.info(
            "{:12s} token AP {:.3f} sentence AP {:.3f} top1 {:.3f} {:.0f} ms",
            name,
            m["token_ap"],
            m["sentence_ap"],
            m["sentence_top1"],
            1000 * m["seconds"],
        )


if __name__ == "__main__":
    main()
