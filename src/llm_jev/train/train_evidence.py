"""Trains and evaluates the evidence head (modeling.evidence) on a frozen encoder run.

Usage:
  CUDA_VISIBLE_DEVICES=3 .venv/bin/python -m llm_jev.train.train_evidence \
      --config configs/evidence/ultrabert_380m.yaml [data.labels=gold out=runs/x ...]

Training rows are the training split's rows with evidence labels: gold spans,
plus (data.labels=gold+generated) the generated best-of-N spans where a row has
no gold. The head is conditioned on the gold option's mask (the spans support
the gold answer), trained with token-level binary cross-entropy over the
document tokens; data.val_frac of the rows pick the best epoch (token AP).

Evaluation uses attribution_mlm.select_rows (the rows AT2 and ContextCite were
scored on), conditioned on the model's predicted answer as they are:
- tokens: average precision of the token probabilities against gold tokens,
  precision of the top 10, and the gold-token share (chance AP);
- sentences: each sentence's weight is its highest token probability, scored
  with context_cite.span_metrics (ap, top1, precision / recall of the top 3);
- seconds per row (the frozen forward plus the head).
Then one global word threshold is tuned (best word F1) on eval rows with gold spans that
are not the report rows, and word precision / recall / F1 at it are reported on the report
rows. Writes <out>/config.yaml, head.pt, metrics.json, threshold.json and train_evidence.log.
"""

import argparse
import collections
import json
import math
import pathlib
import random
import time

from loguru import logger
import numpy as np
from omegaconf import OmegaConf
from sklearn.metrics import average_precision_score
import torch

from llm_jev.config import evidence_config
from llm_jev.dataloader import schema
from llm_jev.dataloader import storage
from llm_jev.evaluation import attribution_mlm
from llm_jev.evaluation import evaluate_mlm
from llm_jev.modeling import context_cite
from llm_jev.modeling import encoder_at2
from llm_jev.modeling import evidence


def generated_spans(cfg: evidence_config.Config) -> dict[str, list[schema.Span]]:
    """Row id -> the generated evidence spans (best-of-N, the release's spans file)."""
    from huggingface_hub import hf_hub_download
    import pandas as pd

    d = pd.read_parquet(hf_hub_download(cfg.data.generated_repo, cfg.data.generated_file, repo_type="dataset"))
    return {i: [schema.Span(int(s["part"]), int(s["start"]), int(s["end"])) for s in spans]
            for i, spans in zip(d["id"], d["spans"]) if len(spans)}


def labelled_rows(cfg: evidence_config.Config) -> list[tuple]:
    """(example, spans, "gold" | "generated") for every training row with labels."""
    gen = {} if cfg.data.labels == "gold" else generated_spans(cfg)
    if cfg.data.labels not in ("gold", "gold+generated"):
        raise ValueError(f"data.labels is gold or gold+generated, not {cfg.data.labels!r}")
    out = []
    for r in storage.load_examples(cfg.data.source, cfg.data.split):
        if cfg.data.datasets and r.source not in cfg.data.datasets:
            continue
        if r.has_spans:
            out.append((r, list(r.spans), "gold"))
        elif r.id in gen:
            out.append((r, gen[r.id], "generated"))
    return out


def token_ap(labels: np.ndarray, scores: np.ndarray) -> float:
    """Average precision; NaN when the row has no gold token."""
    return float(average_precision_score(labels, scores)) if labels.any() else float("nan")


def run_val(scorer, head, tok, fmt, prepared: list[dict]) -> float:
    """Mean token AP over held-out training rows (conditioned on the gold option)."""
    head.eval()
    aps = []
    with torch.no_grad():
        for p in prepared:
            row = evidence.encode_row(scorer, tok, fmt, p["example"])
            h = row["hidden"]
            logits = head(h[p["idx"]], h[row["ex"]["masks"][p["example"].gold]])
            aps.append(token_ap(p["labels"].numpy(), logits.float().cpu().numpy()))
    head.train()
    return float(np.nanmean(aps))


def prepare(rows: list[tuple], tok, fmt) -> list[dict]:
    """Document token indices and labels per row (CPU, once)."""
    out = []
    for r, spans, kind in rows:
        idx, _, labels = evidence.document_labels(r, tok, fmt, spans)
        if idx and labels.sum() > 0:
            out.append({"example": r, "idx": idx, "labels": labels, "kind": kind})
    return out


def train(cfg: evidence_config.Config, scorer, tok, fmt, out: pathlib.Path) -> evidence.EvidenceHead:
    """Fits the head; returns the best epoch's head (saved as head.pt)."""
    rows = labelled_rows(cfg)
    logger.info("{} labelled rows: {}", len(rows), dict(collections.Counter((r.source, k) for r, _, k in rows)))
    data = prepare(rows, tok, fmt)
    rng = random.Random(cfg.train.seed)
    rng.shuffle(data)
    n_val = max(1, int(len(data) * cfg.data.val_frac))
    val, tr = data[:n_val], data[n_val:]
    pos = sum(float(p["labels"].sum()) for p in tr)
    tot = sum(len(p["labels"]) for p in tr)
    pos_weight = cfg.train.pos_weight or (tot - pos) / pos
    logger.info("{} train / {} val rows; {:.0f} of {:.0f} training tokens are evidence ({:.1%}); pos_weight {:.2f}",
                len(tr), len(val), pos, tot, pos / tot, pos_weight)
    torch.manual_seed(cfg.train.seed)
    head = evidence.EvidenceHead(scorer.model.config.hidden_size, cfg.head.width, cfg.head.dropout).cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    steps = cfg.train.epochs * math.ceil(len(tr) / cfg.train.rows_per_step)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device="cuda"))
    best, best_ap = None, -1.0
    logger.info("val token AP before training: {:.4f}", run_val(scorer, head, tok, fmt, val))
    for epoch in range(cfg.train.epochs):
        rng.shuffle(tr)
        t0, running = time.time(), []
        for i in range(0, len(tr), cfg.train.rows_per_step):
            opt.zero_grad(set_to_none=True)
            chunk = tr[i: i + cfg.train.rows_per_step]
            for p in chunk:
                row = evidence.encode_row(scorer, tok, fmt, p["example"])
                h = row["hidden"]
                logits = head(h[p["idx"]], h[row["ex"]["masks"][p["example"].gold]])
                loss = bce(logits, p["labels"].cuda()) / len(chunk)
                loss.backward()
                running.append(loss.item() * len(chunk))
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            sched.step()
            if (i // cfg.train.rows_per_step) % 100 == 0:
                logger.info("epoch {} step {}: loss {:.4f} ({:.1f}m)", epoch + 1, i // cfg.train.rows_per_step,
                            float(np.mean(running[-200:])), (time.time() - t0) / 60)
        ap = run_val(scorer, head, tok, fmt, val)
        logger.info("epoch {}: val token AP {:.4f} ({:.1f}m)", epoch + 1, ap, (time.time() - t0) / 60)
        if ap > best_ap:
            best_ap, best = ap, {k: v.detach().clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best)
    torch.save(best, out / "head.pt")
    logger.info("kept the best epoch: val token AP {:.4f}", best_ap)
    return head.eval()


@torch.no_grad()
def evaluate(cfg: evidence_config.Config, scorer, head, tok, fmt) -> dict:
    """Token and sentence metrics on attribution_mlm's rows, per dataset and ALL (dataset mean)."""
    rows = attribution_mlm.select_rows(storage.load_examples(cfg.data.source, cfg.data.eval_split),
                                       cfg.data.eval_per, cfg.data.eval_seed)
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        res = evidence.predict(scorer, head, tok, fmt, r)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - t0
        p = np.array([t["p"] for t in res["tokens"]])
        gold = np.array([any(sp.part == t["part"] and t["start"] < sp.end and t["end"] > sp.start
                             for sp in r.spans) for t in res["tokens"]])
        top = np.argsort(-p)[:10]
        sources = context_cite.split_sources(r)
        _, ranges = encoder_at2.source_token_ranges(r, tok, fmt, sources)
        by_index = {t["index"]: t["p"] for t in res["tokens"]}
        w = [max((by_index.get(i, 0.0) for i in range(a, b)), default=0.0) for a, b in ranges]
        span = context_cite.span_metrics(r, {"sources": [{**s, "w": v} for s, v in zip(sources, w)]}) or {}
        m = per[r.source]
        m["token_ap"].append(token_ap(gold, p))
        m["token_p@10"].append(float(gold[top].mean()))
        m["gold_share"].append(float(gold.mean()))
        for k in ("ap", "top1", "precision", "recall"):
            m[f"sentence_{k}"].append(span.get(k, float("nan")))
        m["seconds"].append(seconds)
        m["answer_correct"].append(float(res["answer"] == r.gold))
    out = {src: {k: float(np.nanmean(v)) for k, v in ms.items()} | {"rows": len(ms["seconds"])}
           for src, ms in sorted(per.items())}
    keys = [k for k in next(iter(out.values())) if k != "rows"]
    out["ALL"] = {k: float(np.mean([out[s][k] for s in out])) for k in keys}
    return out


def words(text: str, tokens: list[dict]) -> list[tuple[int, int, float]]:
    """(start, end, score) per word of `text` (a run of non-space characters, as the UI's
    WordDoc shows it): the highest probability of the tokens inside it. Unscored words are skipped."""
    import re

    out, toks, i = [], sorted(tokens, key=lambda t: t["start"]), 0
    for m in re.finditer(r"\S+", text):
        while i < len(toks) and toks[i]["end"] <= m.start():
            i += 1
        inside, j = [], i
        while j < len(toks) and toks[j]["start"] < m.end():  # sorted: stop at the first token past the word
            inside.append(toks[j]["p"])
            j += 1
        if inside:
            out.append((m.start(), m.end(), max(inside)))
    return out


@torch.no_grad()
def word_scores(cfg, scorer, head, tok, fmt, rows: list) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Word scores, gold labels and dataset per word, for `rows` (conditioned on the predicted answer)."""
    scores, labels, sources = [], [], []
    for r in rows:
        res = evidence.predict(scorer, head, tok, fmt, r)
        for p, part in enumerate(r.parts):
            for st, en, w in words(part.text, [t for t in res["tokens"] if t["part"] == p]):
                scores.append(w)
                labels.append(any(sp.part == p and st < sp.end and en > sp.start for sp in r.spans))
                sources.append(r.source)
    return np.array(scores), np.array(labels), sources


def prf(scores: np.ndarray, labels: np.ndarray, t: float) -> tuple[float, float, float]:
    """Word precision, recall and F1 of `scores >= t` against labels."""
    sel = scores >= t
    tp = float((sel & labels).sum())
    p = tp / max(sel.sum(), 1)
    r = tp / max(labels.sum(), 1)
    return p, r, 2 * p * r / max(p + r, 1e-12)


def tune_threshold(cfg, scorer, head, tok, fmt) -> dict:
    """One global word threshold: best word F1 on eval rows with gold spans that are NOT
    the report rows (attribution_mlm.select_rows), then P / R / F1 at it on the report rows."""
    examples = storage.load_examples(cfg.data.source, cfg.data.eval_split)
    report = attribution_mlm.select_rows(examples, cfg.data.eval_per, cfg.data.eval_seed)
    held = [r for r in examples if r.has_spans and r.id not in {x.id for x in report}]
    s, y, _ = word_scores(cfg, scorer, head, tok, fmt, held)
    grid = np.unique(np.quantile(s, np.linspace(0, 1, 401)))
    f1 = [prf(s, y, t)[2] for t in grid]
    t = float(grid[int(np.argmax(f1))])
    logger.info("threshold tuned on {} held-out rows ({} words, {:.1%} gold): {:.4f} (word F1 {:.3f})",
                len(held), len(s), y.mean(), t, max(f1))
    s, y, src = word_scores(cfg, scorer, head, tok, fmt, report)
    out = {"threshold": t, "tuning_rows": len(held), "tuning_f1": float(max(f1)), "report": {}}
    for name in sorted(set(src)) + ["ALL"]:
        m = np.ones(len(s), bool) if name == "ALL" else np.array([x == name for x in src])
        p, r, f = prf(s[m], y[m], t)
        out["report"][name] = {"precision": p, "recall": r, "f1": f, "selected_share": float((s[m] >= t).mean()),
                               "gold_share": float(y[m].mean())}
        logger.info("  {:12s} words selected {:5.1%} (gold {:5.1%}): precision {:.3f} recall {:.3f} F1 {:.3f}",
                    name, out["report"][name]["selected_share"], out["report"][name]["gold_share"], p, r, f)
    return out


def main() -> None:
    """Trains the head (unless <out>/head.pt exists) and evaluates it."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="YAML (config.evidence_config)")
    parser.add_argument("overrides", nargs="*", help="dotlist, e.g. data.labels=gold")
    args = parser.parse_args()
    raw = evidence_config.load(args.config, args.overrides)
    cfg = evidence_config.resolve(raw)
    out = pathlib.Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    logger.add(out / "train_evidence.log")
    OmegaConf.save(raw, out / "config.yaml")
    scorer, tok, fmt = evaluate_mlm.load_run(cfg.encoder.run, attention=cfg.encoder.attention)
    for p in scorer.parameters():
        p.requires_grad_(False)
    if (out / "head.pt").exists():
        logger.info("using existing {} (delete it to retrain)", out / "head.pt")
        head = evidence.EvidenceHead(scorer.model.config.hidden_size, cfg.head.width, cfg.head.dropout).cuda()
        head.load_state_dict(torch.load(out / "head.pt"))
        head.eval()
    else:
        head = train(cfg, scorer, tok, fmt, out)
    metrics = evaluate(cfg, scorer, head, tok, fmt)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "threshold.json").write_text(json.dumps(tune_threshold(cfg, scorer, head, tok, fmt), indent=2))
    cols = ["gold_share", "token_ap", "token_p@10", "sentence_ap", "sentence_top1", "sentence_precision",
            "sentence_recall", "seconds", "answer_correct"]
    logger.info("\n{:12s} ".format("dataset") + " ".join(f"{c:>18s}" for c in cols) + "\n" + "\n".join(
        f"{s:12s} " + " ".join(f"{m[c]:18.3f}" for c in cols) for s, m in metrics.items()))


if __name__ == "__main__":
    main()
