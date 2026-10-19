"""Exports ContextCite attributions for one MuSiQue row under the Jev scorer, as JSON.

Usage (from the repo root; needs a GPU):
  CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src .venv/bin/python blogs/export_contextcite.py

ContextCite (Cohen-Wang et al., NeurIPS 2024): sources are the row's 20
paragraphs; each ablation keeps every paragraph with probability 1/2 and
re-scores the answer; the target is the logit of q(gold | ablated prompt),
where q is the tree distribution from answers.score_batch; a sparse linear
surrogate (Lasso) fitted to the ablations gives one weight per paragraph.
Held-out ablations measure the surrogate (LDS), and removing / keeping the
top paragraphs gives SelfCite-style necessity and sufficiency. Writes
blogs/data/contextcite.json (real dataset rows: git-ignored, local only).
"""

import json
import pathlib
import random

import numpy as np
import torch
import transformers
from loguru import logger

from dataloader import answers, data
from modeling import modeling

_HERE = pathlib.Path(__file__).parent
_MODEL = "Qwen/Qwen3-0.6B"
_ADAPTER = "runs/qwen3_0.6b_lam0.01/adapter"
_SEED = 0
_TRAIN, _TEST = 64, 32  # ablations for the fit and for checking it
_ALPHA = 0.01  # Lasso strength, ContextCite's default
_SEP = "\n\n"


def paragraphs(row: dict) -> list[str]:
    return row["parts"][0]["text"].split(_SEP)


def ablate(row: dict, keep: list[bool]) -> dict:
    """The row with only the kept paragraphs."""
    text = _SEP.join(p for p, k in zip(paragraphs(row), keep) if k)
    return {**row, "parts": [{**row["parts"][0], "text": text}], "spans": []}


@torch.no_grad()
def logit_q(model, tok, row: dict, masks: list[list[bool]]) -> np.ndarray:
    """logit q(gold | prompt) for every ablation mask (the ContextCite target)."""
    exs = [answers.encode(ablate(row, m), tok, None) for m in masks]
    out = []
    for i in range(0, len(exs), 16):
        batch = exs[i:i + 16]
        res = answers.score_batch(model, [e["input_ids"][: e["path"][0][0] + 1] for e in batch],
                                  tok, [e["n_options"] for e in batch], "cuda")
        for e, (s, _, _) in zip(batch, res):
            lq = torch.log_softmax(s.float(), 0)[e["gold"]].item()
            out.append(lq - np.log1p(-min(np.exp(lq), 1 - 1e-6)))
    return np.array(out)


def lasso(X: np.ndarray, y: np.ndarray, alpha: float, iters: int = 2000):
    """min_w,b 1/(2n) ||y - b - X w||^2 + alpha ||w||_1, by coordinate descent."""
    n, d = X.shape
    xm, ym = X.mean(0), y.mean()
    Xc, yc = X - xm, y - ym
    w = np.zeros(d)
    sq = (Xc ** 2).sum(0) / n
    for _ in range(iters):
        for j in range(d):
            r = yc - Xc @ w + Xc[:, j] * w[j]
            rho = Xc[:, j] @ r / n
            w[j] = np.sign(rho) * max(abs(rho) - alpha, 0) / sq[j]
    return w, ym - xm @ w


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = a.argsort().argsort(), b.argsort().argsort()
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> None:
    """Writes data/contextcite.json."""
    tok = transformers.AutoTokenizer.from_pretrained(_MODEL)
    model = modeling.load_trained(_MODEL, _ADAPTER).merge_and_unload()
    rows = [r for r in data.load_rows(split="eval") if r["source"] == "musique"
            and r["spans"] and r["label"] == "yes"]
    random.Random(_SEED).shuffle(rows)
    # the first row the model gets right with some confidence
    for row in rows:
        d = len(paragraphs(row))
        full = logit_q(model, tok, row, [[True] * d])[0]
        if full > 1.0:
            break
    paras = paragraphs(row)
    gold_text = {s["text"] for s in row["spans"]}
    gold = [p in gold_text for p in paras]
    rng = np.random.default_rng(_SEED)
    masks = rng.random((_TRAIN + _TEST, d)) < 0.5
    y = logit_q(model, tok, row, masks.tolist())
    w, b = lasso(masks[:_TRAIN].astype(float), y[:_TRAIN], _ALPHA)
    pred = masks[_TRAIN:].astype(float) @ w + b
    lds = spearman(pred, y[_TRAIN:])

    def keep_all_but(S):
        return [i not in S for i in range(d)]

    def keep_only(S):
        return [i in S for i in range(d)]

    order = list(np.argsort(-w))
    sets = {f"top-{k}": order[:k] for k in (1, 2, 3)}
    sets["gold"] = [i for i in range(d) if gold[i]]
    rand = [sorted(rng.choice(d, size=len(sets["gold"]), replace=False).tolist()) for _ in range(8)]
    probe = [keep_all_but(S) for S in sets.values()] + [keep_only(S) for S in sets.values()]
    probe += [keep_all_but(S) for S in rand] + [keep_only(S) for S in rand]
    v = logit_q(model, tok, row, probe)
    n = len(sets)
    lq = lambda z: -np.log1p(np.exp(-z))  # log q from logit q
    evals = {name: {"necessity": float(lq(full) - lq(v[i])), "sufficiency": float(lq(v[n + i]) - lq(full)),
                    "set": [int(j) for j in S]} for i, (name, S) in enumerate(sets.items())}
    r0 = 2 * n
    evals["random"] = {"necessity": float(np.mean(lq(full) - lq(v[r0:r0 + 8]))),
                       "sufficiency": float(np.mean(lq(v[r0 + 8:r0 + 16]) - lq(full))), "set": []}
    logger.info("{}: full logit q {:.2f}, LDS {:.2f}, top weights {}, gold {}", row["id"], full, lds,
                [(int(i), round(float(w[i]), 2)) for i in order[:4]], sets["gold"])
    for k, e in evals.items():
        logger.info("  {}: necessity {:.2f}, sufficiency {:.2f}", k, e["necessity"], e["sufficiency"])

    payload = {
        "model": _MODEL, "adapter": _ADAPTER, "alpha": _ALPHA, "train": _TRAIN, "test": _TEST,
        "row": {"id": row["id"], "question": row["question"], "label": row["label"],
                "options": row["labels"]},
        "paragraphs": [{"title": p.split("\n", 1)[0], "text": p.split("\n", 1)[-1], "gold": g,
                        "tokens": len(tok(p, add_special_tokens=False)["input_ids"])}
                       for p, g in zip(paras, gold)],
        "full": float(full), "masks": masks.astype(int).tolist(), "y": y.tolist(),
        "w": w.tolist(), "b": float(b), "pred": pred.tolist(), "lds": lds, "evals": evals,
    }
    out = _HERE / "data" / "contextcite.json"
    out.write_text(json.dumps(payload, ensure_ascii=False))
    logger.info("wrote {} ({:.0f} KB)", out, out.stat().st_size / 1024)


if __name__ == "__main__":
    main()
