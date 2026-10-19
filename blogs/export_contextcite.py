"""Exports ContextCite (src/modeling/context_cite.py) on one row, plus an eval table, as JSON.

Usage (from the repo root; needs a GPU):
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src .venv/bin/python blogs/export_contextcite.py

Everything is computed with the functions in modeling/context_cite.py:
split_sources, ablate, ablation_masks, ablation_targets, fit_surrogate,
span_metrics. The worked row is the first ContractNLI eval row with gold
spans; the table scores 20 eval rows per span dataset. Writes
blogs/data/contextcite.json (real dataset rows: git-ignored, local only).
"""

import collections
import json
import pathlib
import random
import time

import numpy as np
import torch
import transformers
from loguru import logger
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dataloader import answers, data
from modeling import context_cite as cc, modeling

_HERE = pathlib.Path(__file__).parent
_MODEL = "Qwen/Qwen3-0.6B"
_ADAPTER = "runs/qwen3_0.6b_lam0.01/adapter"
_N, _TEST, _ALPHA = 64, 32, 0.01


def spearman(a, b) -> float:
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


def main() -> None:
    """Writes data/contextcite.json."""
    tok = transformers.AutoTokenizer.from_pretrained(_MODEL)
    model = modeling.load_trained(_MODEL, _ADAPTER).merge_and_unload()
    rows = data.load_rows(split="eval")
    row = next(r for r in rows if r["source"] == "contractnli" and r["spans"])

    prompt, answer, s = cc.predict(model, tok, row)
    sources = cc.split_sources(row)
    d = len(sources)
    gold = cc.gold_sentences(row, sources)
    text = lambda j: row["parts"][sources[j]["part"]]["text"][sources[j]["start"]:sources[j]["end"]]

    t0 = time.time()
    V = cc.ablation_masks(_N, d)                       # fitting masks, seeds 0..63
    y = cc.ablation_targets(model, tok, row, sources, V, answer)
    t_abl = time.time() - t0
    Vt = cc.ablation_masks(_TEST, d, seed=_N)          # held-out masks, seeds 64..95
    yt = cc.ablation_targets(model, tok, row, sources, Vt, answer)
    w, b = cc.fit_surrogate(V, y, _ALPHA)
    # the same fit, opened up to show the standardised coefficients
    scaler, lasso = StandardScaler(), Lasso(alpha=_ALPHA, random_state=0)
    make_pipeline(scaler, lasso).fit(V.astype(np.float32), y)
    lds = spearman(Vt.astype(float) @ w + b, yt)

    # top-k drop (ContextCite's other metric): remove the k top sentences
    order = list(np.argsort(-w))
    full = cc.logit_q(s, answer)
    lq = lambda z: -np.log1p(np.exp(-z))               # log q from logit q
    rng = np.random.default_rng(0)
    drops = []
    for k in (1, 3, 5):
        keep_top = np.ones((1, d), bool); keep_top[0, order[:k]] = False
        rand = np.ones((8, d), bool)
        for i in range(8):
            rand[i, rng.choice(d, k, replace=False)] = False
        yk = cc.ablation_targets(model, tok, row, sources, np.vstack([keep_top, rand]), answer)
        drops.append({"k": k, "top": float(lq(full) - lq(yk[0])),
                      "random": float(np.mean(lq(full) - lq(yk[1:])))})

    # one ablation drawn out: which sentences the first mask keeps, and the prompt it gives
    example = cc.ablate(row, sources, V[0])

    # eval table: 20 rows per span dataset, as docs/architecture.md §7 step 5b
    by = collections.defaultdict(list)
    for r in rows:
        if r["spans"]:
            by[r["source"]].append(r)
    table = {}
    for src, rs in sorted(by.items()):
        m = collections.defaultdict(list)
        for r in random.Random(0).sample(rs, min(20, len(rs))):
            res = cc.attribute(model, tok, r)
            met = cc.span_metrics(r, res, k=3, beta=1.0)
            for key, v in met.items():
                m[key].append(v)
            m["random_ap"].append(len(cc.gold_sentences(r, res["sources"])) / len(res["sources"]))
            m["correct"].append(res["answer"] == list(r["labels"]).index(r["label"]))
            m["sentences"].append(len(res["sources"]))
        table[src] = {k: float(np.mean(v)) for k, v in m.items()}
        logger.info("{}: {}", src, {k: round(v, 2) for k, v in table[src].items()})

    options = [row["labels"][k] for k in row["labels"]]
    payload = {
        "model": _MODEL, "adapter": _ADAPTER, "alpha": _ALPHA, "n": _N, "test": _TEST,
        "row": {"id": row["id"], "question": row["question"], "options": options,
                "gold": list(row["labels"]).index(row["label"]), "answer": answer,
                "s": s.float().tolist(), "q": torch.softmax(s.float(), 0).tolist(),
                "logit": full, "partName": row["parts"][0]["name"], "chars": len(row["parts"][0]["text"])},
        "sentences": [{"text": text(j), "sep": sources[j]["sep"], "gold": j in gold, "w": float(w[j]),
                       "mu": float(scaler.mean_[j]), "sigma": float(scaler.scale_[j]),
                       "beta": float(lasso.coef_[j])} for j in range(d)],
        "bias": float(b), "intercept": float(lasso.intercept_),
        "masks": V.astype(int).tolist(), "y": y.tolist(), "yt": yt.tolist(),
        "predt": (Vt.astype(float) @ w + b).tolist(), "lds": lds, "drops": drops,
        "example": {"mask": V[0].astype(int).tolist(), "y": float(y[0]),
                    "chars": len(example["parts"][0]["text"])},
        "seconds": t_abl, "table": table,
    }
    out = _HERE / "data" / "contextcite.json"
    out.write_text(json.dumps(payload, ensure_ascii=False))
    logger.info("wrote {} ({:.0f} KB): LDS {:.2f}, top-k drops {}", out, out.stat().st_size / 1024, lds, drops)


if __name__ == "__main__":
    main()
