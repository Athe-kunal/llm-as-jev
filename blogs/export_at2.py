"""Exports AT2 (src/modeling/at2.py) on the ContextCite blog row, as JSON.

Usage (from the repo root; needs a GPU):
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src .venv/bin/python blogs/export_at2.py

The row is the one export_contextcite.py works through. Computed with
modeling.at2: attention_weights, source_token_ranges, source_features,
example_data. If a trained theta exists (runs/at2_qwen3_0.6b/theta.pt), its
scores are exported too. Writes blogs/data/at2.json (real dataset rows:
git-ignored, local only).
"""

import json
import pathlib
import time

import numpy as np
import torch
import transformers
from loguru import logger

from dataloader import answers, data
from modeling import at2, context_cite as cc, modeling

_HERE = pathlib.Path(__file__).parent
_MODEL = "Qwen/Qwen3-0.6B"
_ADAPTER = "runs/qwen3_0.6b_lam0.01/adapter"
_THETA = pathlib.Path("runs/at2_qwen3_0.6b/theta.pt")
_DATA = pathlib.Path("runs/at2_qwen3_0.6b/data.npz")
_EVAL = pathlib.Path("runs/at2_qwen3_0.6b/eval_attribution.json")


def main() -> None:
    """Writes data/at2.json."""
    tok = transformers.AutoTokenizer.from_pretrained(_MODEL)
    model = modeling.load_trained(_MODEL, _ADAPTER).merge_and_unload()
    row = next(r for r in data.load_rows(split="eval") if r["source"] == "contractnli" and r["spans"])
    prompt, answer, _ = cc.predict(model, tok, row)
    suffix = answers._candidates(tok, len(row["labels"]))[0][answer]
    queries = list(range(len(prompt) - 1, len(prompt) - 1 + len(suffix)))
    sources = cc.split_sources(row)
    ranges = at2.source_token_ranges(row, tok, sources)
    attn = at2.attention_weights(model, prompt + suffix, queries)              # [L, H, Q, N]
    phi = at2.source_features(attn, ranges)                                    # [S, L*H]
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(5):                                                         # inference cost of AT2
        at2.source_features(at2.attention_weights(model, prompt + suffix, queries), ranges)
    torch.cuda.synchronize(); seconds = (time.time() - t0) / 5
    L, H = attn.shape[:2]
    x, y = at2.example_data(model, tok, row)                                   # [32, L*H], [32]
    uniform = phi.mean(1)

    def pearson(a, b):
        return float(np.corrcoef(a, b)[0, 1])

    out = {"model": _MODEL, "seconds": seconds, "layers": L, "heads": H, "tokens": len(prompt) + len(suffix),
           "queries": queries, "suffix": [tok.decode([t]) for t in suffix],
           "ranges": ranges, "gold": sorted(cc.gold_sentences(row, sources)),
           "phi": np.round(phi.cpu().numpy(), 4).tolist(),
           "uniform": uniform.cpu().tolist(),
           "y": y.tolist(), "predUniform": x.float().mean(1).cpu().tolist(),
           "pearsonUniform": pearson(x.float().mean(1).cpu().numpy(), y)}
    if _DATA.exists():
        d = np.load(_DATA)
        X, Y = d["features"].astype(np.float32), d["targets"]
        out["trainRows"] = int(len(Y))
        out["trainSources"] = {s: int((d["sources"] == s).sum()) for s in np.unique(d["sources"])}
        out["trainPearsonUniform"] = float(np.mean([pearson(X[i].mean(1), Y[i]) for i in range(len(Y))
                                                    if Y[i].std() > 0]))
    if _EVAL.exists():
        out["eval"] = json.loads(_EVAL.read_text())
    if _THETA.exists():
        theta = torch.load(_THETA, map_location="cpu").float()
        out["theta"] = theta.tolist()
        out["trained"] = (phi.cpu() @ theta).tolist()
        out["predTrained"] = (x.float().cpu() @ theta).tolist()
        out["pearsonTrained"] = pearson((x.float().cpu() @ theta).numpy(), y)
    path = _HERE / "data" / "at2.json"
    path.write_text(json.dumps(out))
    logger.info("wrote {} ({:.0f} KB): uniform Pearson {:.2f}", path, path.stat().st_size / 1024,
                out["pearsonUniform"])


if __name__ == "__main__":
    main()
