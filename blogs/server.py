"""Local API for the blog's "Try it" section: a trained decision model with ContextCite and AT2.

Usage (from the repo root; needs a GPU; serves 127.0.0.1 only, real rows):
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src .venv/bin/python blogs/server.py --port 8767

Endpoints (the Vite dev server proxies /api here):
  GET /api/datasets                         eval datasets, their row counts and rows with gold spans
  GET /api/sample?source=&spans=1           a random eval row id
  GET /api/row?id=                          the row, the model's tree q, guided answer, sentences
  GET /api/attribute?id=&method=at2|contextcite   one weight per sentence, top-k, span metrics
"""

import argparse
import collections
import random
import threading
import time

import fastapi
from fastapi.responses import JSONResponse
from loguru import logger
import torch
import uvicorn

from dataloader import data
from modeling import at2, context_cite as cc, modeling

_MODEL = "Qwen/Qwen3-0.6B"
_ADAPTER = "runs/qwen3_0.6b_lam0.01/adapter"
_THETA = "runs/at2_qwen3_0.6b/theta.pt"
_K = 3


def make_app() -> fastapi.FastAPI:
    """Loads the model, AT2's theta and the eval rows; returns the app."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(_MODEL)
    model = modeling.load(_MODEL, _ADAPTER)
    theta = torch.load(_THETA, map_location="cpu").float()
    rows = {r["id"]: r for r in data.load_rows(split="eval")}
    by = collections.defaultdict(list)
    for r in rows.values():
        by[r["source"]].append(r["id"])
    gpu = threading.Lock()  # one model, one request at a time
    app = fastapi.FastAPI()

    def get(row_id: str) -> dict:
        if row_id not in rows:
            raise fastapi.HTTPException(404, f"unknown row {row_id}")
        return rows[row_id]

    @app.get("/api/datasets")
    def datasets():
        return [{"id": s, "rows": len(ids), "withSpans": sum(bool(rows[i]["spans"]) for i in ids)}
                for s, ids in sorted(by.items())]

    @app.get("/api/sample")
    def sample(source: str = "", spans: int = 0):
        pool = [i for s, ids in by.items() if not source or s == source for i in ids
                if not spans or rows[i]["spans"]]
        if not pool:
            raise fastapi.HTTPException(404, "no rows match")
        return {"id": random.choice(pool)}

    @app.get("/api/row")
    def row(id: str):
        r = get(id)
        t0 = time.time()
        with gpu:
            _, answer, s = cc.predict(model, tok, r)
        sources = cc.split_sources(r)
        gold = cc.gold_sentences(r, sources)
        options = list(r["labels"])
        return {
            "id": id, "source": r["source"], "question": r["question"],
            "options": [r["labels"][k] for k in options], "gold": options.index(r["label"]),
            "answer": answer, "q": torch.softmax(s.float(), 0).tolist(),
            "parts": [p["name"] for p in r["parts"]], "hasSpans": bool(r["spans"]),
            "sentences": [{"part": x["part"], "sep": x["sep"], "gold": j in gold,
                           "text": r["parts"][x["part"]]["text"][x["start"]:x["end"]]}
                          for j, x in enumerate(sources)],
            "seconds": time.time() - t0,
        }

    @app.get("/api/attribute")
    def attribute(id: str, method: str = "at2"):
        r = get(id)
        if method not in ("at2", "contextcite"):
            raise fastapi.HTTPException(400, "method is at2 or contextcite")
        if len(cc.split_sources(r)) < 2:
            return {"method": method, "w": None, "reason": "fewer than two sentences: nothing to attribute"}
        t0 = time.time()
        with gpu:
            res = at2.attribute(model, tok, r, theta) if method == "at2" else cc.attribute(model, tok, r)
        seconds = time.time() - t0
        w = [x["w"] for x in res["sources"]]
        top = [j for j in sorted(range(len(w)), key=lambda j: -w[j])[:_K] if w[j] > 0]
        return {"method": method, "w": w, "top": top, "seconds": seconds,
                "metrics": cc.span_metrics(r, res, k=_K) if r["spans"] else None}

    @app.exception_handler(Exception)
    def failed(_, exc):
        logger.exception("request failed")
        return JSONResponse({"detail": f"{type(exc).__name__}: {exc}"}, status_code=500)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8767)
    args = parser.parse_args()
    uvicorn.run(make_app(), host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
