"""Local API for the blog's "Try it" section: a trained decision model with one-pass ContextCite.

Usage (from the repo root; needs a GPU; serves 127.0.0.1 only, real rows):
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src .venv/bin/python blogs/server.py --port 8767

Endpoints (the Vite dev server proxies /api here):
  GET /api/datasets                         eval datasets, their row counts and rows with gold spans
  GET /api/sample?source=&spans=1           a random eval row id
  GET /api/row?id=                          the row, the model's tree q, guided answer, sentences
  GET /api/attribute?id=                    ContextCite: one weight per sentence, top-k, span metrics

ContextCite is masked_cite.attribute: the decision and its 64 ablations from
one packed forward (dropped sentences hidden from the reader, not deleted),
on a flex-attention copy of the model compiled at startup. The answer shown
by /api/row comes from the usual SDPA scorer (answers.score).
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
from modeling import context_cite as cc, masked_cite, modeling

_MODEL = "Qwen/Qwen3-0.6B"
_ADAPTER = "runs/qwen3_0.6b_lam0.01/adapter"
_K = 3


def make_app() -> fastapi.FastAPI:
    """Loads the model twice (SDPA for the answer, compiled flex attention
    for ContextCite) and the eval rows; returns the app."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(_MODEL)
    model = modeling.load(_MODEL, _ADAPTER)
    flex = modeling.load(_MODEL, _ADAPTER)
    flex.set_attn_implementation("flex_attention")
    flex = torch.compile(flex, dynamic=True)
    rows = {r["id"]: r for r in data.load_rows(split="eval")}
    t0 = time.time()  # compile now, not on the first request
    masked_cite.attribute(flex, tok, next(r for r in rows.values() if r["source"] == "contractnli"))
    logger.info("compiled flex attention in {:.0f} s", time.time() - t0)
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
            "partTexts": [p["text"] for p in r["parts"]],
            "sentences": [{"part": x["part"], "start": x["start"], "end": x["end"], "sep": x["sep"],
                           "gold": j in gold, "text": r["parts"][x["part"]]["text"][x["start"]:x["end"]]}
                          for j, x in enumerate(sources)],
            "seconds": time.time() - t0,
        }

    @app.get("/api/attribute")
    def attribute(id: str):
        r = get(id)
        if len(cc.split_sources(r)) < 2:
            return {"w": None, "reason": "fewer than two sentences: nothing to attribute"}
        t0 = time.time()
        with gpu:
            res = masked_cite.attribute(flex, tok, r)
        seconds = time.time() - t0
        w = [x["w"] for x in res["sources"]]
        top = [j for j in sorted(range(len(w)), key=lambda j: -w[j])[:_K] if w[j] > 0]
        return {"w": w, "top": top, "seconds": seconds, "answer": res["answer"],
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
