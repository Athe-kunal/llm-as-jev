"""Local API for the blog's "Try it" section: an encoder decision model on held-out rows.

Usage (from the repo root; needs a GPU and `uv sync --extra blog`; serves 127.0.0.1 only):
  CUDA_VISIBLE_DEVICES=2 .venv/bin/python blogs/server.py --run runs/ultrabert_380m_lr1.5e-5

Endpoints (the Vite dev server proxies /api here):
  GET /api/info                    the loaded run, its attention, and whether it can cite
  GET /api/datasets                the held-out datasets, their rows, and rows with gold evidence
  GET /api/sample?source=&spans=1  a random row id (spans=1: only rows with gold evidence)
  GET /api/row?id=                 the row, q over its options, the answer, its sentences
                                   (gold evidence marked) and the time to score it

The model is evaluation.evaluate_mlm.load_run: the run's own config and
prompt format, fp32 weights under bf16 autocast, so the page shows the
numbers evaluation computes. ContextCite for the encoder is not wired yet
(/api/info says cite: false).
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

from llm_jev.dataloader import mlm
from llm_jev.dataloader import storage
from llm_jev.evaluation import evaluate_mlm
from llm_jev.modeling import context_cite
from llm_jev.modeling import encoder


def make_app(run: str, data: str, split: str, attention: str | None) -> fastapi.FastAPI:
    """Loads the run and the rows; returns the app."""
    scorer, tok, fmt = evaluate_mlm.load_run(run, attention=attention)
    cfg = evaluate_mlm.run_config(run)
    rows = {r.id: r for r in storage.load_examples(data, split)}
    by = collections.defaultdict(list)
    for r in rows.values():
        by[r.source].append(r.id)
    info = {"run": run, "model": cfg.model.id, "kind": "encoder",
            "attention": attention or cfg.model.attention, "format": vars(fmt), "cite": False}
    logger.info("{} rows from {} ({}); {}", len(rows), data, split, info)
    gpu = threading.Lock()  # one model, one request at a time
    app = fastapi.FastAPI()

    def get(row_id: str):
        if row_id not in rows:
            raise fastapi.HTTPException(404, f"unknown row {row_id}")
        return rows[row_id]

    @app.get("/api/info")
    def get_info():
        return info

    @app.get("/api/datasets")
    def datasets():
        return [{"id": s, "rows": len(ids), "withSpans": sum(rows[i].has_spans for i in ids)}
                for s, ids in sorted(by.items())]

    @app.get("/api/sample")
    def sample(source: str = "", spans: int = 0):
        pool = [i for s, ids in by.items() if not source or s == source for i in ids
                if not spans or rows[i].has_spans]
        if not pool:
            raise fastapi.HTTPException(404, "no rows match")
        return {"id": random.choice(pool)}

    @app.get("/api/row")
    def row(id: str):
        r = get(id)
        ex = mlm.encode(r, tok, fmt)  # options in release order: q[k] is option k
        with gpu, torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            s = encoder.score(scorer, ex, "cuda")
            torch.cuda.synchronize()
            seconds = time.perf_counter() - t0
        sources = context_cite.split_sources(r)
        gold = context_cite.gold_sentences(r, sources)
        return {
            "id": id, "source": r.source, "question": r.question,
            "model": f"{cfg.model.id} ({run})", "options": list(r.options), "gold": r.gold,
            "answer": int(s.argmax()), "q": torch.softmax(s, 0).tolist(),
            "parts": [p.name for p in r.parts], "hasSpans": r.has_spans,
            "partTexts": [p.text for p in r.parts], "tokens": len(ex["input_ids"]),
            "sentences": [{"part": x["part"], "start": x["start"], "end": x["end"], "sep": x["sep"],
                           "gold": j in gold, "text": r.parts[x["part"]].text[x["start"]:x["end"]]}
                          for j, x in enumerate(sources)],
            "seconds": seconds,
        }

    @app.exception_handler(Exception)
    def failed(_, exc):
        logger.exception("request failed")
        return JSONResponse({"detail": f"{type(exc).__name__}: {exc}"}, status_code=500)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", default="runs/ultrabert_380m_lr1.5e-5", help="train_mlm run directory")
    parser.add_argument("--attention", help="override the run's model.attention (e.g. varlen, sdpa)")
    parser.add_argument("--data", default="data/splits/reason_long_nimble")
    parser.add_argument("--split", default="eval")
    parser.add_argument("--port", type=int, default=8767)
    args = parser.parse_args()
    uvicorn.run(make_app(args.run, args.data, args.split, args.attention),
                host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
