"""Local API for the blog's "Try it" section: an encoder decision model on held-out rows.

Usage (from the repo root; needs a GPU and `uv sync --extra blog`; serves 127.0.0.1 only):
  CUDA_VISIBLE_DEVICES=2 .venv/bin/python blogs/server.py --run runs/ultrabert_380m_lr1.5e-5

Endpoints (the Vite dev server proxies /api here):
  GET /api/info                    the loaded run, its attention, and whether it can cite
  GET /api/datasets                the held-out datasets, their rows, and rows with gold evidence
  GET /api/sample?source=&spans=1  a random row id (spans=1: only rows with gold evidence)
  GET /api/row?id=&shuffle=&seed=  the row, q over its options, the answer, its sentences
                                   (gold evidence marked) and the time to score it; shuffle=1
                                   scores the options in a seeded random order (as in training)
                                   and returns them in that order, with `order` = each slot's
                                   original option index

  GET /api/attribute?id=           AT2 and ContextCite for the row's answer (options in release
                                   order): per-sentence weights, top 3, span metrics, seconds

The model is evaluation.evaluate_mlm.load_run: the run's own config and
prompt format, fp32 weights under bf16 autocast, so the page shows the
numbers evaluation computes. With --at2 (an AT2 run directory) the encoder is
that run's (config.yaml), on FA2 varlen as AT2's attention capture needs, and
/api/info says cite: true.
"""

import argparse
import collections
import json
import pathlib
import random
import threading
import time

import fastapi
from fastapi.responses import JSONResponse
from loguru import logger
import torch
import uvicorn

from llm_jev.config import at2_config
from llm_jev.config import evidence_config
from llm_jev.dataloader import mlm
from llm_jev.dataloader import storage
from llm_jev.evaluation import evaluate_mlm
from llm_jev.modeling import context_cite
from llm_jev.modeling import encoder
from llm_jev.modeling import encoder_at2
from llm_jev.modeling import encoder_cite
from llm_jev.modeling import evidence


def make_app(run: str, data: str, split: str, attention: str | None,
             at2: str | None = None, evidence_dir: str | None = None,
             max_share: float = 0.15) -> fastapi.FastAPI:
    """Loads the run (or the AT2 run's encoder) and the rows; returns the app."""
    theta = None
    if at2:
        acfg = at2_config.resolve(at2_config.load(pathlib.Path(at2) / "config.yaml"))
        run, attention = acfg.encoder.run, acfg.encoder.attention
        theta = torch.load(pathlib.Path(at2) / "theta.pt").cuda()
    scorer, tok, fmt = evaluate_mlm.load_run(run, attention=attention)
    cfg = evaluate_mlm.run_config(run)
    head = None
    if evidence_dir:
        ecfg = evidence_config.resolve(evidence_config.load(pathlib.Path(evidence_dir) / "config.yaml"))
        if pathlib.Path(ecfg.encoder.run).resolve() != pathlib.Path(run).resolve():
            raise ValueError(f"the evidence head was trained on {ecfg.encoder.run}, the server loaded {run}")
        head = evidence.EvidenceHead(scorer.model.config.hidden_size, ecfg.head.width, ecfg.head.dropout).cuda()
        head.load_state_dict(torch.load(pathlib.Path(evidence_dir) / "head.pt"))
        head.eval()
    # highlight rule: p >= the tuned global threshold (threshold.json), capped at the top max_share of words
    tfile = pathlib.Path(evidence_dir or "") / "threshold.json"
    threshold = json.loads(tfile.read_text())["threshold"] if evidence_dir and tfile.exists() else 0.5
    rows = {r.id: r for r in storage.load_examples(data, split)}
    by = collections.defaultdict(list)
    for r in rows.values():
        by[r.source].append(r.id)
    info = {"run": run, "model": cfg.model.id, "kind": "encoder",
            "attention": attention or cfg.model.attention, "format": vars(fmt),
            "cite": theta is not None or head is not None,
            "methods": (["evidence"] if head is not None else []) + (["at2", "contextcite"] if theta is not None else []),
            "at2": at2, "evidence": evidence_dir}
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
    def row(id: str, shuffle: int = 0, seed: int = 0):
        r = get(id)
        # release order, or a seeded shuffle; everything below is in the order the model saw
        ex = mlm.encode(r, tok, fmt, random.Random(f"{id}:{seed}") if shuffle else None)
        order = ex["order"]
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
            "model": f"{cfg.model.id} ({run})", "options": [r.options[i] for i in order],
            "order": order, "shuffled": bool(shuffle), "gold": ex["gold"],
            "answer": int(s.argmax()), "q": torch.softmax(s, 0).tolist(),
            "parts": [p.name for p in r.parts], "hasSpans": r.has_spans,
            "spans": [{"part": sp.part, "start": sp.start, "end": sp.end} for sp in r.spans],
            "partTexts": [p.text for p in r.parts], "tokens": len(ex["input_ids"]),
            "sentences": [{"part": x["part"], "start": x["start"], "end": x["end"], "sep": x["sep"],
                           "gold": j in gold, "text": r.parts[x["part"]].text[x["start"]:x["end"]]}
                          for j, x in enumerate(sources)],
            "seconds": seconds,
        }

    @app.get("/api/attribute")
    def attribute(id: str):
        if theta is None and head is None:
            raise fastapi.HTTPException(400, "no AT2 run or evidence head loaded (--at2 / --evidence)")
        r = get(id)
        out = {}
        if head is not None:
            with gpu:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                res = evidence.predict(scorer, head, tok, fmt, r)
                torch.cuda.synchronize()
                seconds = time.perf_counter() - t0
            sources = context_cite.split_sources(r)
            _, ranges = encoder_at2.source_token_ranges(r, tok, fmt, sources)
            by_index = {t["index"]: t["p"] for t in res["tokens"]}
            w = [max((by_index.get(i, 0.0) for i in range(a, b)), default=0.0) for a, b in ranges]
            text = [p_.text for p_ in r.parts]
            real = [t for t in res["tokens"] if text[t["part"]][t["start"]:t["end"]].strip()]
            out["evidence"] = {
                "w": w,  # words are scored whatever the sentence count; only sentence metrics need >= 2
                "top": [j for j in sorted(range(len(w)), key=lambda j: -w[j])[:3]],
                "seconds": seconds, "answer": res["answer"],
                "metrics": context_cite.span_metrics(r, {"sources": [{**s_, "w": v} for s_, v in zip(sources, w)]})
                if r.has_spans and len(w) >= 2 else None,
                "tokens": [{**t, "w": t["p"]} for t in res["tokens"]],
                "threshold": threshold, "maxShare": max_share,
                "topTokens": [t["index"] for t in sorted(real, key=lambda t: -t["p"])[:10]]}
        if theta is None:
            return out
        for name, fn in (("at2", lambda: encoder_at2.attribute(scorer, tok, fmt, r, theta)),
                         ("contextcite", lambda: encoder_cite.attribute(
                             scorer, tok, fmt, r, budget=cfg.model.pack_tokens))):
            with gpu:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                res = fn()
                torch.cuda.synchronize()
                seconds = time.perf_counter() - t0
            w = [x["w"] for x in res["sources"]]
            if len(w) < 2 and name == "contextcite":  # ablating one sentence can only remove everything
                out[name] = {"w": None, "reason": "one sentence: ContextCite needs at least two to ablate"}
                continue
            out[name] = {"w": w, "top": [j for j in sorted(range(len(w)), key=lambda j: -w[j])[:3] if w[j] > 0],
                         "seconds": seconds, "answer": res["answer"], "ablations": res.get("ablations"),
                         "metrics": context_cite.span_metrics(r, res) if r.has_spans and len(w) >= 2 else None}
            if "tokens" in res:  # AT2: per-token scores over the document, and its 10 highest (not whitespace)
                toks = res["tokens"]
                text = [p.text for p in r.parts]
                real = [t for t in toks if text[t["part"]][t["start"]:t["end"]].strip()]
                out[name]["tokens"] = toks
                out[name]["topTokens"] = [t["index"] for t in sorted(real, key=lambda t: -t["w"])[:10]]
        return out

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
    parser.add_argument("--at2", default="runs/at2_ultrabert_380m",
                        help="AT2 run directory (train.train_at2) for citations; '' for none")
    parser.add_argument("--evidence", default="runs/evidence_ultrabert_380m",
                        help="evidence head directory (train.train_evidence); '' for none")
    parser.add_argument("--max-share", type=float, default=0.15,
                        help="evidence words shown: at most this share of a document's words")
    parser.add_argument("--port", type=int, default=8767)
    args = parser.parse_args()
    at2 = args.at2 if args.at2 and (pathlib.Path(args.at2) / "theta.pt").exists() else None
    if args.at2 and not at2:
        logger.warning("no {}/theta.pt: serving without citations", args.at2)
    ev = args.evidence if args.evidence and (pathlib.Path(args.evidence) / "head.pt").exists() else None
    if args.evidence and not ev:
        logger.warning("no {}/head.pt: serving without the evidence head", args.evidence)
    uvicorn.run(make_app(args.run, args.data, args.split, args.attention, at2, ev, args.max_share),
                host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
