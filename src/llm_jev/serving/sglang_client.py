"""Decisions and ContextCite through an SGLang server (its /v1/score API).

Usage (a server from scripts/sglang_serve.sh, or: make sglang-serve):
  .venv/bin/python -m llm_jev.serving.sglang_client --url http://127.0.0.1:30000 \
      --tokenizer runs/qwen3_4b/merged --source contractnli musique --per 5

Prompts are tokenised here and sent as token ids, so the server scores the
merged model (llm_jev.serving.merge) exactly as trained. SGLang's /v1/decisions
endpoint is not used: it cannot select a LoRA adapter.
"""

import argparse
import concurrent.futures
import json
import pathlib
import random
import time
import urllib.request
from typing import Any

from loguru import logger
import numpy as np
import torch

from llm_jev.dataloader import answers, schema, storage
from llm_jev.modeling import context_cite, masked_cite


def post(url: str, payload: dict[str, Any], timeout: float = 600) -> dict[str, Any]:
    """POST payload as JSON to url, the JSON reply."""
    req = urllib.request.Request(url, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as reply:
        return json.loads(reply.read())


def score_prompts(url: str, tok, prompts: list[list[int]], ks: list[int], chunk: int = 8,
                  workers: int = 8) -> list[tuple]:
    """answers.score_batch through the server: (s, root legal mass, decoded) per prompt.

    prompts[i] ends at "Best answer: [" and has ks[i] options. Prompts go in
    chunks; each chunk sends its deeper trie nodes as soon as its own prompts
    are scored, so they hit the prefix cache while later chunks still prefill.
    """
    labels = sorted({t for k in set(ks) for legal in answers.candidates(tok, k)[1].values() for t in legal})
    width = labels[-1] + 1

    def request(items: list[list[int]]) -> list[torch.Tensor]:
        reply = post(f"{url}/v1/score", {"query": [], "items": items, "label_token_ids": labels,
                                         "return_token_logprobs": True})
        rows = []
        for row_lp in reply["token_logprobs"]:
            lp = torch.full((width,), -torch.inf, dtype=torch.float64)
            lp[labels] = torch.tensor(row_lp, dtype=torch.float64)
            rows.append(lp)
        return rows

    def score_chunk(idx: list[int]) -> list[dict]:
        depths: dict[int, list] = {}
        for j, i in enumerate(idx):
            for node in answers.candidates(tok, ks[i])[1]:
                depths.setdefault(len(node), []).append((j, node))
        logp: list[dict] = [{} for _ in idx]
        for _, level in sorted(depths.items()):
            for (j, node), lp in zip(level, request([prompts[idx[j]] + list(node) for j, node in level])):
                logp[j][node] = lp
        return logp

    parts = [list(range(c, min(c + chunk, len(prompts)))) for c in range(0, len(prompts), chunk)]
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        logp = [lp for part in pool.map(score_chunk, parts) for lp in part]
    out = []
    for lp, k in zip(logp, ks):
        suffixes, nodes = answers.candidates(tok, k)
        out.append(answers.read_trie(lp, suffixes, nodes))
    return out


def prompt_ids(tok, example: schema.Example) -> list[int]:
    """The example's prompt up to "Best answer: [", options in release order."""
    ex = answers.encode(example, tok, None)
    return ex["input_ids"][: ex["path"][0][0] + 1]


def predict(url: str, tok, example: schema.Example) -> tuple[int, torch.Tensor]:
    """The predicted option (argmax s) and the candidate scores s [K]."""
    s, _, _ = score_prompts(url, tok, [prompt_ids(tok, example)], [len(example.options)])[0]
    return int(s.argmax()), s


def attribute(url: str, tok, example: schema.Example, num_ablations: int | None = None,
              keep_prob: float = 0.5, alpha: float = 0.01, seed: int = 0, **request) -> dict[str, Any]:
    """context_cite.attribute through the server; num_ablations None uses masked_cite.ablation_count.

    The full prompt is scored with the ablations: the answer is only needed
    once every score is back.
    """
    sources = context_cite.split_sources(example)
    if not sources:  # nothing to cite: the decision alone
        answer, s = predict(url, tok, example)
        return context_cite.no_sources_result(s, answer)
    n = num_ablations or masked_cite.ablation_count(len(sources))
    masks = context_cite.ablation_masks(n, len(sources), keep_prob, seed)
    prompts = [prompt_ids(tok, example)]
    prompts += [prompt_ids(tok, context_cite.ablate(example, sources, keep)) for keep in masks]
    scored = score_prompts(url, tok, prompts, [len(example.options)] * (n + 1), **request)
    s = scored[0][0]
    answer = int(s.argmax())
    y = np.array([context_cite.logit_q(s_i, answer) for s_i, _, _ in scored[1:]])
    return context_cite.result(s, answer, sources, masks, y, alpha)


def main() -> None:
    """Attributes --per seeded eval rows per dataset and logs the results."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:30000")
    parser.add_argument("--tokenizer", required=True, help="the served model's folder or HF id")
    parser.add_argument("--data", default=storage.DEFAULT_SOURCE)
    parser.add_argument("--split", default="eval")
    parser.add_argument("--source", nargs="*", help="only these datasets")
    parser.add_argument("--per", type=int, default=5, help="rows per dataset")
    parser.add_argument("--ablations", type=int, help="default: masked_cite.ablation_count")
    parser.add_argument("--rows-in-flight", type=int, default=4,
                        help="rows attributed at once (1: per-row seconds are unloaded latency)")
    parser.add_argument("--out", help="JSON lines of the results")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    rows = [r for r in storage.load_examples(args.data, args.split)
            if (not args.source or r.source in args.source)
            and len(context_cite.split_sources(r)) >= 2]
    by: dict[str, list] = {}
    for r in rows:
        by.setdefault(r.source, []).append(r)
    picked = [row for _, rs in sorted(by.items())
              for row in random.Random(0).sample(rs, min(args.per, len(rs)))]

    def timed(row: schema.Example) -> tuple[schema.Example, dict[str, Any], float]:
        t0 = time.time()
        res = attribute(args.url, tok, row, args.ablations)
        return row, res, time.time() - t0

    out = open(args.out, "w") if args.out else None
    seconds = []
    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(args.rows_in_flight) as pool:
        for row, res, took in pool.map(timed, picked):
            seconds.append(took)
            top = sorted(res["sources"], key=lambda x: -x["w"])[:3]
            logger.info("{} {}: {} sentences, {:.2f} s, q {:.3f}, top w {}, {}", row.source, row.id,
                        len(res["sources"]), took, res["q"], [round(x["w"], 3) for x in top],
                        context_cite.span_metrics(row, res))
            if out:
                out.write(json.dumps({"id": row.id, "seconds": took, **res}) + "\n")
    wall = time.time() - start
    if seconds:
        logger.info("{} rows in {:.1f} s ({:.2f} rows/s, {} in flight): row seconds p50 {:.2f}, p95 {:.2f}",
                    len(seconds), wall, len(seconds) / wall, args.rows_in_flight,
                    np.percentile(seconds, 50), np.percentile(seconds, 95))
    if out:
        out.close()
        logger.info("wrote {}", pathlib.Path(args.out))


if __name__ == "__main__":
    main()
