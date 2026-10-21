"""Decisions and ContextCite through an SGLang server (its /v1/score API).

Usage (a server from scripts/sglang_serve.sh, or: make sglang-serve):
  .venv/bin/python -m llm_jev.serving.sglang_client --url http://127.0.0.1:30000 \
      --tokenizer runs/qwen3_4b/merged --source contractnli musique --per 5

The prompt is ours (answers.encode: decision.j2, then "Best answer: ["),
tokenised here and sent as token ids, so the server needs no chat template
and the merged model (llm_jev.serving.merge) scores exactly what it was
trained on. SGLang's /v1/decisions endpoint is not used: it renders its own
prompt with single-token A-Z labels and cannot select a LoRA adapter.

Scoring: every trie node of a prompt is one item, prompt + node tokens,
with query []; the server returns the next-token log-probabilities of the
label tokens (every legal token of the trie; return_token_logprobs: the
full vocabulary's log-softmax), and answers.read_trie sums them along each
suffix into s_k (Eq. 1). An option count of 9 or less has a one-level trie, one
item per prompt.

ContextCite: the ablations delete the dropped sentences
(context_cite.ablate), as the reference does; RadixAttention reuses each
ablated prompt's prefix up to its first dropped sentence, and the trie
items reuse their whole prompt. The surrogate is context_cite's.
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


def score_prompts(url: str, tok, prompts: list[list[int]], ks: list[int], chunk: int = 32,
                  workers: int = 8) -> list[tuple]:
    """answers.score_batch through the server: (s, root legal mass, decoded)
    per prompt, prompts[i] ending at "Best answer: [" with ks[i] options.

    The items (every prompt's trie nodes) go out one trie depth at a time,
    the prompts first: a deeper node then finds its prompt in the radix
    cache and prefills only its own tokens (sent together, every node of a
    prompt would prefill the whole prompt again). Within a depth, requests
    of `chunk` items, `workers` at a time; the server batches them.
    """
    depths: dict[int, list] = {}
    for i, (p, k) in enumerate(zip(prompts, ks)):
        for node in answers.candidates(tok, k)[1]:
            depths.setdefault(len(node), []).append((i, node, p + list(node)))
    labels = sorted({t for k in set(ks) for legal in answers.candidates(tok, k)[1].values() for t in legal})

    def request(items: list[list[int]]) -> list[list[float]]:
        reply = post(f"{url}/v1/score", {"query": [], "items": items, "label_token_ids": labels,
                                         "return_token_logprobs": True})
        return reply["token_logprobs"]

    owners, scores = [], []
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        for _, level in sorted(depths.items()):
            parts = [[x[2] for x in level[c:c + chunk]] for c in range(0, len(level), chunk)]
            scores += [row for part in pool.map(request, parts) for row in part]
            owners += [(i, node) for i, node, _ in level]
    width = labels[-1] + 1
    logp: list[dict] = [{} for _ in prompts]
    for (i, node), row_lp in zip(owners, scores):
        lp = torch.full((width,), -torch.inf, dtype=torch.float64)
        lp[labels] = torch.tensor(row_lp, dtype=torch.float64)
        logp[i][node] = lp
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
    """context_cite.attribute through the server: the predicted answer, then
    one ablated prompt per mask (sentences deleted), all scored in one
    score_prompts call. num_ablations None: masked_cite.ablation_count."""
    answer, s = predict(url, tok, example)
    sources = context_cite.split_sources(example)
    n = num_ablations or masked_cite.ablation_count(len(sources))
    masks = context_cite.ablation_masks(n, len(sources), keep_prob, seed)
    prompts = [prompt_ids(tok, context_cite.ablate(example, sources, keep)) for keep in masks]
    scored = score_prompts(url, tok, prompts, [len(example.options)] * n, **request)
    y = np.array([context_cite.logit_q(s_i, answer) for s_i, _, _ in scored])
    return context_cite.result(s, answer, sources, masks, y, alpha)


def main() -> None:
    """Attributes --per seeded eval rows per dataset; logs the time, the top
    sentences and (rows with gold spans) context_cite.span_metrics."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:30000")
    parser.add_argument("--tokenizer", required=True, help="the served model's folder or HF id")
    parser.add_argument("--data", default=storage.DEFAULT_SOURCE)
    parser.add_argument("--split", default="eval")
    parser.add_argument("--source", nargs="*", help="only these datasets")
    parser.add_argument("--per", type=int, default=5, help="rows per dataset")
    parser.add_argument("--ablations", type=int, help="default: masked_cite.ablation_count")
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
    out = open(args.out, "w") if args.out else None
    for source, rs in sorted(by.items()):
        for row in random.Random(0).sample(rs, min(args.per, len(rs))):
            t0 = time.time()
            res = attribute(args.url, tok, row, args.ablations)
            took = time.time() - t0
            top = sorted(res["sources"], key=lambda x: -x["w"])[:3]
            logger.info("{} {}: {} sentences, {:.2f} s, q {:.3f}, top w {}, {}", source, row.id,
                        len(res["sources"]), took, res["q"], [round(x["w"], 3) for x in top],
                        context_cite.span_metrics(row, res))
            if out:
                out.write(json.dumps({"id": row.id, "seconds": took, **res}) + "\n")
    if out:
        out.close()
        logger.info("wrote {}", pathlib.Path(args.out))


if __name__ == "__main__":
    main()
