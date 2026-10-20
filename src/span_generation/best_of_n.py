"""Best-of-N evidence labels: the LLM proposes citation sets and scores them (SelfCite).

Usage (servers from serve.sh; no local GPU needed):
  PYTHONPATH=src .venv/bin/python -m span_generation.best_of_n \
      --model Qwen/Qwen3.6-27B --base-url http://127.0.0.1:8011/v1 \
      --limit 20 --out runs/span_generation/labels.jsonl

Chuang et al., "SelfCite" (arXiv 2502.09604): the same model proposes and
scores. Per row, the context's sentences (context_cite.split_sources) are
numbered into cite.j2 with the question and the gold answer, and N citation
sets are sampled (the paper: N = 10, temperature 1.2, top-p 0.9; duplicates
removed; sets citing more than 384 tokens dropped unless a single sentence).
Each set S is scored on the decision prompt (answers.build_prompt) by the
tree probability q of the gold answer, from the server's echoed prompt
log-probabilities:
  reward(S) = log q(gold | S only) - log q(gold | x without S)
            = sufficiency + necessity,
and the highest-reward set becomes the row's spans. Unlike the paper, which
scores the model's own response, we score the gold answer, so each row also
records q(gold | x) and the model's own answer: the reward is uninformative
where the model does not believe the gold answer (cut-off to be chosen).
"""

import argparse
import asyncio
import collections
import json
import pathlib
import re
from typing import Any

from loguru import logger
import numpy as np
import torch

from dataloader import answers, data, prompts
from modeling import context_cite

_HERE = pathlib.Path(__file__).parent


def sentences(row: dict[str, Any], sources: list[dict[str, Any]]) -> list[str]:
    return [row["parts"][s["part"]]["text"][s["start"]:s["end"]] for s in sources]


async def propose(client, model: str, row: dict[str, Any], sources: list[dict[str, Any]], n: int,
                  temperature: float, top_p: float) -> list[tuple[int, ...]]:
    """n sampled citation sets (0-based sentence indices); replies that do not
    parse are dropped. TODO: thinking on or off for the 27B; retry when few parse."""
    text = prompts.render("cite", directory=_HERE, sentences=sentences(row, sources),
                          question=row["question"], answer=row["labels"][row["label"]])
    reply = await client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": text}], n=n, temperature=temperature,
        top_p=top_p, max_tokens=256, extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    out = []
    for choice in reply.choices:
        m = re.search(r"\{.*\}", choice.message.content or "", re.S)
        try:
            idx = json.loads(m.group(0))["sentences"] if m else None
        except (json.JSONDecodeError, KeyError, TypeError):
            idx = None
        if isinstance(idx, list) and idx and all(isinstance(i, int) and 1 <= i <= len(sources) for i in idx):
            out.append(tuple(sorted({i - 1 for i in idx})))
    return out


async def option_scores(client, model: str, prompt: list[int], suffixes: list[list[int]]) -> torch.Tensor:
    """s_k = log p(suffix k | prompt) for every option (Eq. 1), from the
    server's echoed prompt log-probabilities of prompt + suffix k. Token ids
    are sent, so the server does not re-tokenise; the K requests share the
    prompt prefix (prefix caching)."""
    async def one(suffix):
        r = await client.completions.create(model=model, prompt=prompt + suffix, max_tokens=1,
                                            echo=True, logprobs=0, temperature=0)
        lp = r.choices[0].logprobs.token_logprobs[: len(prompt) + len(suffix)]  # drop the generated token
        return sum(lp[len(prompt):])
    return torch.tensor(await asyncio.gather(*(one(s) for s in suffixes)), dtype=torch.float64)


async def rewards(client, model: str, tok, row: dict[str, Any], sources: list[dict[str, Any]],
                  sets: list[tuple[int, ...]]) -> tuple[dict[str, Any], list[dict[str, float]]]:
    """Scores the full context, each set removed and each set alone on the
    decision prompt; returns the full-context record (q of gold, the model's
    own answer) and each set's necessity, sufficiency and reward."""
    d = len(sources)
    masks = [np.ones(d, bool)]
    for S in sets:
        keep = np.isin(np.arange(d), S)
        masks += [~keep, keep]
    suffixes = answers.candidates(tok, len(row["labels"]))[0]
    gold = list(row["labels"]).index(row["label"])
    prompts_ = []
    for keep in masks:
        ex = answers.encode(context_cite.ablate(row, sources, keep), tok, None)
        prompts_.append(ex["input_ids"][: ex["path"][0][0] + 1])
    s = await asyncio.gather(*(option_scores(client, model, p, suffixes) for p in prompts_))
    log_q = [float(torch.log_softmax(x, 0)[gold]) for x in s]
    full = {"log_q_gold": log_q[0], "model_answer": int(s[0].argmax()), "gold": gold}
    scored = [{"necessity": log_q[0] - log_q[1 + 2 * i], "sufficiency": log_q[2 + 2 * i] - log_q[0],
               "reward": log_q[2 + 2 * i] - log_q[1 + 2 * i]} for i in range(len(sets))]
    return full, scored


async def label(client, model: str, tok, row: dict[str, Any], args) -> dict[str, Any] | None:
    """One row's record: the distinct proposed sets with their rewards and the
    best one as spans (None if no reply parsed or no set survives the cap)."""
    sources = context_cite.split_sources(row)
    proposed = await propose(client, model, row, sources, args.n, args.temperature, args.top_p)
    lengths = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in sentences(row, sources)]
    sets = [S for S in collections.Counter(proposed)                          # distinct, first-seen order
            if len(S) == 1 or sum(lengths[i] for i in S) <= args.max_cited_tokens]
    if not sets:
        return None
    full, scored = await rewards(client, model, tok, row, sources, sets)
    best = max(range(len(sets)), key=lambda i: scored[i]["reward"])
    return {"id": row["id"], "source": row["source"], **full, "n_parsed": len(proposed),
            "candidates": [{"sentences": list(S), "votes": proposed.count(S), **r}
                           for S, r in zip(sets, scored)],
            "spans": context_cite.to_spans(row, [sources[i] for i in sets[best]])}


async def run(args) -> None:
    from openai import AsyncOpenAI
    from transformers import AutoTokenizer

    clients = [AsyncOpenAI(base_url=u, api_key="local") for u in args.base_url]
    tok = AutoTokenizer.from_pretrained(args.model)
    rows = [r for r in data.load_rows(args.data, args.split)
            if (not args.source or r["source"] in args.source) and (not args.with_spans or r["spans"])
            and len(context_cite.split_sources(r)) >= 2]
    by: dict[str, list] = collections.defaultdict(list)                     # --limit rows per dataset
    for r in rows:
        by[r["source"]].append(r)
    rows = [r for rs in by.values() for r in rs[: args.limit]]
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    gate = asyncio.Semaphore(args.concurrency)

    async def one(i, r):
        async with gate:
            return await label(clients[i % len(clients)], args.served_name, tok, r, args)

    with open(out, "w") as f:
        for task in asyncio.as_completed([one(i, r) for i, r in enumerate(rows)]):
            rec = await task
            if rec is not None:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                logger.info("{}: {} sets, best reward {:.2f}, q(gold) {:.2f}", rec["id"], len(rec["candidates"]),
                            max(c["reward"] for c in rec["candidates"]), float(np.exp(rec["log_q_gold"])))
    logger.info("wrote {}", out)


def main() -> None:
    """Labels --limit rows per dataset with at least two sentences, one JSON line each.

    TODO: which rows first (the rows with human spans, to validate against
    them), the q(gold) cut-off, and resuming a partial run.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="Qwen/Qwen3.6-27B", help="HF id, for the tokenizer")
    parser.add_argument("--served-name", default="labeller", help="the name serve.sh gives the model")
    parser.add_argument("--base-url", nargs="+", default=["http://127.0.0.1:8011/v1"],
                        help="one or more servers (rows are spread round-robin)")
    parser.add_argument("--data", default=data.TRAIN_EVAL)
    parser.add_argument("--split", default="train")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-cited-tokens", type=int, default=384)
    parser.add_argument("--concurrency", type=int, default=16, help="rows in flight")
    parser.add_argument("--source", nargs="*", help="only these datasets")
    parser.add_argument("--with-spans", action="store_true",
                        help="only rows with human spans (to validate the labels against them)")
    parser.add_argument("--limit", type=int, default=20, help="rows per dataset")
    parser.add_argument("--out", required=True)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
