"""Best-of-N evidence labels: the LLM proposes citation sets and scores them (SelfCite).

Usage (servers from serve.sh; no local GPU needed):
  PYTHONPATH=src .venv/bin/python -m span_generation.best_of_n \
      --model Qwen/Qwen3.6-27B --base-url http://127.0.0.1:8011/v1 \
      --limit 20 --out runs/span_generation/labels.jsonl

Chuang et al., "SelfCite" (arXiv 2502.09604): the same model proposes and
scores. Per row, the context's sentences (context_cite.split_sources) are
tagged <c1>, <c2>, ... in cite.j2 with the question and the gold answer; the
model thinks, then cites <citations><c2><c7></citations>; N citation
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
import time
from typing import Any

from loguru import logger
import numpy as np
import torch

from dataloader import answers, data, prompts
from modeling import context_cite

_HERE = pathlib.Path(__file__).parent


def sentences(row: dict[str, Any], sources: list[dict[str, Any]]) -> list[str]:
    return [row["parts"][s["part"]]["text"][s["start"]:s["end"]] for s in sources]


_CITATIONS = re.compile(r"<citations>(.*?)</citations>", re.S)
_TAG = re.compile(r"<c(\d+)>")


def parse_citations(reply: str, d: int, thinking: bool = True) -> tuple[int, ...] | None:
    """0-based sentence indices from a reply "<think>…</think> <citations><c2><c7></citations>":
    the last citations block after the thinking; None if there is none, it is
    empty, or a tag is out of range. With thinking, a reply without "</think>"
    was cut off mid-reasoning and is rejected: a citations block inside the
    reasoning is a draft, not the answer."""
    if thinking and "</think>" not in reply:
        return None
    answer = reply.rsplit("</think>", 1)[-1]
    blocks = _CITATIONS.findall(answer)
    idx = [int(i) for i in _TAG.findall(blocks[-1])] if blocks else []
    if not idx or not all(1 <= i <= d for i in idx):
        return None
    return tuple(sorted({i - 1 for i in idx}))


async def propose(client, model: str, row: dict[str, Any], sources: list[dict[str, Any]], n: int,
                  temperature: float, top_p: float, max_tokens: int) -> tuple[list[tuple[int, ...]], dict]:
    """n sampled citation sets (0-based sentence indices) and reply stats;
    replies without a valid citations block (e.g. cut off while thinking) are
    dropped. The model thinks first (Qwen's <think> block), then cites.
    TODO: sampling settings for thinking mode (Qwen suggests T 0.6, top-p 0.95;
    the paper uses 1.2 / 0.9 for diversity); retry when few parse."""
    text = prompts.render("cite", directory=_HERE, sentences=sentences(row, sources),
                          question=row["question"], answer=row["labels"][row["label"]])
    reply = await client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": text}], n=n, temperature=temperature,
        top_p=top_p, max_tokens=max_tokens, extra_body={"chat_template_kwargs": {"enable_thinking": True}})
    # max_tokens is the caller's budget, already capped to fit the context (label)
    out = [s for c in reply.choices if (s := parse_citations(c.message.content or "", len(sources)))]
    stats = {"cut_off": sum(c.finish_reason == "length" for c in reply.choices),
             "completion_tokens": reply.usage.completion_tokens / n}
    return out, stats


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
    prompt_len = len(tok(prompts.render("cite", directory=_HERE, sentences=sentences(row, sources),
                                        question=row["question"], answer=row["labels"][row["label"]]))["input_ids"])
    budget = min(args.max_tokens, args.max_model_len - prompt_len - 256)  # 256: chat template margin
    proposed, stats = await propose(client, model, row, sources, args.n, args.temperature, args.top_p, budget)
    lengths = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in sentences(row, sources)]
    sets = [S for S in collections.Counter(proposed)                          # distinct, first-seen order
            if len(S) == 1 or sum(lengths[i] for i in S) <= args.max_cited_tokens]
    if not sets:
        return None
    full, scored = await rewards(client, model, tok, row, sources, sets)
    best = max(range(len(sets)), key=lambda i: scored[i]["reward"])
    return {"id": row["id"], "source": row["source"], **full, "n_parsed": len(proposed), **stats,
            "candidates": [{"sentences": list(S), "votes": proposed.count(S), **r}
                           for S, r in zip(sets, scored)],
            "spans": context_cite.to_spans(row, [sources[i] for i in sets[best]])}


async def run(args) -> None:
    from openai import AsyncOpenAI
    from transformers import AutoTokenizer

    # 8 long thinking replies in one request can take well over the client's default 600 s
    clients = [AsyncOpenAI(base_url=u, api_key="local", timeout=args.timeout, max_retries=2)
               for u in args.base_url]
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
            try:
                return await label(clients[i % len(clients)], args.served_name, tok, r, args)
            except Exception as e:  # one bad row must not stop the run; it is retried on resume
                logger.warning("{}: skipped ({}: {})", r["id"], type(e).__name__, str(e)[:200])
                return None

    # resume: rows already labelled in --out are skipped; new records are appended
    done = {json.loads(line)["id"] for line in open(out)} if out.exists() else set()
    rows = [r for r in rows if r["id"] not in done]
    logger.info("{} rows to label ({} already in {})", len(rows), len(done), out)
    start, finished = time.time(), 0
    with open(out, "a") as f:
        for task in asyncio.as_completed([one(i, r) for i, r in enumerate(rows)]):
            rec = await task
            finished += 1
            if rec is not None:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
            if finished % 50 == 0 or finished == len(rows):
                el = time.time() - start
                logger.info("{}/{} rows, {:.1f} rows/min, eta {:.1f} h", finished, len(rows), finished / el * 60,
                            el / finished * (len(rows) - finished) / 3600)
    logger.info("wrote {}", out)


def main() -> None:
    """Labels --limit rows per dataset with at least two sentences, one JSON line each.

    Re-running with the same --out resumes: rows already in it are skipped.
    TODO: the q(gold) cut-off and the picker (reward, vote) are chosen after
    the run from the stored candidates.
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
    parser.add_argument("--max-tokens", type=int, default=16384, help="per completion, thinking included")
    parser.add_argument("--max-model-len", type=int, default=32768, help="the server's context (serve.sh)")
    parser.add_argument("--concurrency", type=int, default=16, help="rows in flight")
    parser.add_argument("--timeout", type=float, default=3600, help="seconds per request")
    parser.add_argument("--source", nargs="*", help="only these datasets")
    parser.add_argument("--with-spans", action="store_true",
                        help="only rows with human spans (to validate the labels against them)")
    parser.add_argument("--limit", type=int, default=20, help="rows per dataset")
    parser.add_argument("--out", required=True)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
