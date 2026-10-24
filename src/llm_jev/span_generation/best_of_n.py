"""Best-of-N evidence labels: the LLM proposes citation sets and scores them (SelfCite).

Chuang et al., SelfCite (arXiv 2502.09604). The model proposes N citation sets
per example (the paper: N = 10, temperature 1.2, top-p 0.9) and each set S is
scored by reward(S) = log q(gold | S only) - log q(gold | x without S); the
highest-reward set becomes the spans. Unlike the paper, which scores the
model's own response, we score the gold answer, so the reward is uninformative
where the model does not believe the gold answer (cut-off to be chosen).
"""

import argparse
import asyncio
import collections
import dataclasses
import json
import pathlib
import re
import time
from typing import Any

from loguru import logger
import numpy as np
import torch

from llm_jev.dataloader import answers, prompts, schema, storage
from llm_jev.modeling import context_cite

_HERE = pathlib.Path(__file__).parent


def sentences(example: schema.Example, sources: list[dict[str, Any]]) -> list[str]:
    return [example.parts[s["part"]].text[s["start"]:s["end"]] for s in sources]


_CITATIONS = re.compile(r"<citations>(.*?)</citations>", re.S)
_TAG = re.compile(r"<c(\d+)>")


def parse_citations(reply: str, d: int, thinking: bool = True) -> tuple[int, ...] | None:
    """0-based sentence indices from the last <citations> block after the thinking.

    None if there is no valid block. With thinking, a reply without "</think>"
    was cut off and is rejected.
    """
    if thinking and "</think>" not in reply:
        return None
    answer = reply.rsplit("</think>", 1)[-1]
    blocks = _CITATIONS.findall(answer)
    idx = [int(i) for i in _TAG.findall(blocks[-1])] if blocks else []
    if not idx or not all(1 <= i <= d for i in idx):
        return None
    return tuple(sorted({i - 1 for i in idx}))


async def propose(client, model: str, example: schema.Example, sources: list[dict[str, Any]], n: int,
                  temperature: float, top_p: float, max_tokens: int) -> tuple[list[tuple[int, ...]], dict]:
    """n sampled citation sets (0-based sentence indices) and reply stats.

    Replies without a valid citations block are dropped.
    TODO: sampling settings for thinking mode (Qwen suggests T 0.6, top-p 0.95;
    the paper uses 1.2 / 0.9 for diversity); retry when few parse."""
    text = prompts.render("cite", directory=_HERE, sentences=sentences(example, sources),
                          question=example.question, answer=example.answer)
    reply = await client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": text}], n=n, temperature=temperature,
        top_p=top_p, max_tokens=max_tokens, extra_body={"chat_template_kwargs": {"enable_thinking": True}})
    # max_tokens: the caller's budget, already capped to fit the context.
    out = [s for c in reply.choices if (s := parse_citations(c.message.content or "", len(sources)))]
    stats = {"cut_off": sum(c.finish_reason == "length" for c in reply.choices),
             "completion_tokens": reply.usage.completion_tokens / n}
    return out, stats


async def option_scores(client, model: str, prompt: list[int], suffixes: list[list[int]]) -> torch.Tensor:
    """s_k = log p(suffix k | prompt) for every option (Eq. 1).

    Each trie node is one prefill-only request for the log-probabilities of
    its legal next tokens (vLLM's logprob_token_ids), one depth at a time so
    deeper nodes reuse the prompt from the prefix cache. Token ids are sent,
    so the server does not re-tokenise.
    """
    nodes = answers.trie(suffixes)

    async def one(node):
        r = await client.completions.create(
            model=model, prompt=prompt + list(node), max_tokens=1, logprobs=len(nodes[node]), temperature=0,
            extra_body={"logprob_token_ids": nodes[node], "return_tokens_as_token_ids": True})
        top = r.choices[0].logprobs.top_logprobs[0]
        return node, {t: top[f"token_id:{t}"] for t in nodes[node]}

    lp: dict[tuple[int, ...], dict[int, float]] = {}
    for d in sorted({len(n) for n in nodes}):
        lp.update(await asyncio.gather(*(one(n) for n in nodes if len(n) == d)))
    return torch.tensor([sum(lp[tuple(s[:t])][s[t]] for t in range(len(s))) for s in suffixes],
                        dtype=torch.float64)


async def rewards(client, model: str, tok, example: schema.Example, sources: list[dict[str, Any]],
                  sets: list[tuple[int, ...]]) -> tuple[dict[str, Any], list[dict[str, float]]]:
    """Scores the full context, each set removed and each set alone.

    Returns the full-context record and each set's necessity, sufficiency and reward.
    """
    d = len(sources)
    masks = [np.ones(d, bool)]
    for S in sets:
        keep = np.isin(np.arange(d), S)
        masks += [~keep, keep]
    suffixes = answers.candidates(tok, len(example.options))[0]
    gold = example.gold
    prompts_ = []
    for keep in masks:
        ex = answers.encode(context_cite.ablate(example, sources, keep), tok, None)
        prompts_.append(ex["input_ids"][: ex["path"][0][0] + 1])
    s = await asyncio.gather(*(option_scores(client, model, p, suffixes) for p in prompts_))
    log_q = [float(torch.log_softmax(x, 0)[gold]) for x in s]
    full = {"log_q_gold": log_q[0], "model_answer": int(s[0].argmax()), "gold": gold}
    scored = [{"necessity": log_q[0] - log_q[1 + 2 * i], "sufficiency": log_q[2 + 2 * i] - log_q[0],
               "reward": log_q[2 + 2 * i] - log_q[1 + 2 * i]} for i in range(len(sets))]
    return full, scored


async def label(client, model: str, tok, example: schema.Example, args) -> dict[str, Any] | None:
    """One row's record: the proposed sets with their rewards, and the best as spans.

    None if no set survives parsing and the token cap.
    """
    sources = context_cite.split_sources(example)
    prompt_len = len(tok(prompts.render("cite", directory=_HERE, sentences=sentences(example, sources),
                                        question=example.question, answer=example.answer))["input_ids"])
    budget = min(args.max_tokens, args.max_model_len - prompt_len - 256)  # 256: chat template margin
    proposed, stats = await propose(client, model, example, sources, args.n, args.temperature, args.top_p, budget)
    lengths = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in sentences(example, sources)]
    sets = [S for S in collections.Counter(proposed)                          # distinct, first-seen order
            if len(S) == 1 or sum(lengths[i] for i in S) <= args.max_cited_tokens]
    if not sets:
        return None
    full, scored = await rewards(client, model, tok, example, sources, sets)
    best = max(range(len(sets)), key=lambda i: scored[i]["reward"])
    return {"id": example.id, "source": example.source, **full, "n_parsed": len(proposed), **stats,
            "candidates": [{"sentences": list(S), "votes": proposed.count(S), **r}
                           for S, r in zip(sets, scored)],
            "spans": [dataclasses.asdict(s)
                      for s in context_cite.to_spans([sources[i] for i in sets[best]])]}


def labelled_ids(out: pathlib.Path) -> set[str]:
    """The ids already written to out, for resuming.

    A partial last line (from a killed run) is cut from the file, so its row is
    labelled again. A bad line anywhere else is an error.
    """
    if not out.exists():
        return set()
    lines = out.read_text().splitlines(keepends=True)
    done = set()
    for number, line in enumerate(lines):
        try:
            done.add(json.loads(line)["id"])
        except (json.JSONDecodeError, KeyError):
            if number != len(lines) - 1:
                raise ValueError(f"{out}:{number + 1}: not a record")
            logger.warning("{}: dropping the partial last line", out)
            out.write_text("".join(lines[:-1]))
    return done


async def run(args) -> None:
    from openai import AsyncOpenAI
    from transformers import AutoTokenizer

    # Long thinking replies can exceed the client's default 600 s timeout.
    clients = [AsyncOpenAI(base_url=u, api_key="local", timeout=args.timeout, max_retries=2)
               for u in args.base_url]
    tok = AutoTokenizer.from_pretrained(args.model)
    rows = [r for r in storage.load_examples(args.data, args.split)
            if (not args.source or r.source in args.source) and (not args.with_spans or r.has_spans)
            and len(context_cite.split_sources(r)) >= 2]
    by: dict[str, list] = collections.defaultdict(list)                     # --limit rows per dataset
    for r in rows:
        by[r.source].append(r)
    rows = [r for rs in by.values() for r in rs[: args.limit]]
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    gate = asyncio.Semaphore(args.concurrency)

    async def one(i, r):
        async with gate:
            try:
                return await label(clients[i % len(clients)], args.served_name, tok, r, args)
            except Exception as e:  # one bad row must not stop the run; it is retried on resume
                logger.warning("{}: skipped ({}: {})", r.id, type(e).__name__, str(e)[:200])
                return None

    # Resume: skip rows already in --out.
    done = labelled_ids(out)
    rows = [r for r in rows if r.id not in done]
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
    """Labels --limit rows per dataset, one JSON line each; re-running with the same --out resumes.

    TODO: the q(gold) cut-off and the picker (reward, vote) are chosen after
    the run from the stored candidates.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="Qwen/Qwen3.6-27B", help="HF id, for the tokenizer")
    parser.add_argument("--served-name", default="labeller", help="the name serve.sh gives the model")
    parser.add_argument("--base-url", nargs="+", default=["http://127.0.0.1:8011/v1"],
                        help="one or more servers (rows are spread round-robin)")
    parser.add_argument("--data", default=storage.DEFAULT_SOURCE)
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
