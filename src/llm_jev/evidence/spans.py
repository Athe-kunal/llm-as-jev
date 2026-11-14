"""Best-of-N evidence labels: an LLM proposes citation sets and scores them.

Chuang et al., SelfCite (arXiv 2502.09604). The model proposes N citation sets
per example (the paper: N = 10, temperature 1.2, top-p 0.9) and each set S is
scored by reward(S) = log q(gold | S only) - log q(gold | x without S); the
highest-reward set becomes the spans. Unlike the paper, which scores the
model's own response, we score the gold answer.

Usage:
  scripts/serve_labeller.sh                  # vLLM server(s), see the script
  llm-jev-spans --config configs/evidence/spans.yaml out=runs/spans/train.jsonl

Writes one JSON line per row; re-running with the same `out` resumes.
"""

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

from llm_jev import config
from llm_jev.data import prompts
from llm_jev.data import schema
from llm_jev.data import storage
from llm_jev.evidence import sentences as sentences_lib
from llm_jev.models import decoder


def sentences(
    example: schema.Example, sources: list[sentences_lib.Sentence]
) -> list[str]:
    """The texts of the sentences."""
    return [example.parts[s.part].text[s.start : s.end] for s in sources]


_CITATIONS = re.compile(r"<citations>(.*?)</citations>", re.S)
_TAG = re.compile(r"<c(\d+)>")


def parse_citations(
    reply: str, d: int, thinking: bool = True
) -> tuple[int, ...] | None:
    """0-based sentence indices from the last <citations> block after thinking.

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


async def propose(
    client,
    model: str,
    example: schema.Example,
    sources: list[sentences_lib.Sentence],
    n: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> tuple[list[tuple[int, ...]], dict]:
    """N sampled citation sets (0-based sentence indices) and reply stats.

    Replies without a valid citations block are dropped.
    """
    text = prompts.render(
        "cite",
        sentences=sentences(example, sources),
        question=example.question,
        answer=example.answer,
    )
    reply = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": text}],
        n=n,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": True}},
    )
    # max_tokens: the caller's budget, already capped to fit the context.
    out = [
        s
        for c in reply.choices
        if (s := parse_citations(c.message.content or "", len(sources)))
    ]
    stats = {
        "cut_off": sum(c.finish_reason == "length" for c in reply.choices),
        "completion_tokens": reply.usage.completion_tokens / n,
    }
    return out, stats


async def option_scores(
    client, model: str, prompt: list[int], suffixes: list[list[int]]
) -> torch.Tensor:
    """s_k = log p(suffix k | prompt) for every option (Eq. 1).

    Uses the server's echoed prompt log-probabilities. Token ids are sent, so
    the server does not re-tokenise.
    """

    async def one(suffix):
        r = await client.completions.create(
            model=model,
            prompt=prompt + suffix,
            max_tokens=1,
            echo=True,
            logprobs=0,
            temperature=0,
        )
        lp = r.choices[0].logprobs.token_logprobs[
            : len(prompt) + len(suffix)
        ]  # drops the generated token
        return sum(lp[len(prompt) :])

    return torch.tensor(
        await asyncio.gather(*(one(s) for s in suffixes)), dtype=torch.float64
    )


async def rewards(
    client,
    model: str,
    tok,
    example: schema.Example,
    sources: list[sentences_lib.Sentence],
    sets: list[tuple[int, ...]],
) -> tuple[dict[str, Any], list[dict[str, float]]]:
    """Scores the full context, each set removed and each set alone.

    Returns:
      The full-context record, and each set's necessity, sufficiency and
      reward.
    """
    d = len(sources)
    masks = [np.ones(d, bool)]
    for cited in sets:
        keep = np.isin(np.arange(d), cited)
        masks += [~keep, keep]
    suffixes = decoder.candidates(tok, len(example.options))[0]
    gold = example.gold
    prompts_ = []
    for keep in masks:
        ablated = sentences_lib.ablate(example, sources, keep)
        prompts_.append(decoder.prompt_ids(ablated, tok))
    s = await asyncio.gather(
        *(option_scores(client, model, p, suffixes) for p in prompts_)
    )
    log_q = [float(torch.log_softmax(x, 0)[gold]) for x in s]
    full = {
        "log_q_gold": log_q[0],
        "model_answer": int(s[0].argmax()),
        "gold": gold,
    }
    scored = [
        {
            "necessity": log_q[0] - log_q[1 + 2 * i],
            "sufficiency": log_q[2 + 2 * i] - log_q[0],
            "reward": log_q[2 + 2 * i] - log_q[1 + 2 * i],
        }
        for i in range(len(sets))
    ]
    return full, scored


async def label(
    client, model: str, tok, example: schema.Example, args: config.SpansConfig
) -> dict[str, Any] | None:
    """One row's record: the proposed sets, their rewards, the best as spans.

    None if no set survives parsing and the token cap.
    """
    sources = sentences_lib.split(example)
    prompt_len = len(
        tok(
            prompts.render(
                "cite",
                sentences=sentences(example, sources),
                question=example.question,
                answer=example.answer,
            )
        )["input_ids"]
    )
    budget = min(
        args.max_tokens, args.max_model_len - prompt_len - 256
    )  # 256: chat template margin
    proposed, stats = await propose(
        client,
        model,
        example,
        sources,
        args.n,
        args.temperature,
        args.top_p,
        budget,
    )
    lengths = [
        len(tok(t, add_special_tokens=False)["input_ids"])
        for t in sentences(example, sources)
    ]
    sets = [
        cited
        for cited in collections.Counter(proposed)  # distinct, first-seen order
        if len(cited) == 1
        or sum(lengths[i] for i in cited) <= args.max_cited_tokens
    ]
    if not sets:
        return None
    full, scored = await rewards(client, model, tok, example, sources, sets)
    best = max(range(len(sets)), key=lambda i: scored[i]["reward"])
    return {
        "id": example.id,
        "source": example.source,
        **full,
        "n_parsed": len(proposed),
        **stats,
        "candidates": [
            {"sentences": list(c), "votes": proposed.count(c), **r}
            for c, r in zip(sets, scored, strict=True)
        ],
        "spans": [
            dataclasses.asdict(s)
            for s in sentences_lib.to_spans([sources[i] for i in sets[best]])
        ],
    }


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
                raise ValueError(f"{out}:{number + 1}: not a record") from None
            logger.warning("{}: dropping the partial last line", out)
            out.write_text("".join(lines[:-1]))
    return done


async def run(args: config.SpansConfig) -> None:
    """Labels `args.limit` rows per dataset, resuming from `args.out`."""
    from openai import AsyncOpenAI
    from transformers import AutoTokenizer

    # Long thinking replies can exceed the client's default 600 s timeout.
    clients = [
        AsyncOpenAI(
            base_url=u, api_key="local", timeout=args.timeout, max_retries=2
        )
        for u in args.base_url
    ]
    tok = AutoTokenizer.from_pretrained(args.model)
    rows = [
        r
        for r in storage.load_examples(args.source, args.split)
        if (not args.datasets or r.source in args.datasets)
        and (not args.with_spans or r.has_spans)
        and len(sentences_lib.split(r)) >= 2
    ]
    by: dict[str, list] = collections.defaultdict(
        list
    )  # --limit rows per dataset
    for r in rows:
        by[r.source].append(r)
    rows = [r for rs in by.values() for r in rs[: args.limit]]
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    gate = asyncio.Semaphore(args.concurrency)

    async def one(i, r):
        async with gate:
            try:
                return await label(
                    clients[i % len(clients)], args.served_name, tok, r, args
                )
            # One bad row must not stop the run; it is retried on resume.
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "{}: skipped ({}: {})", r.id, type(e).__name__, str(e)[:200]
                )
                return None

    # Resume: skip rows already in --out.
    done = labelled_ids(out)
    rows = [r for r in rows if r.id not in done]
    logger.info(
        "{} rows to label ({} already in {})", len(rows), len(done), out
    )
    start, finished = time.time(), 0
    with open(out, "a") as f:
        for task in asyncio.as_completed(
            [one(i, r) for i, r in enumerate(rows)]
        ):
            rec = await task
            finished += 1
            if rec is not None:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
            if finished % 50 == 0 or finished == len(rows):
                el = time.time() - start
                logger.info(
                    "{}/{} rows, {:.1f} rows/min, eta {:.1f} h",
                    finished,
                    len(rows),
                    finished / el * 60,
                    el / finished * (len(rows) - finished) / 3600,
                )
    logger.info("wrote {}", out)


def main() -> None:
    """Labels rows from a YAML config and dotlist overrides."""
    cfg, _ = config.from_command_line(config.SpansConfig, __doc__)
    asyncio.run(run(cfg))


if __name__ == "__main__":
    main()
