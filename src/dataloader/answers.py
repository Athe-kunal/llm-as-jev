"""The answer interface: prompt, candidate trie, encoding, scoring.

Li & Wagle, "LLM-as-Jev" (arXiv 2610.02076v3, §3.2-3.3): the assistant turn
is prefilled with "Best answer: [" and candidate k is the suffix "k]". The
suffixes form a prefix-free trie; each node's legal tokens are the next
tokens that stay inside it. See docs/architecture.md.
"""

import copy
import functools
import random
from typing import Any

import torch

from dataloader import prompts

ANSWER_PREFIX = "Best answer: ["
# The assistant turn up to the answer, as the chat template renders it with
# add_generation_prompt=True, enable_thinking=False (Qwen3 and Qwen3.5).
_TAIL = "<|im_start|>assistant\n<think>\n\n</think>\n\n" + ANSWER_PREFIX
# Trie nodes off the gold path anchored per training example (App. I).
OFF_PATH = 3


def user_message(
    row: dict[str, Any], order: list[str]
) -> tuple[str, dict[str, tuple[int, int]]]:
    """The paper's user message (prompts/decision.j2) and each part's range in it.

    The state is the row's part texts (no names), then the question, on
    every row (like the paper's MMBench state: the hint, then the
    question). Options are numbered from 1 in the given order.
    """
    text = prompts.render(
        "decision",
        parts=row["parts"],
        question=row["question"],
        options=[row["labels"][label] for label in order],
    )
    where, cursor = {}, 0
    for p in row["parts"]:  # rendered in order, so search after the last one
        start = text.index(p["text"], cursor)
        where[p["name"]] = (start, start + len(p["text"]))
        cursor = start + len(p["text"])
    return text, where


def build_prompt(
    row: dict[str, Any], order: list[str], tokenizer: Any
) -> tuple[str, dict[str, tuple[int, int]]]:
    """Prompt text up to "Best answer: [", and each part's range in it."""
    content, where = user_message(row, order)
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    ) + ANSWER_PREFIX
    if not text.endswith(_TAIL):  # suffix_ids assumes this tail
        raise ValueError("the chat template's assistant turn is not _TAIL")
    start = text.index(content)
    return text, {n: (start + s, start + e) for n, (s, e) in where.items()}


def suffix_ids(tokenizer: Any, k: int) -> list[list[int]]:
    """Token ids of the suffixes "1]" .. "k]" after the answer prefix."""
    base = tokenizer(_TAIL, add_special_tokens=False)["input_ids"]
    out = []
    for i in range(1, k + 1):
        full = tokenizer(_TAIL + f"{i}]", add_special_tokens=False)["input_ids"]
        if full[: len(base)] != base:
            raise ValueError(f"suffix {i}] changes the prompt's tokens")
        out.append(full[len(base) :])
    return out


def trie(suffixes: list[list[int]]) -> dict[tuple[int, ...], list[int]]:
    """Each trie node (a prefix of a suffix) -> its legal next tokens, sorted.

    The root is (). Complete suffixes are leaves, not nodes.
    """
    nodes: dict[tuple[int, ...], set[int]] = {}
    for s in suffixes:
        for t in range(len(s)):
            nodes.setdefault(tuple(s[:t]), set()).add(s[t])
    if any(tuple(s) in nodes for s in suffixes):
        raise ValueError("the suffixes are not prefix-free")
    return {n: sorted(v) for n, v in nodes.items()}


@functools.cache
def _candidates(tokenizer: Any, k: int):
    """suffix_ids and their trie, once per tokenizer and option count."""
    suffixes = suffix_ids(tokenizer, k)
    return suffixes, trie(suffixes)


@functools.cache
def _prefix_len(tokenizer: Any) -> int:
    """Number of tokens of ANSWER_PREFIX at the end of _TAIL."""
    full = tokenizer(_TAIL, add_special_tokens=False)["input_ids"]
    bare = tokenizer(_TAIL[: -len(ANSWER_PREFIX)], add_special_tokens=False)
    if full[: len(bare["input_ids"])] != bare["input_ids"]:
        raise ValueError("ANSWER_PREFIX changes the tokens before it")
    return len(full) - len(bare["input_ids"])


def encode(row: dict[str, Any], tokenizer: Any, rng: random.Random | None):
    """Prompt + gold suffix token ids, the gold path, and the anchor sites.

    The input is the prompt followed by the whole gold suffix (teacher
    forcing). path[t] = (position, legal tokens, gold token): the logits at
    that position predict suffix token t, among the legal tokens of the node
    it leaves. reply: the position whose logits predict the reply's first
    token ("Best"). With rng (training) the options are shuffled and
    off_path holds OFF_PATH sampled trie nodes off the gold path, each as
    (node tokens, legal tokens); without rng it is empty.
    """
    order = list(row["labels"])
    if rng is not None:
        rng.shuffle(order)
    text, _ = build_prompt(row, order, tokenizer)
    prompt = tokenizer(text, add_special_tokens=False)["input_ids"]
    suffixes, nodes = _candidates(tokenizer, len(order))
    gold = order.index(row["label"])
    s = suffixes[gold]
    path = [
        (len(prompt) - 1 + t, nodes[tuple(s[:t])], s[t]) for t in range(len(s))
    ]
    off_path = []
    if rng is not None:
        on = {tuple(s[:t]) for t in range(len(s))}
        off = [n for n in nodes if n not in on]
        off_path = [(list(n), nodes[n])
                    for n in rng.sample(off, min(OFF_PATH, len(off)))]
    return {
        "input_ids": prompt + s,
        "gold": gold,
        "n_options": len(order),
        "path": path,
        "reply": len(prompt) - _prefix_len(tokenizer) - 1,
        "off_path": off_path,
    }


def _repeat_cache(cache, n: int) -> None:
    """Repeats a prompt's cache n times along the batch, in place.

    Attention layers use the library's batch_repeat_interleave; Qwen3.5's
    linear-attention layers have none, so their conv and recurrent states
    (fixed-size, one per state index) are repeated here.
    """
    for layer in cache.layers:
        if hasattr(layer, "batch_repeat_interleave"):
            layer.batch_repeat_interleave(n)
            continue
        for name in ("conv_states", "recurrent_states"):
            states = getattr(layer, name)
            keys = states.keys() if isinstance(states, dict) else range(len(states))
            for i in keys:
                if isinstance(states[i], torch.Tensor):
                    states[i] = states[i].repeat_interleave(n, dim=0)


def score(model, prompt: list[int], tokenizer: Any, k: int, device: str):
    """s_k (Eq. 1) for each candidate, the root's legal mass, and the answer
    decoded inside the trie.

    One prefill of the prompt; its cache is then copied for each depth of
    the trie, and all nodes of that depth run as one batch on top of it
    (1-3 short steps, App. C). The decoded answer takes the most likely
    legal token at each node from the root until the closing "]", so it is
    always a valid "k]" (greedy within the trie; nothing after "]").
    """
    suffixes, nodes = _candidates(tokenizer, k)
    out = model(input_ids=torch.tensor([prompt], device=device),
                use_cache=True, logits_to_keep=1)
    logp = {(): torch.log_softmax(out.logits[0, -1].float(), -1)}
    depths: dict[int, list] = {}
    for n in nodes:
        if n:
            depths.setdefault(len(n), []).append(n)
    for d, keys in sorted(depths.items()):
        cache = copy.deepcopy(out.past_key_values)
        _repeat_cache(cache, len(keys))
        o = model(input_ids=torch.tensor(keys, device=device),
                  past_key_values=cache, use_cache=True, logits_to_keep=1)
        for n, lp in zip(keys, torch.log_softmax(o.logits[:, -1].float(), -1)):
            logp[n] = lp
    s = torch.stack([
        sum(logp[tuple(x[:t])][x[t]] for t in range(len(x))) for x in suffixes
    ])
    node: tuple = ()
    while node in nodes:  # greedy over the legal tokens, down to a leaf
        node += (max(nodes[node], key=lambda t: logp[node][t].item()),)
    decoded = suffixes.index(list(node))
    return s, logp[()][nodes[()]].exp().sum().item(), decoded
