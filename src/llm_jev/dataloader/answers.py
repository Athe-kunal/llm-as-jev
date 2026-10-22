"""The answer interface: prompt, candidate trie, encoding, scoring.

Li & Wagle, "LLM-as-Jev" (arXiv 2610.02076v3, §3.2-3.3): the assistant turn
is prefilled with "Best answer: [" and candidate k is the suffix "k]". The
suffixes form a prefix-free trie; each node's legal tokens are the next
tokens that stay inside it.
"""

import copy
import functools
import random
from collections.abc import Sequence
from typing import Any

import torch

from llm_jev.dataloader import prompts
from llm_jev.dataloader import schema

ANSWER_PREFIX = "Best answer: ["
# The assistant turn up to the answer, as the chat template renders it with
# add_generation_prompt=True, enable_thinking=False (Qwen3 and Qwen3.5).
_TAIL = "<|im_start|>assistant\n<think>\n\n</think>\n\n" + ANSWER_PREFIX
# Trie nodes off the gold path anchored per training example (App. I).
OFF_PATH = 3


def user_message(
    example: schema.Example, order: Sequence[int]
) -> tuple[str, list[tuple[int, int]]]:
    """The paper's user message (prompts/decision.j2) and each part's range in it.

    The state is the example's part texts (no names), then the question
    (like the paper's MMBench state: the hint, then the question). Options
    are numbered from 1 in the given order.

    Args:
      example: The example.
      order: The option indices in the order they are shown.

    Returns:
      (the message, the (start, end) character range of every part, indexed
      like `Example.parts`).
    """
    # The template inserts each part's text verbatim, so render with a marker
    # in place of every text, locate the markers, then put the texts in.
    # (Searching for the texts could match the template's own words.)
    marks = [f"\x00{i}\x00" for i in range(len(example.parts))]
    marked = prompts.render(
        "decision",
        parts=[schema.Part(p.name, m) for p, m in zip(example.parts, marks)],
        question=example.question,
        options=[example.options[i] for i in order],
    )
    pieces, where, cursor, length = [], [], 0, 0
    for part, mark in zip(example.parts, marks):
        found = marked.index(mark, cursor)
        pieces += [marked[cursor:found], part.text]
        length += (found - cursor)
        where.append((length, length + len(part.text)))
        length += len(part.text)
        cursor = found + len(mark)
    pieces.append(marked[cursor:])
    return "".join(pieces), where


def build_prompt(
    example: schema.Example, order: Sequence[int], tokenizer: Any
) -> tuple[str, list[tuple[int, int]]]:
    """Prompt text up to "Best answer: [", and each part's range in it.

    Args:
      example: The example.
      order: The option indices in the order they are shown.
      tokenizer: The tokenizer whose chat template builds the prompt.

    Returns:
      (the prompt, the (start, end) character range of every part in it,
      indexed like `Example.parts`).

    Raises:
      ValueError: If the chat template's assistant turn is not _TAIL.
    """
    content, where = user_message(example, order)
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    ) + ANSWER_PREFIX
    if not text.endswith(_TAIL):  # suffix_ids assumes this tail
        raise ValueError("the chat template's assistant turn is not _TAIL")
    start = text.index(content)
    return text, [(start + a, start + b) for a, b in where]


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
def candidates(tokenizer: Any, k: int):
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


def encode(
    example: schema.Example, tokenizer: Any, rng: random.Random | None
) -> dict[str, Any]:
    """Prompt + gold suffix token ids, the gold path, and the anchor sites.

    The input is the prompt followed by the whole gold suffix (teacher
    forcing). path[t] = (position, legal tokens, gold token): the logits at
    that position predict suffix token t, among the legal tokens of the node
    it leaves. reply: the position whose logits predict the reply's first
    token ("Best"). With rng (training) the options are shuffled and
    off_path holds OFF_PATH sampled trie nodes off the gold path, each as
    (node tokens, legal tokens); without rng it is empty.
    """
    order = list(range(len(example.options)))
    if rng is not None:
        rng.shuffle(order)
    text, _ = build_prompt(example, order, tokenizer)
    prompt = tokenizer(text, add_special_tokens=False)["input_ids"]
    suffixes, nodes = candidates(tokenizer, len(order))
    gold = order.index(example.gold)
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


def batches(lengths: list[int], budget: int, max_rows: int) -> list[list[int]]:
    """Row indices sorted by length, packed so rows x longest <= budget and
    rows <= max_rows."""
    out: list[list[int]] = []
    for i in sorted(range(len(lengths)), key=lengths.__getitem__):
        if out and len(out[-1]) < max_rows \
                and (len(out[-1]) + 1) * lengths[i] <= budget:
            out[-1].append(i)
        else:
            out.append([i])
    return out


def _select_cache(cache, rows: torch.Tensor):
    """A new cache holding the given batch rows of cache (repeats allowed).

    Each layer is shallow-copied with its own state dicts, then the
    library's reorder_cache picks the rows (index_select along the batch);
    that covers attention keys/values and Qwen3.5's linear-attention conv
    and recurrent states, and leaves cache itself untouched.
    """
    new = copy.copy(cache)
    new.layers = []
    for layer in cache.layers:
        layer = copy.copy(layer)
        for name, value in vars(layer).items():
            if isinstance(value, (dict, list)):
                setattr(layer, name, copy.copy(value))
        layer.reorder_cache(rows)
        new.layers.append(layer)
    return new


def score_batch(model, prompts: list[list[int]], tokenizer: Any,
                ks: list[int], device: str, budget: int = 32768,
                max_rows: int = 64) -> list[tuple]:
    """score() for several prompts: (s, root legal mass, decoded) per prompt.

    One left-padded prefill of all prompts (positions skip the padding).
    Then, per trie depth d, every node of that depth of every prompt runs
    on top of a copy of its prompt's cache, as one batch of d-token inputs,
    in chunks of at most budget // (longest prompt) and max_rows nodes (each
    copy also holds Qwen3.5's fixed-size linear-attention states).
    """
    n = max(map(len, prompts))
    ids = torch.full((len(prompts), n), tokenizer.pad_token_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, p in enumerate(prompts):
        ids[i, n - len(p):] = torch.tensor(p)
        mask[i, n - len(p):] = 1
    ids, mask = ids.to(device), mask.to(device)                         # [B, n]
    out = model(input_ids=ids, attention_mask=mask,
                position_ids=(mask.cumsum(-1) - 1).clamp(min=0),
                use_cache=True, logits_to_keep=1)
    root = torch.log_softmax(out.logits[:, -1].float(), -1)             # [B, V]
    logp = [{(): lp} for lp in root]
    tries = [candidates(tokenizer, k) for k in ks]
    depths: dict[int, list] = {}
    for i, (_, nodes) in enumerate(tries):
        for node in nodes:
            if node:
                depths.setdefault(len(node), []).append((i, node))
    length = mask.sum(-1)                                               # [B]
    step = max(1, min(max_rows, budget // n))
    for d, items in sorted(depths.items()):
        for c in range(0, len(items), step):
            chunk = items[c : c + step]
            rows = torch.tensor([i for i, _ in chunk], device=device)   # [M]
            o = model(
                input_ids=torch.tensor([node for _, node in chunk], device=device),
                attention_mask=torch.cat([mask[rows], mask.new_ones(len(chunk), d)], -1),
                position_ids=length[rows, None] + torch.arange(d, device=device),
                past_key_values=_select_cache(out.past_key_values, rows),
                use_cache=True, logits_to_keep=1)
            for (i, node), lp in zip(chunk, torch.log_softmax(o.logits[:, -1].float(), -1)):
                logp[i][node] = lp
    return [read_trie(lp, suffixes, nodes) for lp, (suffixes, nodes) in zip(logp, tries)]


def read_trie(lp: dict[tuple[int, ...], torch.Tensor], suffixes: list[list[int]],
              nodes: dict[tuple[int, ...], list[int]]) -> tuple:
    """(s, root legal mass, decoded) from each trie node's next-token
    log-probabilities lp[node] [V]: s_k sums the log-probabilities along
    suffix k (Eq. 1); decoded takes the most likely legal token at each node
    from the root until the closing "]"."""
    s = torch.stack([sum(lp[tuple(x[:t])][x[t]] for t in range(len(x))) for x in suffixes])
    node: tuple = ()
    while node in nodes:  # greedy over the legal tokens, down to a leaf
        node += (max(nodes[node], key=lambda t: lp[node][t].item()),)
    return s, lp[()][nodes[()]].exp().sum().item(), suffixes.index(list(node))


def score(model, prompt: list[int], tokenizer: Any, k: int, device: str):
    """s_k (Eq. 1) for each candidate, the root's legal mass, and the answer
    decoded inside the trie.

    One prefill of the prompt; its cache is then copied for each depth of
    the trie, and all nodes of that depth run as one batch on top of it
    (1-3 short steps, App. C). The decoded answer takes the most likely
    legal token at each node from the root until the closing "]", so it is
    always a valid "k]" (greedy within the trie; nothing after "]").
    """
    return score_batch(model, [prompt], tokenizer, [k], device)[0]
