"""Decisions for a masked-language-model encoder: one mask per option.

The encoder counterpart of `answers`. The prompt (prompts/<Format.template>.j2)
puts the tokenizer's mask token in front of every option; option k's score is
the MLM logit of the verbalizer token (Format.yes_token) at its mask, and a softmax
over the K masks gives the decision. One forward pass, no trie.

Every experiment knob is a field of `Format`, so trying another template or
verbalizer is a change of arguments, not of code. Nothing here is specific to
one encoder: the mask and verbalizer ids come from the tokenizer passed in.
"""

import dataclasses
import functools
import random
from collections.abc import Sequence
from typing import Any

from llm_jev.dataloader import answers
from llm_jev.dataloader import schema


@dataclasses.dataclass(frozen=True)
class Format:
    """How an encoder reads a decision.

    Attributes:
      template: Prompt template name (prompts/<template>.j2). It must render
        the `mask` variable exactly once per option, in option order.
      yes_token: Verbalizer: s_k is the MLM logit of this single token at mask k.
      no_token: If set, s_k = logit(yes_token) - logit(no_token) at mask k.

    (Not `yes` / `no`: YAML reads those bare words as booleans, even as keys.)
    """

    template: str = "decision_mlm"
    yes_token: str = " yes"
    no_token: str | None = None


def _single_token(tokenizer: Any, text: str) -> int:
    """The id of `text`, which must be one token."""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) != 1:
        raise ValueError(f"verbalizer {text!r} is {len(ids)} tokens, not 1: {ids}")
    return ids[0]


@functools.cache
def verbalizer_ids(tokenizer: Any, fmt: Format) -> tuple[int, int | None]:
    """Token ids of fmt.yes_token and fmt.no_token (None if unset)."""
    no = None if fmt.no_token is None else _single_token(tokenizer, fmt.no_token)
    return _single_token(tokenizer, fmt.yes_token), no


def build_prompt(
    example: schema.Example, order: Sequence[int], tokenizer: Any, fmt: Format
) -> tuple[str, list[tuple[int, int]]]:
    """Prompt text with one mask per option, and each part's (start, end) range in it.

    No chat template: the tokenizer adds its own special tokens when encoding.
    """
    if tokenizer.mask_token is None:
        raise ValueError("the tokenizer has no mask token")
    return answers.user_message(
        example, order, fmt.template, mask=tokenizer.mask_token)


def encode(
    example: schema.Example, tokenizer: Any, fmt: Format,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    """Token ids, the K mask positions, and the gold option's index among them.

    With rng (training) the options are shuffled; without it they stay in
    release order. masks[k] is the position of option order[k]'s mask, so gold
    indexes masks.

    Raises:
      ValueError: If the prompt does not hold exactly one mask per option.
    """
    order = list(range(len(example.options)))
    if rng is not None:
        rng.shuffle(order)
    text, _ = build_prompt(example, order, tokenizer, fmt)
    ids = tokenizer(text)["input_ids"]  # with the tokenizer's special tokens
    masks = [i for i, t in enumerate(ids) if t == tokenizer.mask_token_id]
    if len(masks) != len(order):
        raise ValueError(
            f"{example.id}: {len(masks)} masks for {len(order)} options "
            f"(template {fmt.template!r})")
    return {
        "input_ids": ids,
        "masks": masks,
        "gold": order.index(example.gold),
        "n_options": len(order),
        "order": order,
    }
