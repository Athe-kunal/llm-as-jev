"""Bespoke Labs' Nimble (github.com/bespokelabsai/nimble, data/).

Every task is a state (the document), one decision question and its options.
The files come from data/train.jsonl and data/eval.jsonl of the pinned
commit (download.TRAIN_SOURCES["nimble"]), downloaded on first use; if they
cannot be fetched, loading raises FileNotFoundError.

Options are positional like every other source, so Nimble answers go through
the same "Best answer: [k]" trie. Upstream's "eval" file is the validation
split. The evidence sentences of the task's `verified_pair` become its spans.
"""

import json
from typing import Any

from llm_jev.dataloader import schema
from llm_jev.dataloader.sources import base
from llm_jev.dataloader.sources import download

# The verified_pair keys of the evidence sentences, by task variant.
EVIDENCE_KEYS = {
    "base": ("left", "right"),
    "counterfactual": ("negative_left", "negative_right"),
}


def render_text(value: Any) -> str:
    """A state field as text: strings as they are, turns as "Speaker: text"."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(
            f"{v['speaker']}: {v['text']}"
            if isinstance(v, dict) and {"speaker", "text"} <= set(v)
            else render_text(v)
            for v in value)
    return json.dumps(value, ensure_ascii=False)


def state_parts(state: Any) -> list[tuple[str, str]]:
    """Document sections for a Nimble state (text, statements or fields)."""
    if isinstance(state, str):
        return [("Document", state)]
    if isinstance(state, list):
        return [("Statements", render_text(state))]
    parts = []
    for key, value in state.items():
        if key == "questions":  # a nested schema, not document text
            continue
        if key == "state" and not isinstance(value, str):
            parts += state_parts(value)
            continue
        parts.append((key.replace("_", " ").title(), render_text(value)))
    return [(name, text) for name, text in parts if text.strip()]


def options_and_gold(decision: dict[str, Any], target: Any) -> tuple[
        list[str], int]:
    """The options and the gold index of a decision.

    Args:
      decision: The task's `questions.decision` object.
      target: The task's `reference.target`.

    Returns:
      (options in release order, 0-based gold index). Choice: the criteria
      keys name the options. Noul: the criteria keys are true / false.
      Score: the criteria list holds the levels 0.., and the target is the
      level.
    """
    criteria = decision["criteria"]
    if decision["type"] == "score":
        return list(criteria), int(target)
    keys = [str(k).lower() if decision["type"] == "noul" else k
            for k in criteria]
    wanted = str(target).lower() if decision["type"] == "noul" else target
    return list(criteria.values()), keys.index(wanted)


def evidence_spans(
    task: dict[str, Any], parts: list[tuple[str, str]]
) -> tuple[schema.Span, ...]:
    """The task's own evidence sentences, as spans of its parts.

    Nimble's `verified_pair` holds the two sentences the answer rests on: a
    base task uses `left` and `right`, a counterfactual task (the same
    story with one fact changed) `negative_left` and `negative_right`. Each
    occurs verbatim, once, in one part of the state.

    Args:
      task: The raw task.
      parts: Its (name, text) parts.

    Returns:
      The two spans in document order.

    Raises:
      ValueError: If a sentence is not in any part.
    """
    pair = task["evidence_certificate"]["verified_pair"]
    spans = []
    for key in EVIDENCE_KEYS[task["variant"]]:
        for i, (_, text) in enumerate(parts):
            start = text.find(pair[key])
            if start >= 0:
                spans.append(schema.Span(i, start, start + len(pair[key])))
                break
        else:
            raise ValueError(f"nimble:{task['id']}: evidence {key!r} is not "
                             "in the state")
    return tuple(sorted(spans, key=lambda s: (s.part, s.start)))


def nimble_example(task: dict[str, Any]) -> schema.Example:
    """One raw Nimble task as an `Example`."""
    decision = task["input"]["questions"]["decision"]
    options, gold = options_and_gold(decision, task["reference"]["target"])
    parts = state_parts(task["input"]["state"])
    return schema.Example(
        id=f"nimble:{task['id']}",
        source="nimble",
        parts=tuple(schema.Part(n, t) for n, t in parts),
        question=decision["instructions"],
        options=tuple(options),
        gold=gold,
        spans=evidence_spans(task, parts),
        group=f"nimble:{task['family']}",
        subset=decision["type"],
        meta={k: task[k] for k in ("domain", "variant", "source_family",
                                   "method") if k in task},
    )


@base.register("nimble", splits={"train": "train", "validation": "eval"})
def nimble() -> base.Split:
    """Nimble train / eval (the eval file is the validation split)."""
    out = {}
    for split, filename in (("train", "train.jsonl"),
                            ("validation", "eval.jsonl")):
        with open(download.local_path("nimble", filename)) as f:
            out[split] = [nimble_example(json.loads(line))
                          for line in f if line.strip()]
    return out
