"""Sentences as units of evidence: splitting, ablating and matching to spans.

Sentence splitting follows ContextCite (Cohen-Wang et al., NeurIPS 2024;
github.com/MadryLab/context-cite): the nltk sentences of each part, with
content-free fragments merged into a neighbour so the text is unchanged.
"""

import collections
from collections.abc import Sequence
import dataclasses
import functools
import random
import re

import nltk
import numpy as np

from llm_jev.data import sampling
from llm_jev.data import schema

_WORD = re.compile(r"[^\W\d_]{2,}")


@dataclasses.dataclass(frozen=True)
class Sentence:
    """A sentence of one part.

    Attributes:
      part: Index into `Example.parts`.
      start: Character offset in that part's text (as `schema.Span`).
      end: Exclusive end offset.
      sep: The text between the previous sentence of the part and this one.
    """

    part: int
    start: int
    end: int
    sep: str = ""

    @property
    def char_range(self) -> tuple[int, int, int]:
        """(part, start, end)."""
        return (self.part, self.start, self.end)

    def overlaps(self, span: schema.Span) -> bool:
        """Whether the sentence overlaps a span."""
        return (
            span.part == self.part
            and self.start < span.end
            and self.end > span.start
        )


@functools.cache
def _punkt() -> None:
    """Fetches nltk's sentence model once."""
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)


def has_content(text: str, min_words: int = 2, min_alnum: float = 0.5) -> bool:
    """Whether a sentence has enough words and is mostly letters or digits.

    Blank form fields, checkbox lines and abbreviation fragments do not.
    """
    alnum = sum(ch.isalnum() for ch in text) / max(len(text), 1)
    return len(_WORD.findall(text)) >= min_words and alnum >= min_alnum


def split(
    example: schema.Example, min_words: int = 2, min_alnum: float = 0.5
) -> list[Sentence]:
    """The sentences of every part, in order, each with content.

    Raises:
      ValueError: If nltk returns a sentence that is not in the text.
    """
    _punkt()
    out: list[Sentence] = []
    for i, part in enumerate(example.parts):
        text, cursor = part.text, 0
        own: list[Sentence] = []
        lead: Sentence | None = None
        for line in text.splitlines():
            for sentence in nltk.sent_tokenize(line):
                start = text.find(sentence, cursor)
                if start < 0:
                    raise ValueError(
                        f"{example.id}: sentence not found in part {i}"
                    )
                current = Sentence(
                    i, start, start + len(sentence), text[cursor:start]
                )
                cursor = current.end
                if not has_content(sentence, min_words, min_alnum):
                    if own:
                        own[-1] = dataclasses.replace(own[-1], end=current.end)
                    elif lead is not None:
                        lead = dataclasses.replace(lead, end=current.end)
                    else:
                        lead = current
                    continue
                if lead is not None:
                    current = dataclasses.replace(
                        current, start=lead.start, sep=lead.sep
                    )
                    lead = None
                own.append(current)
        out += own or ([lead] if lead is not None else [])
    return out


def ablate(
    example: schema.Example, sentences: Sequence[Sentence], keep: Sequence[bool]
) -> schema.Example:
    """The example with only the kept sentences (ContextCite's get_context).

    Text outside a part's sentences is always kept, so keeping every
    sentence rebuilds the example exactly. Spans are dropped.
    """
    kept: list[list[str]] = [[] for _ in example.parts]
    for s, k in zip(sentences, keep, strict=True):
        if k:
            text = example.parts[s.part].text
            sep = s.sep if kept[s.part] else ""
            kept[s.part].append(sep + text[s.start : s.end])
    parts = []
    for i, p in enumerate(example.parts):
        own = [s for s in sentences if s.part == i]
        lead = p.text[: own[0].start] if own else p.text
        tail = p.text[own[-1].end :] if own else ""
        parts.append(schema.Part(p.name, lead + "".join(kept[i]) + tail))
    return dataclasses.replace(example, parts=tuple(parts), spans=())


def ablation_masks(
    n: int, d: int, keep_prob: float = 0.5, seed: int = 0
) -> np.ndarray:
    """[n, d] bool: n random ablations of d sentences, True = kept.

    Mask i uses RandomState(seed + i), as ContextCite's reference does.
    """
    masks = np.zeros((n, d), dtype=bool)
    for i in range(n):
        rng = np.random.RandomState(seed + i)
        masks[i] = rng.choice(
            [False, True], size=d, p=[1 - keep_prob, keep_prob]
        )
    return masks


def to_spans(sentences: Sequence[Sentence]) -> tuple[schema.Span, ...]:
    """Sentences as spans of the example, in document order."""
    ordered = sorted(sentences, key=lambda s: (s.part, s.start))
    return tuple(schema.Span(s.part, s.start, s.end) for s in ordered)


def gold_indices(
    example: schema.Example, sentences: Sequence[Sentence]
) -> set[int]:
    """Indices of the sentences that overlap a gold span."""
    return {
        i
        for i, s in enumerate(sentences)
        if any(s.overlaps(span) for span in example.spans)
    }


def rows_with_evidence(
    examples: Sequence[schema.Example],
    per: int = 50,
    seed: int = 0,
    part: str | None = None,
) -> list[schema.Example]:
    """Rows with gold spans and >= 2 sentences, `per` per dataset (0: all).

    Args:
      examples: The held-out split.
      per: Rows per dataset; 0 keeps every such row.
      seed: Seeds the draw.
      part: "dev" / "test" first restricts to that part of the split
        (sampling.select_part with the default dev sample); None keeps all.
    """
    if part is not None:
        from llm_jev import config

        keep = {
            r.id for r in sampling.select_part(examples, part, 100, config.SEED)
        }
        examples = [r for r in examples if r.id in keep]
    by_source: dict[str, list[schema.Example]] = collections.defaultdict(list)
    for r in examples:
        if r.has_spans and len(split(r)) >= 2:
            by_source[r.source].append(r)
    rng = random.Random(seed)
    out: list[schema.Example] = []
    for name in sorted(by_source):
        rows = by_source[name]
        out += rows if per == 0 else rng.sample(rows, min(per, len(rows)))
    return out
