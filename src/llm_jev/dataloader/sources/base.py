"""The source registry and the helpers every source uses.

A source reads one dataset's raw files and returns its official train and
validation splits as `Example`s. Sources only parse: they do not
deduplicate, filter or sample (that is `processing`).
"""

import dataclasses
import hashlib
from collections.abc import Callable, Iterable, Mapping
from typing import Any, Literal

from llm_jev.dataloader import schema

SEED = 20260922  # the paper's training seed (Table 2)

# {"train": [...], "validation": [...]}: the official train split and the
# official validation / dev split (labelled test splits are not used).
Split = dict[str, list[schema.Example]]

DEFAULT_SPLITS = {"train": "train", "validation": "validation"}

# How many training examples each stratum gets: a share per stratum (the
# last one takes the remainder), or "equal" for equal shares.
Shares = Mapping[str, float] | Literal["equal"]


@dataclasses.dataclass(frozen=True)
class Stratification:
    """How a source's training examples are drawn: by stratum, not uniformly.

    Attributes:
      stratum: A pure function giving an example's stratum.
      shares: How many training examples each stratum gets.
    """

    stratum: Callable[[schema.Example], str]
    shares: Shares


@dataclasses.dataclass(frozen=True)
class Source:
    """A registered dataset.

    Attributes:
      name: Registry key; also the `source` of its examples.
      load: Reads the raw files and returns the splits.
      splits: Which official split of the release each side is read from,
        for the manifest.
      stratification: How training examples are drawn, or None for a uniform
        draw.
    """

    name: str
    load: Callable[[], Split]
    splits: Mapping[str, str]
    stratification: Stratification | None = None


SOURCES: dict[str, Source] = {}


def register(
    name: str,
    splits: Mapping[str, str] | None = None,
    stratification: Stratification | None = None,
) -> Callable[[Callable[[], Split]], Callable[[], Split]]:
    """Registers a loader function as the source `name`.

    Args:
      name: The dataset name.
      splits: Which official split each side is read from; defaults to
        train / validation.
      stratification: The source's own way of drawing training examples, if
        it has one.

    Returns:
      A decorator that registers the function and returns it unchanged.
    """

    def decorator(load: Callable[[], Split]) -> Callable[[], Split]:
        if name in SOURCES:
            raise ValueError(f"source {name!r} is already registered")
        SOURCES[name] = Source(
            name, load, splits or DEFAULT_SPLITS, stratification)
        return load

    return decorator


def get_source(name: str) -> Source:
    """The registered source `name`.

    Raises:
      KeyError: If no such source is registered.
    """
    try:
        return SOURCES[name]
    except KeyError:
        raise KeyError(
            f"unknown source {name!r}; known: {sorted(SOURCES)}") from None


def stable_hash(*parts: Any) -> str:
    """A short, stable hash of the parts (ids and groups)."""
    return hashlib.sha256(
        "\x1f".join(map(str, parts)).encode()).hexdigest()[:16]


def make_example(
    source: str,
    key: Any,
    subset: str,
    group: Any,
    parts: list[tuple[str, str]],
    question: str,
    options: Iterable[str],
    gold: int,
    spans: Iterable[tuple[str, int, int]] = (),
    overlap_text: Iterable[str] | None = None,
) -> schema.Example:
    """One example, with a stable id and group derived from the raw keys.

    Args:
      source: Dataset name.
      key: Identifies the row inside the dataset (with the question, it
        makes the id).
      subset: Stratification key, usually the gold label's name.
      group: Raw identifier of the leakage unit.
      parts: (name, text) pairs in prompt order.
      question: The question text.
      options: The answer options in release order.
      gold: 0-based index of the correct option.
      spans: (part name, start, end) triples; duplicates are collapsed and
        those that do not fit their part are dropped.
      overlap_text: Source texts the benchmark-overlap filter checks, when
        they differ from parts + question + options.

    Returns:
      The example.
    """
    texts = dict(parts)
    names = [name for name, _ in parts]
    located = sorted(
        {(p, a, b) for p, a, b in spans if 0 <= a < b <= len(texts[p])})
    return schema.Example(
        id=f"{source}:{stable_hash(key, question)}",
        source=source,
        parts=tuple(schema.Part(name, text) for name, text in parts),
        question=question,
        options=tuple(options),
        gold=gold,
        spans=tuple(schema.Span(names.index(p), a, b) for p, a, b in located),
        group=f"{source}:{stable_hash(group)}",
        subset=subset,
        overlap_text=None if overlap_text is None else tuple(overlap_text),
    )
