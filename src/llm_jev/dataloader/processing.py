"""Pipeline steps over a `datasets.Dataset` of examples.

`compute_features` is the one text-heavy step: a stateless batched function
for `Dataset.map` that adds the FEATURE_COLUMNS to each example independently.
Everything after it selects rows from those columns (duplicates, validation
overlap, contamination, prompt length); take_eval_indices and sample_indices
draw seeded subsets.
"""

import dataclasses
import random
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from llm_jev.dataloader import answers
from llm_jev.dataloader import contamination
from llm_jev.dataloader import normalize
from llm_jev.dataloader import schema
from llm_jev.dataloader import sources
from llm_jev.dataloader.sources import base

FEATURE_COLUMNS = (
    "input_key", "n_tokens", "ngram_mask", "sentence_mask", "stratum",
    "answer")


@dataclasses.dataclass(frozen=True)
class FeatureContext:
    """The read-only objects `compute_features` needs (tokenizer, overlap index)."""

    tokenizer: Any
    index: contamination.OverlapIndex


# Set in the parent before `Dataset.map`; forked workers inherit it, so the index is not pickled.
_CONTEXT: FeatureContext | None = None


def set_context(context: FeatureContext) -> None:
    """Sets the objects `compute_features` uses (call before mapping)."""
    global _CONTEXT  # pylint: disable=global-statement
    _CONTEXT = context


def compute_features(batch: dict[str, list]) -> dict[str, list]:
    """The FEATURE_COLUMNS of a batch of dataset rows (for `Dataset.map`).

    Raises:
      RuntimeError: If `set_context` was not called.
    """
    if _CONTEXT is None:
        raise RuntimeError("call processing.set_context before mapping")
    size = len(batch["id"])
    examples = [
        schema.Example.from_row({k: v[i] for k, v in batch.items()})
        for i in range(size)]
    texts = [answers.build_prompt(e, range(len(e.options)),
                                  _CONTEXT.tokenizer)[0] for e in examples]
    encoded = _CONTEXT.tokenizer(texts, add_special_tokens=False)
    hits = [_CONTEXT.index.example_hits(e) for e in examples]
    strata = []
    for e in examples:
        stratification = sources.get_source(e.source).stratification
        strata.append(stratification.stratum(e) if stratification else "")
    return {
        "input_key": [normalize.input_key(e) for e in examples],
        "n_tokens": [len(ids) for ids in encoded["input_ids"]],
        "ngram_mask": [ngram for ngram, _ in hits],
        "sentence_mask": [sentence for _, sentence in hits],
        "stratum": strata,
        "answer": [e.answer for e in examples],
    }


def dedupe_indices(
    keys: Sequence[str], answers: Sequence[str]
) -> tuple[list[int], int, int]:
    """Rows to keep when exact duplicate inputs are collapsed.

    Copies that agree on the gold option's text collapse to one, even if their
    options are ordered differently. Inputs whose copies disagree on the answer
    are dropped entirely.

    Returns:
      (indices to keep in order, number of dropped copies, number of rows
      dropped for conflicting answers).
    """
    frame = pd.DataFrame({"key": list(keys), "answer": list(answers)})
    first = ~frame["key"].duplicated()
    conflict = frame.groupby("key")["answer"].transform("nunique") > 1
    kept = frame.index[first & ~conflict].tolist()
    return kept, int((~first & ~conflict).sum()), int(conflict.sum())


def take_eval_indices(
    groups: Sequence[str], subsets: Sequence[str], cap: int,
    rng: random.Random
) -> list[int]:
    """Up to cap rows in whole groups, round-robin over subsets.

    Groups are taken per subset in a shuffled order; a group is only taken
    while it fits.
    """
    members: dict[str, list[int]] = {}
    for i, group in enumerate(groups):
        members.setdefault(group, []).append(i)
    by_subset: dict[str, list[list[int]]] = {}
    for group in sorted(members):
        rows = members[group]
        counts: dict[str, int] = {}
        for i in rows:
            counts[subsets[i]] = counts.get(subsets[i], 0) + 1
        subset = max(counts, key=counts.get)  # first most common
        by_subset.setdefault(subset, []).append(rows)
    for subset in by_subset:
        rng.shuffle(by_subset[subset])
    taken: list[int] = []
    order = sorted(by_subset)
    while any(by_subset[s] for s in order):
        for s in order:
            while by_subset[s] and len(taken) + len(by_subset[s][-1]) > cap:
                by_subset[s].pop()  # too big to fit: skip
            if by_subset[s]:
                taken += by_subset[s].pop()
    return taken


def sample_indices(
    strata: Sequence[str], n: int | None, rng: random.Random,
    shares: base.Shares | None, name: str = "dataset"
) -> list[int]:
    """n shuffled row indices: a uniform draw, or by stratum shares (see base.Shares).

    Raises:
      ValueError: If the pool has fewer than n rows.
    """
    if n is None:
        n = len(strata)
    if len(strata) < n:
        raise ValueError(f"{name}: only {len(strata)} clean examples for {n}")
    if shares is None:
        out = rng.sample(range(len(strata)), n)
    else:
        by_stratum: dict[str, list[int]] = {}
        for i, stratum in enumerate(strata):
            by_stratum.setdefault(stratum, []).append(i)
        out = []
        for stratum, count in _quotas(sorted(by_stratum), n, shares).items():
            out += rng.sample(by_stratum.get(stratum, []), count)
    rng.shuffle(out)
    return out


def _quotas(
    present: list[str], n: int, shares: base.Shares
) -> dict[str, int]:
    """How many rows each stratum gets, in draw order."""
    if shares == "equal":
        return {s: n // len(present) + (i < n % len(present))
                for i, s in enumerate(present)}
    assert isinstance(shares, Mapping)
    names = list(shares)
    counts = [round(shares[s] * n) for s in names[:-1]]
    return dict(zip(names, counts + [n - sum(counts)], strict=True))


def length_stats(lengths: Sequence[int]) -> dict[str, int]:
    """Median, 95th percentile, max and min of the lengths ({} if empty)."""
    if not lengths:
        return {}
    ordered = sorted(lengths)
    return {
        "median": int(statistics.median(ordered)),
        "p95": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "max": ordered[-1],
        "min": ordered[0],
    }
