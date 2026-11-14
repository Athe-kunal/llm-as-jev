"""Seeded row selections shared by training, evaluation and the analyses."""

import collections
from collections.abc import Sequence
import json
import pathlib
import random

from llm_jev.data import schema

PARTS = ("dev", "test", "all")


def per_dataset(
    rows: Sequence[schema.Example], per: int, seed: int
) -> list[schema.Example]:
    """A fixed sample of up to `per` rows from each source dataset.

    Args:
      rows: The rows to sample from.
      per: Rows per dataset.
      seed: Seeds the draw.

    Returns:
      The sample, datasets in sorted order.
    """
    by_source: dict[str, list[schema.Example]] = collections.defaultdict(list)
    for row in rows:
        by_source[row.source].append(row)
    rng = random.Random(seed)
    out: list[schema.Example] = []
    for name in sorted(by_source):
        out += rng.sample(by_source[name], min(per, len(by_source[name])))
    return out


def select_part(
    rows: Sequence[schema.Example],
    part: str = "dev",
    per: int = 100,
    seed: int = 0,
    ids: set[str] | None = None,
) -> list[schema.Example]:
    """One part of the held-out split.

    Args:
      rows: The held-out split.
      part: "dev" is `per_dataset(rows, per, seed)`, the sample settings were
        chosen on; "test" is every other row, never used for a choice; "all"
        is every row.
      per: Rows per dataset of the dev sample.
      seed: Seeds the dev sample.
      ids: If given, keep only these rows (e.g. rows that fit a context).

    Returns:
      The selected rows.

    Raises:
      ValueError: If `part` is unknown.
    """
    if part not in PARTS:
        raise ValueError(f"part is one of {PARTS}, not {part!r}")
    dev = per_dataset(rows, per, seed)
    if part == "dev":
        chosen = dev
    elif part == "test":
        dev_ids = {r.id for r in dev}
        chosen = [r for r in rows if r.id not in dev_ids]
    else:
        chosen = list(rows)
    return chosen if ids is None else [r for r in chosen if r.id in ids]


def read_ids(path: str | pathlib.Path | None) -> set[str] | None:
    """The row ids in a JSON list file, or None without a file."""
    if path is None:
        return None
    return set(json.loads(pathlib.Path(path).read_text()))


def balanced(
    rows: Sequence[schema.Example], n: int, seed: int
) -> list[schema.Example]:
    """`n` rows taken round-robin over the datasets after a seeded shuffle."""
    by_source: dict[str, list[schema.Example]] = collections.defaultdict(list)
    for row in rows:
        by_source[row.source].append(row)
    rng = random.Random(seed)
    for name in sorted(by_source):
        rng.shuffle(by_source[name])
    out: list[schema.Example] = []
    i = 0
    while len(out) < n and any(i < len(v) for v in by_source.values()):
        out += [
            by_source[name][i]
            for name in sorted(by_source)
            if i < len(by_source[name])
        ][: n - len(out)]
        i += 1
    return out
