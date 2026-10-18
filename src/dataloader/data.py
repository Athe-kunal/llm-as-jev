"""Loading decision rows from disk or from the Hugging Face Hub.

A row: {id, source, parts: [{name, text}], question, labels: {answer id:
option text}, label (the gold answer id), spans: [{part, start, end, text}],
...} (docs/data.md).

load_rows(source, split) reads, in this order:
  - a .jsonl file (split ignored);
  - a directory holding <split>.jsonl;
  - otherwise a Hugging Face dataset id, via datasets.load_dataset.

On the Hub, a mapping with different keys per row (labels) cannot be one
Arrow column, so labels may also be stored as a list of {"id", "text"}; Hub
structs pad absent keys with None. Both are turned back into a dict here.
"""

import json
import pathlib
from typing import Any

from loguru import logger

TRAIN_EVAL = "data/splits/reason_long_12k"  # local copy; or a Hub dataset id


def _labels(labels: Any) -> dict[str, str]:
    if isinstance(labels, list):
        return {x["id"]: x["text"] for x in labels}
    return {k: v for k, v in labels.items() if v is not None}


def load_rows(source: str = TRAIN_EVAL, split: str = "train") -> list[dict]:
    """The rows of one split, from a local file or folder or the Hub."""
    path = pathlib.Path(source)
    if path.is_dir():
        path = path / f"{split}.jsonl"
    if path.suffix == ".jsonl" and path.exists():
        with open(path) as f:
            rows = [json.loads(line) for line in f]
        logger.info("{} rows from {}", len(rows), path)
        return rows
    if pathlib.Path(source).exists():
        raise FileNotFoundError(f"no {split}.jsonl in {source}")
    import datasets

    rows = [dict(r) for r in datasets.load_dataset(source, split=split)]
    for r in rows:
        r["labels"] = _labels(r["labels"])
    logger.info("{} rows from the Hub: {} ({})", len(rows), source, split)
    return rows
