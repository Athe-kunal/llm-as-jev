"""Examples as a `datasets.Dataset`: the container the pipeline maps over.

A Dataset has one row per example, in the schema of `FEATURES` (the dict form
of `Example.to_row`). It is saved as Parquet, which is also what a Hugging
Face Hub dataset holds, so a built dataset can be pushed as it is.

`load_dataset(source, split)` reads, in this order:
  - a .parquet file (the split is ignored);
  - a directory holding <split>.parquet;
  - otherwise a Hugging Face dataset id, via `datasets.load_dataset`.
"""

import pathlib
from collections.abc import Iterable

import datasets
from loguru import logger

from llm_jev.dataloader import schema

DEFAULT_SOURCE = "data/splits/reason_long_12k_v2"  # local copy or a Hub id

FEATURES = datasets.Features({
    "id": datasets.Value("string"),
    "source": datasets.Value("string"),
    "parts": [{"name": datasets.Value("string"),
               "text": datasets.Value("string")}],
    "question": datasets.Value("string"),
    "options": [datasets.Value("string")],
    "gold": datasets.Value("int64"),
    "spans": [{"part": datasets.Value("int64"),
               "start": datasets.Value("int64"),
               "end": datasets.Value("int64")}],
    "group": datasets.Value("string"),
    "subset": datasets.Value("string"),
    "meta": datasets.Value("string"),  # JSON object
})


def to_dataset(
    examples: Iterable[schema.Example], with_overlap_text: bool = False
) -> datasets.Dataset:
    """A Dataset with one row per example.

    Args:
      examples: The examples.
      with_overlap_text: Whether to add the `overlap_text` working column
        (a list of strings, empty for the default) after the schema columns.

    Returns:
      The Dataset, in the schema of `FEATURES` (plus the working column).
    """
    features = FEATURES.copy()
    if with_overlap_text:
        features["overlap_text"] = [datasets.Value("string")]
    return datasets.Dataset.from_list(
        [e.to_row(with_overlap_text) for e in examples], features=features)


def to_examples(dataset: datasets.Dataset) -> list[schema.Example]:
    """The examples of a Dataset (validated)."""
    return [schema.Example.from_dict(row) for row in dataset]


def save(dataset: datasets.Dataset, path: str | pathlib.Path) -> None:
    """Writes a Dataset as one Parquet file, creating the parent folder."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(str(path))


def load_dataset(
    source: str = DEFAULT_SOURCE, split: str = "train"
) -> datasets.Dataset:
    """One split as a Dataset, from a local file or folder or the Hub.

    Args:
      source: A .parquet file, a folder holding <split>.parquet, or a
        Hugging Face dataset id.
      split: The split to read (ignored for a .parquet file).

    Returns:
      The Dataset, in the schema of `FEATURES`.

    Raises:
      FileNotFoundError: If `source` is a local folder without <split>.parquet.
    """
    path = pathlib.Path(source)
    if path.is_dir():
        path = path / f"{split}.parquet"
    if path.suffix == ".parquet" and path.exists():
        dataset = datasets.Dataset.from_parquet(str(path))
        logger.info("{} examples from {}", len(dataset), path)
        return dataset
    if pathlib.Path(source).exists():
        raise FileNotFoundError(f"no {split}.parquet in {source}")
    dataset = datasets.load_dataset(source, split=split)
    logger.info("{} examples from the Hub: {} ({})", len(dataset), source, split)
    return dataset


def load_examples(
    source: str = DEFAULT_SOURCE, split: str = "train"
) -> list[schema.Example]:
    """`load_dataset` as validated `Example`s."""
    return to_examples(load_dataset(source, split))
