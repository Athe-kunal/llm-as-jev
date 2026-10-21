"""Reason+Long-12k: the training mixture of Li & Wagle (arXiv 2610.02076v3).

Usage (or: make data):
  # fetch the pinned sources into the Hugging Face cache (once)
  .venv/bin/python -m llm_jev.dataloader.build --download
  # build <out>/{train,eval}.jsonl and manifest.json
  .venv/bin/python -m llm_jev.dataloader.build

Twelve public human-annotated datasets, 1,000 training examples each (Table
3 of the paper): CLINC150, MASSIVE (en-US), Bitext customer support,
ReClor, LogiQA 2.0, CosmosQA ("none of the above" gold upsampled to 16%),
PIQA, alpha-NLI, ProofWriter, ContractNLI, MuSiQue (answerability over 20
paragraphs) and QuALITY. The labels are the datasets' own gold answers.
Native evidence spans are kept where the release has them: ContractNLI
(human spans), MuSiQue (supporting paragraphs of answerable questions) and
ProofWriter (the facts and rules of the gold proof). Every source is a
registered `Example` source (`dataloader.sources`).

Cleaning, applied to each training pool before sampling (paper section 3.5,
Appendix D): examples sharing a 13-word n-gram, or an identical sentence of
six or more words, with any evaluation benchmark are dropped
(`contamination.decontaminate`); examples whose input also appears in the
dataset's official validation split are dropped; exact duplicate inputs
within the pool are collapsed. Training examples come from each dataset's
official train split, eval examples (up to --eval-rows per dataset, whole
groups) from its official validation split; Bitext has no split, so a
seeded group hold-out stands in for one. Examples whose prompt
(`answers.build_prompt`, the paper's prompt, with the LENGTH_TOKENIZER
tokenizer) exceeds --max-tokens are left out, never truncated.

Processing is a `Dataset.map` (the per-example features, parallel over
--num-proc workers) followed by column selections (`processing`); the result
is saved as Parquet, ready for training or `push_to_hub`.
"""

import argparse
import collections
import json
import os
import pathlib
import random
from collections.abc import Iterable
from typing import Any

import datasets
import numpy as np
from loguru import logger

from llm_jev.dataloader import contamination
from llm_jev.dataloader import normalize
from llm_jev.dataloader import processing
from llm_jev.dataloader import sources
from llm_jev.dataloader import storage
from llm_jev.dataloader.sources import base
from llm_jev.dataloader.sources import download

PER_DATASET = 1000
EVAL_ROWS = 500  # default; change with --eval-rows
MAX_TOKENS = 10240  # Table 2: 10,240 for Reason+Long-12k, never truncated
OUT = "data/splits/reason_long_12k_v2"
LENGTH_TOKENIZER = "Qwen/Qwen3.5-4B"  # the paper's model; lengths only
# What exact-duplicate inputs must agree on to keep one copy (column name).
DEDUPE_CONFLICT = ("gold", "answer")
REASON_LONG_12K = (
    "clinc150", "massive", "bitext", "reclor", "logiqa2", "cosmosqa", "piqa",
    "anli", "proofwriter", "contractnli", "musique", "quality")


def _featurize(examples: list, num_proc: int) -> datasets.Dataset:
    """The examples as a Dataset with the feature columns (parallel map)."""
    return storage.to_dataset(examples, with_overlap_text=True).map(
        processing.compute_features, batched=True, batch_size=256,
        num_proc=num_proc, load_from_cache_file=False)


def _where(mask: np.ndarray) -> list[int]:
    """The indices where the mask holds."""
    return np.flatnonzero(mask).tolist()


def build_dataset(
    source: base.Source,
    index: contamination.OverlapIndex,
    eval_rows: int,
    max_tokens: int,
    num_proc: int,
    dedupe_conflict: str = "gold",
) -> tuple[datasets.Dataset, datasets.Dataset, dict[str, Any]]:
    """Cleans one dataset and draws its training and eval examples.

    `processing.set_context` must have been called.

    Args:
      source: The dataset.
      index: The benchmarks' overlap index (for the manifest).
      eval_rows: The most eval examples to take.
      max_tokens: Examples with longer prompts are left out.
      num_proc: Worker processes of the feature map.
      dedupe_conflict: The column duplicate inputs must agree on ("gold":
        the answer's position, as in the original build; "answer": its
        text).

    Returns:
      (training examples, eval examples, the dataset's manifest entry); the
      Datasets are in the schema of `storage.FEATURES`.
    """
    rng = random.Random(f"{base.SEED}:{source.name}")
    data = source.load()
    train = _featurize(data["train"], num_proc)
    validation = _featurize(data["validation"], num_proc)
    pinned = download.TRAIN_SOURCES.get(source.name, {})
    stats: dict[str, Any] = {
        "source_split": dict(source.splits),
        "revision": pinned.get("revision"),
        "source": pinned.get("repo") or list(pinned.get("files", {}).values()),
        "train_split_rows": len(train),
        "validation_split_rows": len(validation),
    }
    # eval: deduplicated validation examples that fit, whole groups
    kept, _, _ = processing.dedupe_indices(
        validation["input_key"], validation[dedupe_conflict])
    deduped = validation.select(kept)
    fit = deduped.select(_where(np.asarray(deduped["n_tokens"]) <= max_tokens))
    eval_set = fit.select(processing.take_eval_indices(
        fit["group"], fit["subset"], eval_rows, rng))
    stats["validation_too_long"] = len(deduped) - len(fit)
    # train pool, cleaned in order; each example counted at its first drop
    kept, copies, conflict = processing.dedupe_indices(
        train["input_key"], train[dedupe_conflict])
    pool = train.select(kept)
    stats["drop_duplicate_copies"] = copies
    stats["drop_conflicting_duplicates"] = conflict
    before = len(pool)
    pool = pool.select(_where(~np.isin(
        pool["input_key"], list(set(validation["input_key"])))))
    stats["drop_input_in_validation"] = before - len(pool)
    before = len(pool)
    pool = pool.select(_where(~np.isin(
        pool["group"], list(set(validation["group"])))))
    stats["drop_group_in_validation"] = before - len(pool)
    stats.update(contamination.contamination_stats(
        pool["id"], pool["ngram_mask"], pool["sentence_mask"], index))
    pool = pool.select(_where(
        (np.asarray(pool["ngram_mask"]) == 0)
        & (np.asarray(pool["sentence_mask"]) == 0)))
    too_long = np.asarray(pool["n_tokens"]) > max_tokens
    stats["drop_too_long"] = int(too_long.sum())
    clean = pool.select(_where(~too_long))
    stats["clean_pool"] = len(clean)
    stats["clean_pool_labels"] = dict(
        sorted(collections.Counter(clean["subset"]).items()))
    train_set = clean.select(processing.sample_indices(
        clean["stratum"], PER_DATASET, rng, source.shares, source.name))
    stats.update({
        "train": len(train_set),
        "eval": len(eval_set),
        "train_labels": dict(
            sorted(collections.Counter(train_set["subset"]).items())),
        "eval_labels": dict(
            sorted(collections.Counter(eval_set["subset"]).items())),
        "train_with_spans": sum(bool(s) for s in train_set["spans"]),
        "eval_with_spans": sum(bool(s) for s in eval_set["spans"]),
        "options": len(train_set[0]["options"]),
        "train_tokens": processing.length_stats(train_set["n_tokens"]),
        "eval_tokens": processing.length_stats(eval_set["n_tokens"]),
        "clean_pool_tokens": processing.length_stats(clean["n_tokens"]),
    })
    if source.stratum:
        stats["train_strata"] = dict(
            sorted(collections.Counter(train_set["stratum"]).items()))
        stats["clean_pool_strata"] = dict(
            sorted(collections.Counter(clean["stratum"]).items()))
    columns = [*processing.FEATURE_COLUMNS, "overlap_text"]
    return (train_set.remove_columns(columns), eval_set.remove_columns(columns),
            stats)


def build(
    out: pathlib.Path,
    eval_rows: int = EVAL_ROWS,
    max_tokens: int = MAX_TOKENS,
    names: Iterable[str] | None = None,
    num_proc: int = 1,
    dedupe_conflict: str = "gold",
) -> dict[str, Any]:
    """Writes train.parquet, eval.parquet and manifest.json under out.

    Args:
      out: The output folder.
      eval_rows: The most eval examples per dataset.
      max_tokens: Examples with longer prompts are left out.
      names: The datasets to build; defaults to the twelve of Reason+Long.
      num_proc: Worker processes of the feature map.
      dedupe_conflict: See `build_dataset`.

    Returns:
      The manifest.
    """
    from transformers import AutoTokenizer  # pylint: disable=C0415

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # we fork
    tokenizer = AutoTokenizer.from_pretrained(LENGTH_TOKENIZER)
    logger.info("indexing benchmarks ...")
    index = contamination.OverlapIndex(contamination.benchmark_items())
    processing.set_context(processing.FeatureContext(tokenizer, index))
    manifest: dict[str, Any] = {
        "paper": "Li & Wagle, LLM-as-Jev, arXiv 2610.02076v3, Reason+Long-12k",
        "seed": base.SEED, "per_dataset": PER_DATASET,
        "eval_rows": eval_rows, "max_tokens": max_tokens,
        "dedupe_conflict": dedupe_conflict,
        "prompt": "answers.build_prompt", "tokenizer": LENGTH_TOKENIZER,
        "normalization": {"ngram": normalize.NGRAM,
                          "min_sentence_words": normalize.MIN_SENTENCE_WORDS},
        "benchmarks": {
            name: {
                "items": index.sizes[name],
                "revision": download.BENCH_SOURCES[name]["revision"],
                "source": download.BENCH_SOURCES[name].get("repo")
                or list(download.BENCH_SOURCES[name]["files"].values()),
            }
            for name in index.names},
        "benchmark_suites": {k: list(v) for k, v in download.SUITES.items()},
        "datasets": {},
    }
    train_parts, eval_parts = [], []
    for name in names or REASON_LONG_12K:
        logger.info("{} ...", name)
        train_set, eval_set, stats = build_dataset(
            sources.get_source(name), index, eval_rows, max_tokens, num_proc,
            dedupe_conflict)
        manifest["datasets"][name] = stats
        train_parts.append(train_set)
        eval_parts.append(eval_set)
        logger.info(json.dumps({k: v for k, v in stats.items()
                                if not k.startswith("contaminated_examples")}))
    train = datasets.concatenate_datasets(train_parts)
    order = list(range(len(train)))
    random.Random(base.SEED).shuffle(order)
    train = train.select(order)
    evals = datasets.concatenate_datasets(eval_parts)
    storage.save(train, out / "train.parquet")
    storage.save(evals, out / "eval.parquet")
    manifest["totals"] = {"train": len(train), "eval": len(evals)}
    manifest["checks"] = {
        "id_overlap": len(set(train["id"]) & set(evals["id"])),
        "group_overlap": len(set(train["group"]) & set(evals["group"])),
        "duplicate_train_ids": len(train) - len(set(train["id"])),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    logger.info("{} {}", json.dumps(manifest["totals"]),
                json.dumps(manifest["checks"]))
    return manifest


def main() -> None:
    """The command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--download", action="store_true",
                        help="fetch the pinned sources and stop")
    parser.add_argument("--out", default=OUT)
    parser.add_argument("--eval-rows", type=int, default=EVAL_ROWS)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--dedupe-conflict", choices=DEDUPE_CONFLICT,
                        default="gold",
                        help="what exact-duplicate inputs must agree on to "
                        "keep one copy: the answer's position (gold, the "
                        "original build) or its text (answer)")
    parser.add_argument("--num-proc", type=int,
                        default=min(8, os.cpu_count() or 1),
                        help="worker processes of the feature map")
    parser.add_argument("--datasets", nargs="*", default=None,
                        choices=sorted(sources.SOURCES),
                        help="subset of datasets (default: the twelve of "
                        "Reason+Long-12k)")
    args = parser.parse_args()
    if args.download:
        download.download()
        return
    build(pathlib.Path(args.out), args.eval_rows, args.max_tokens,
          args.datasets, args.num_proc, args.dedupe_conflict)


if __name__ == "__main__":
    main()
