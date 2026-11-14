"""Builds the training mix: Reason+Long-12k (arXiv 2610.02076v3) plus Nimble.

Usage:
  llm-jev-build-data download_only=true     # fetch the pinned sources
  llm-jev-build-data --config configs/data/reason_long_nimble.yaml

Every dataset is deduplicated, decontaminated against the evaluation
benchmarks, length-filtered and sampled; writes train.parquet, eval.parquet
and manifest.json.
"""

import collections
import json
import os
import pathlib
import random
from typing import Any

import datasets
from loguru import logger
import numpy as np

from llm_jev import config
from llm_jev.data import contamination
from llm_jev.data import normalize
from llm_jev.data import processing
from llm_jev.data import sources
from llm_jev.data import storage
from llm_jev.data.sources import base
from llm_jev.data.sources import download

DATASETS = (
    "clinc150",
    "massive",
    "bitext",
    "reclor",
    "logiqa2",
    "cosmosqa",
    "piqa",
    "anli",
    "proofwriter",
    "contractnli",
    "musique",
    "quality",
    "nimble",
)


def _featurize(examples: list, num_proc: int) -> datasets.Dataset:
    """The examples as a Dataset with the feature columns (parallel map)."""
    return storage.to_dataset(examples, with_overlap_text=True).map(
        processing.compute_features,
        batched=True,
        batch_size=256,
        num_proc=num_proc,
        load_from_cache_file=False,
    )


def _where(mask: np.ndarray) -> list[int]:
    """The indices where the mask holds."""
    return np.flatnonzero(mask).tolist()


def build_dataset(
    source: base.Source,
    index: contamination.OverlapIndex,
    eval_rows: int,
    max_tokens: int,
    num_proc: int,
    per_dataset: int | None,
) -> tuple[datasets.Dataset, datasets.Dataset, dict[str, Any]]:
    """Cleans one dataset and draws its training and eval examples.

    `processing.set_context` must have been called.
    """
    rng = random.Random(f"{base.SEED}:{source.name}")
    data = source.load()
    train = _featurize(data["train"], num_proc)
    validation = _featurize(data["validation"], num_proc)
    stats: dict[str, Any] = {
        "source_split": dict(source.splits),
        **download.provenance(download.TRAIN_SOURCES[source.name]),
        "train_split_rows": len(train),
        "validation_split_rows": len(validation),
    }
    kept, _, _ = processing.dedupe_indices(
        validation["input_key"], validation["answer"]
    )
    deduped = validation.select(kept)
    fit = deduped.select(_where(np.asarray(deduped["n_tokens"]) <= max_tokens))
    eval_set = fit.select(
        processing.take_eval_indices(
            fit["group"], fit["subset"], eval_rows, rng
        )
    )
    stats["validation_too_long"] = len(deduped) - len(fit)
    # Each example is counted at its first drop.
    kept, copies, conflict = processing.dedupe_indices(
        train["input_key"], train["answer"]
    )
    pool = train.select(kept)
    stats["drop_duplicate_copies"] = copies
    stats["drop_conflicting_duplicates"] = conflict
    for column, key in (
        ("input_key", "drop_input_in_validation"),
        ("group", "drop_group_in_validation"),
    ):
        before = len(pool)
        pool = pool.select(
            _where(~np.isin(pool[column], list(set(validation[column]))))
        )
        stats[key] = before - len(pool)
    stats.update(
        contamination.contamination_stats(
            pool["id"], pool["ngram_mask"], pool["sentence_mask"], index
        )
    )
    pool = pool.select(
        _where(
            (np.asarray(pool["ngram_mask"]) == 0)
            & (np.asarray(pool["sentence_mask"]) == 0)
        )
    )
    too_long = np.asarray(pool["n_tokens"]) > max_tokens
    stats["drop_too_long"] = int(too_long.sum())
    clean = pool.select(_where(~too_long))
    stats["clean_pool"] = len(clean)
    stats["clean_pool_labels"] = dict(
        sorted(collections.Counter(clean["subset"]).items())
    )
    train_set = clean.select(
        processing.sample_indices(
            clean["stratum"],
            per_dataset,
            rng,
            source.stratification and source.stratification.shares,
            source.name,
        )
    )
    stats.update(
        {
            "train": len(train_set),
            "eval": len(eval_set),
            "train_labels": dict(
                sorted(collections.Counter(train_set["subset"]).items())
            ),
            "eval_labels": dict(
                sorted(collections.Counter(eval_set["subset"]).items())
            ),
            "train_with_spans": sum(bool(s) for s in train_set["spans"]),
            "eval_with_spans": sum(bool(s) for s in eval_set["spans"]),
            "options": len(train_set[0]["options"]),
            "train_tokens": processing.length_stats(train_set["n_tokens"]),
            "eval_tokens": processing.length_stats(eval_set["n_tokens"]),
            "clean_pool_tokens": processing.length_stats(clean["n_tokens"]),
        }
    )
    if source.stratification:
        stats["train_strata"] = dict(
            sorted(collections.Counter(train_set["stratum"]).items())
        )
        stats["clean_pool_strata"] = dict(
            sorted(collections.Counter(clean["stratum"]).items())
        )
    columns = [*processing.FEATURE_COLUMNS, "overlap_text"]
    return (
        train_set.remove_columns(columns),
        eval_set.remove_columns(columns),
        stats,
    )


def build(cfg: config.BuildDataConfig) -> dict[str, Any]:
    """Writes train.parquet, eval.parquet and manifest.json under cfg.out.

    Returns:
      The manifest.
    """
    from transformers import AutoTokenizer

    out = pathlib.Path(cfg.out)
    names = cfg.datasets or DATASETS
    mixture = {name: cfg.per_dataset for name in names}
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # We fork.
    tokenizer = AutoTokenizer.from_pretrained(cfg.length_tokenizer)
    logger.info("indexing benchmarks ...")
    index = contamination.OverlapIndex(contamination.benchmark_items())
    processing.set_context(processing.FeatureContext(tokenizer, index))
    manifest: dict[str, Any] = {
        "paper": "Li & Wagle, LLM-as-Jev, arXiv 2610.02076v3, Reason+Long-12k",
        "seed": base.SEED,
        "mixture": mixture,
        "eval_rows": cfg.eval_rows,
        "max_tokens": cfg.max_tokens,
        "duplicates_agree_on": "answer text",
        "prompt": "models.decoder.build_prompt",
        "tokenizer": cfg.length_tokenizer,
        "normalization": {
            "ngram": normalize.NGRAM,
            "min_sentence_words": normalize.MIN_SENTENCE_WORDS,
        },
        "benchmarks": {
            name: {
                "items": index.sizes[name],
                **download.provenance(download.BENCH_SOURCES[name]),
            }
            for name in index.names
        },
        "benchmark_suites": {k: list(v) for k, v in download.SUITES.items()},
        "datasets": {},
    }
    train_parts, eval_parts = [], []
    for name in names:
        logger.info("{} ...", name)
        train_set, eval_set, stats = build_dataset(
            sources.get_source(name),
            index,
            cfg.eval_rows,
            cfg.max_tokens,
            cfg.num_proc,
            mixture[name],
        )
        manifest["datasets"][name] = stats
        train_parts.append(train_set)
        eval_parts.append(eval_set)
        logger.info(
            json.dumps(
                {
                    k: v
                    for k, v in stats.items()
                    if not k.startswith("contaminated_examples")
                }
            )
        )
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
    logger.info(
        "{} {}", json.dumps(manifest["totals"]), json.dumps(manifest["checks"])
    )
    return manifest


def main() -> None:
    """Downloads the sources, or builds the mix."""
    cfg, _ = config.from_command_line(config.BuildDataConfig, __doc__)
    if cfg.download_only:
        download.download()
        return
    build(cfg)


if __name__ == "__main__":
    main()
