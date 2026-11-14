"""Evaluates a decision model on the external benchmarks, as decision rows.

Usage:
  llm-jev-benchmarks model.kind=decoder model.path=Qwen/Qwen3-0.6B \
      model.adapter=runs/qwen3_0.6b/adapter out=runs/qwen3_0.6b

Li & Wagle (arXiv 2610.02076v3, Sec. 4.1, App. B). The paper's samples are
not released, so ours are seeded uniform draws (seed <SEED>:<name>). The
external_macro metric is the mean accuracy over the six External sets.
Files are the pinned downloads of data.sources.download. Writes
eval_benchmarks_<tag>.json.
"""

import json
import pathlib
import random
from typing import Any

from loguru import logger
import pandas as pd

from llm_jev import config
from llm_jev import models
from llm_jev.data import schema
from llm_jev.data.sources import download
from llm_jev.data.sources import intent
from llm_jev.evaluation import evaluate

SEED = config.SEED
EXTERNAL = ("boolq", "mmlu", "mmlu_pro", "arc_challenge", "winogrande", "sciq")
SIZES = {
    "boolq": 500,
    "mmlu": 1200,
    "mmlu_pro": 800,
    "arc_challenge": 600,
    "winogrande": 500,
    "sciq": 400,
    "banking77": None,
    "jevbench": None,
}


def _parquet(name: str, filename: str) -> list[dict[str, Any]]:
    """The records of one pinned benchmark Parquet file."""
    return pd.read_parquet(download.benchmark_path(name, filename)).to_dict(
        "records"
    )


def _example(
    name: str,
    i: int,
    parts: list[tuple[str, str]],
    question: str,
    options: list[str],
    gold: int,
    meta: dict[str, Any] | None = None,
) -> schema.Example:
    """One benchmark item; empty parts are left out."""
    return schema.Example(
        id=f"{name}:{i}",
        source=name,
        parts=tuple(schema.Part(n, t) for n, t in parts if t),
        question=question,
        options=tuple(options),
        gold=gold,
        meta=meta or {},
    )


def boolq() -> list[schema.Example]:
    """BoolQ validation: yes / no about a passage."""
    return [
        _example(
            "boolq",
            i,
            [("Passage", r["passage"])],
            r["question"][:1].upper() + r["question"][1:] + "?",
            ["Yes", "No"],
            0 if r["answer"] else 1,
        )
        for i, r in enumerate(
            _parquet("boolq", "data/validation-00000-of-00001.parquet")
        )
    ]


def mmlu() -> list[schema.Example]:
    """MMLU test, all subjects."""
    return [
        _example(
            "mmlu",
            i,
            [],
            r["question"],
            [str(o) for o in r["choices"]],
            int(r["answer"]),
        )
        for i, r in enumerate(
            _parquet("mmlu", "all/test-00000-of-00001.parquet")
        )
    ]


def mmlu_pro() -> list[schema.Example]:
    """MMLU-Pro test."""
    return [
        _example(
            "mmlu_pro",
            i,
            [],
            r["question"],
            [str(o) for o in r["options"]],
            int(r["answer_index"]),
        )
        for i, r in enumerate(
            _parquet("mmlu_pro", "data/test-00000-of-00001.parquet")
        )
    ]


def arc_challenge() -> list[schema.Example]:
    """ARC-Challenge test."""
    return [
        _example(
            "arc_challenge",
            i,
            [],
            r["question"],
            list(map(str, r["choices"]["text"])),
            list(map(str, r["choices"]["label"])).index(r["answerKey"]),
        )
        for i, r in enumerate(
            _parquet(
                "arc_challenge", "ARC-Challenge/test-00000-of-00001.parquet"
            )
        )
    ]


def winogrande() -> list[schema.Example]:
    """WinoGrande (XL) validation: which option fills the blank."""
    return [
        _example(
            "winogrande",
            i,
            [("Sentence", r["sentence"])],
            "Which option correctly fills the blank (_)?",
            [r["option1"], r["option2"]],
            int(r["answer"]) - 1,
        )
        for i, r in enumerate(
            _parquet(
                "winogrande", "winogrande_xl/validation-00000-of-00001.parquet"
            )
        )
    ]


def sciq() -> list[schema.Example]:
    """SciQ test; the correct answer at a seeded position."""
    examples = []
    for i, r in enumerate(_parquet("sciq", "data/test-00000-of-00001.parquet")):
        # the correct answer at a seeded position among the distractors
        opts = [
            r["correct_answer"],
            r["distractor1"],
            r["distractor2"],
            r["distractor3"],
        ]
        random.Random(f"{SEED}:sciq:{i}").shuffle(opts)
        examples.append(
            _example(
                "sciq",
                i,
                [("Support", r["support"])],
                r["question"],
                [str(o) for o in opts],
                opts.index(r["correct_answer"]),
            )
        )
    return examples


def banking77() -> list[schema.Example]:
    """Banking77 test: one of 77 intents."""
    frame = pd.read_csv(download.benchmark_path("banking77", "test.csv"))
    intents = sorted(frame["category"].unique())
    return [
        _example(
            "banking77",
            i,
            [("Message", r.text)],
            intent.INTENT_QUESTION,
            intents,
            intents.index(r.category),
        )
        for i, r in enumerate(frame.itertuples())
    ]


def jevbench() -> list[schema.Example]:
    """The 231 public items, rendered as JevBench's own adapters render them.

    Options are "<label>: <criterion>" in the item's label order; dict states
    are compact JSON.
    """
    examples = []
    for tier in ("easy", "original", "hard"):
        with open(download.benchmark_path("jevbench", f"{tier}.jsonl")) as f:
            for line in f:
                t = json.loads(line)
                q, crit = t["question"], t["question"].get("criteria")
                if q["type"] == "noul":
                    crit = crit or {}
                    rubric = {
                        "no": crit.get("false", "No"),
                        "yes": crit.get("true", "Yes"),
                    }
                elif q["type"] == "score":
                    rubric = {str(k): c for k, c in enumerate(crit)}
                else:
                    rubric = {k: v or k for k, v in (crit or {}).items()} or {
                        k: k for k in t["labels"]
                    }
                state = (
                    t["state"]
                    if isinstance(t["state"], str)
                    else json.dumps(t["state"], ensure_ascii=False)
                )
                labels = [str(k) for k in t["labels"]]
                examples.append(
                    schema.Example(
                        id=t["id"],
                        source="jevbench",
                        parts=(schema.Part("State", state),),
                        question=q["instructions"],
                        options=tuple(
                            k if rubric[k] == k else f"{k}: {rubric[k]}"
                            for k in labels
                        ),
                        gold=labels.index(str(t["expected"])),
                        meta={"tier": tier},
                    )
                )
    return examples


LOADERS = {
    "boolq": boolq,
    "mmlu": mmlu,
    "mmlu_pro": mmlu_pro,
    "arc_challenge": arc_challenge,
    "winogrande": winogrande,
    "sciq": sciq,
    "banking77": banking77,
    "jevbench": jevbench,
}


def load(name: str) -> list[schema.Example]:
    """The examples of one benchmark: the paper's sample size, seeded."""
    examples = LOADERS[name]()
    n = SIZES[name]
    if n is not None:
        examples = random.Random(f"{SEED}:{name}").sample(examples, n)
    return examples


def main() -> None:
    """Scores the benchmarks and writes eval_benchmarks_<tag>.json."""
    cfg, _ = config.from_command_line(config.BenchmarksConfig, __doc__)
    names = cfg.benchmarks or list(LOADERS)
    out = pathlib.Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    logger.add(out / f"benchmarks_{cfg.tag}.log")
    rows = [r for name in names for r in load(name)]
    logger.info(
        "{} rows: {}",
        len(rows),
        {n: sum(r.source == n for r in rows) for n in names},
    )
    model = models.load(cfg.model)
    metrics = evaluate.summarize(evaluate.score_rows(model, rows))
    external = [metrics[n]["accuracy"] for n in EXTERNAL if n in metrics]
    if len(external) == len(EXTERNAL):
        macro = sum(external) / len(external)
        metrics["external_macro"] = {"accuracy": macro}
        logger.info("External macro accuracy: {:.4f}", macro)
    evaluate.save(out, f"benchmarks_{cfg.tag}", metrics)
    evaluate.log_summary(f"benchmarks_{cfg.tag}", metrics)


if __name__ == "__main__":
    main()
