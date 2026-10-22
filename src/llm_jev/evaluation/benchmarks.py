"""External evaluation: the paper's benchmarks, as decision rows.

Usage (a trained run, or the base model without --adapter):
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m llm_jev.evaluation.benchmarks \
      --model Qwen/Qwen3.5-4B --adapter runs/qwen3.5_4b/adapter \
      --out runs/qwen3.5_4b --tag final

Li & Wagle (arXiv 2610.02076v3, §4.1, App. B): External = fixed samples of
BoolQ (500), MMLU (1,200), MMLU-Pro (800), ARC-Challenge (600), WinoGrande
(500), SciQ (400) and the full Banking77 test set (3,080), all options kept;
the External macro average is the mean accuracy on the first six. JevBench:
the 231 public items. The paper's samples are not released: ours are seeded
uniform draws (seed 20260922:<name>). Files are the pinned downloads of
dataloader.sources.download (make download). Rendering choices the paper
leaves open are in docs/architecture.md §7.
"""

import argparse
import json
import pathlib
import random
from typing import Any

import pandas as pd
from loguru import logger

from llm_jev.dataloader import schema
from llm_jev.dataloader.sources import base
from llm_jev.dataloader.sources import download
from llm_jev.dataloader.sources import intent
from llm_jev.evaluation import evaluate
from llm_jev.modeling import modeling

SEED = base.SEED
EXTERNAL = ("boolq", "mmlu", "mmlu_pro", "arc_challenge", "winogrande", "sciq")
SIZES = {"boolq": 500, "mmlu": 1200, "mmlu_pro": 800, "arc_challenge": 600,
         "winogrande": 500, "sciq": 400, "banking77": None, "jevbench": None}


def _parquet(name: str, filename: str) -> list[dict]:
    return pd.read_parquet(download.benchmark_path(name, filename)).to_dict("records")


def _example(
    name: str, i: int, parts: list[tuple[str, str]], question: str,
    options: list[str], gold: int, meta: dict[str, Any] | None = None
) -> schema.Example:
    """One benchmark item; empty parts are left out."""
    return schema.Example(
        id=f"{name}:{i}", source=name,
        parts=tuple(schema.Part(n, t) for n, t in parts if t),
        question=question, options=tuple(options), gold=gold,
        meta=meta or {})


def boolq() -> list[schema.Example]:
    return [_example("boolq", i, [("Passage", r["passage"])],
                     r["question"][:1].upper() + r["question"][1:] + "?",
                     ["Yes", "No"], 0 if r["answer"] else 1)
            for i, r in enumerate(_parquet(
                "boolq", "data/validation-00000-of-00001.parquet"))]


def mmlu() -> list[schema.Example]:
    return [_example("mmlu", i, [], r["question"], [str(o) for o in r["choices"]],
                     int(r["answer"]))
            for i, r in enumerate(_parquet(
                "mmlu", "all/test-00000-of-00001.parquet"))]


def mmlu_pro() -> list[schema.Example]:
    return [_example("mmlu_pro", i, [], r["question"],
                     [str(o) for o in r["options"]], int(r["answer_index"]))
            for i, r in enumerate(_parquet(
                "mmlu_pro", "data/test-00000-of-00001.parquet"))]


def arc_challenge() -> list[schema.Example]:
    return [_example("arc_challenge", i, [], r["question"],
                     list(map(str, r["choices"]["text"])),
                     list(map(str, r["choices"]["label"])).index(
                         r["answerKey"]))
            for i, r in enumerate(_parquet(
                "arc_challenge", "ARC-Challenge/test-00000-of-00001.parquet"))]


def winogrande() -> list[schema.Example]:
    return [_example("winogrande", i, [("Sentence", r["sentence"])],
                     "Which option correctly fills the blank (_)?",
                     [r["option1"], r["option2"]], int(r["answer"]) - 1)
            for i, r in enumerate(_parquet(
                "winogrande", "winogrande_xl/validation-00000-of-00001.parquet"))]


def sciq() -> list[schema.Example]:
    examples = []
    for i, r in enumerate(_parquet(
            "sciq", "data/test-00000-of-00001.parquet")):
        # the correct answer at a seeded position among the distractors
        opts = [r["correct_answer"], r["distractor1"], r["distractor2"],
                r["distractor3"]]
        random.Random(f"{SEED}:sciq:{i}").shuffle(opts)
        examples.append(_example(
            "sciq", i, [("Support", r["support"])], r["question"],
            [str(o) for o in opts], opts.index(r["correct_answer"])))
    return examples


def banking77() -> list[schema.Example]:
    frame = pd.read_csv(download.benchmark_path("banking77", "test.csv"))
    intents = sorted(frame["category"].unique())
    return [_example("banking77", i, [("Message", r.text)], intent.INTENT_QUESTION,
                     intents, intents.index(r.category))
            for i, r in enumerate(frame.itertuples())]


def jevbench() -> list[schema.Example]:
    """The 231 public items, rendered as JevBench's own LLM adapters
    (qwen_flash_linear, openai_compat) build them: options in the item's
    labels order, each "<label>: <criterion>" (noul: no / yes described by
    the false / true criteria; score: levels 0..k by their criteria); dict
    states as compact JSON."""
    examples = []
    for tier in ("easy", "original", "hard"):
        with open(download.benchmark_path("jevbench", f"{tier}.jsonl")) as f:
            for line in f:
                t = json.loads(line)
                q, crit = t["question"], t["question"].get("criteria")
                if q["type"] == "noul":
                    crit = crit or {}
                    rubric = {"no": crit.get("false", "No"),
                              "yes": crit.get("true", "Yes")}
                elif q["type"] == "score":
                    rubric = {str(k): c for k, c in enumerate(crit)}
                else:
                    rubric = {k: v or k for k, v in (crit or {}).items()} or {
                        k: k for k in t["labels"]}
                state = t["state"] if isinstance(t["state"], str) else \
                    json.dumps(t["state"], ensure_ascii=False)
                labels = [str(k) for k in t["labels"]]
                examples.append(schema.Example(
                    id=t["id"], source="jevbench",
                    parts=(schema.Part("State", state),),
                    question=q["instructions"],
                    options=tuple(k if rubric[k] == k else f"{k}: {rubric[k]}"
                                  for k in labels),
                    gold=labels.index(str(t["expected"])),
                    meta={"tier": tier}))
    return examples


LOADERS = {"boolq": boolq, "mmlu": mmlu, "mmlu_pro": mmlu_pro,
           "arc_challenge": arc_challenge, "winogrande": winogrande,
           "sciq": sciq, "banking77": banking77, "jevbench": jevbench}


def load(name: str) -> list[schema.Example]:
    """The examples of one benchmark: the paper's sample size, seeded."""
    examples = LOADERS[name]()
    n = SIZES[name]
    if n is not None:
        examples = random.Random(f"{SEED}:{name}").sample(examples, n)
    return examples


def main() -> None:
    """Evaluates a base model or a trained adapter on the benchmarks."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", help="a saved LoRA adapter (else the base)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--tag", default="base")
    parser.add_argument("--benchmarks", nargs="+", default=list(LOADERS),
                        choices=list(LOADERS))
    parser.add_argument("--gen-per", type=int, default=10,
                        help="text-mode rows per benchmark")
    parser.add_argument("--batch-tokens", type=int, default=32768,
                        help="padded prompt tokens per scoring batch")
    parser.add_argument("--batch-rows", type=int, default=64,
                        help="rows (and trie nodes) per scoring forward")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logger.add(out / f"benchmarks_{args.tag}.log")
    rows = [r for name in args.benchmarks for r in load(name)]
    logger.info("{} rows: {}", len(rows), {n: sum(r.source == n for r in rows)
                                          for n in args.benchmarks})
    gen_rows = evaluate.subset(rows, args.gen_per, SEED)
    tok = AutoTokenizer.from_pretrained(args.model)
    # an adapter is merged into the weights: no per-layer adapter calls at inference
    model = modeling.load(args.model, args.adapter)
    metrics, examples = evaluate.evaluate(model, tok, rows, gen_rows, "cuda",
                                         args.batch_tokens, args.batch_rows)
    ext = [metrics[n]["accuracy"] for n in EXTERNAL if n in metrics]
    if len(ext) == len(EXTERNAL):
        metrics["external_macro"] = {"accuracy": sum(ext) / len(ext)}
        logger.info("External macro accuracy: {:.4f}", sum(ext) / len(ext))
    evaluate.save(out, f"benchmarks_{args.tag}", metrics, examples)
    evaluate.log_summary(f"benchmarks_{args.tag}", metrics)


if __name__ == "__main__":
    main()
