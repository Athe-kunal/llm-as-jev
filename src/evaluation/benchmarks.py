"""External evaluation: the paper's benchmarks, as decision rows.

Usage (a trained run, or the base model without --adapter):
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src .venv/bin/python -m evaluation.benchmarks \
      --model Qwen/Qwen3.5-4B --adapter runs/qwen3.5_4b/adapter \
      --out runs/qwen3.5_4b --tag final

Li & Wagle (arXiv 2610.02076v3, §4.1, App. B): External = fixed samples of
BoolQ (500), MMLU (1,200), MMLU-Pro (800), ARC-Challenge (600), WinoGrande
(500), SciQ (400) and the full Banking77 test set (3,080), all options kept;
the External macro average is the mean accuracy on the first six. JevBench:
the 231 public items. The paper's samples are not released: ours are seeded
uniform draws (seed 20260922:<name>). Files come from data/external/
benchmarks (downloaded by build_data.py --download). Rendering choices the
paper leaves open are in docs/architecture.md §7.
"""

import argparse
import json
import pathlib
import random

from loguru import logger

from evaluation import evaluate
from modeling import modeling

ROOT = pathlib.Path("data/external/benchmarks")
SEED = 20260922
EXTERNAL = ("boolq", "mmlu", "mmlu_pro", "arc_challenge", "winogrande", "sciq")
SIZES = {"boolq": 500, "mmlu": 1200, "mmlu_pro": 800, "arc_challenge": 600,
         "winogrande": 500, "sciq": 400, "banking77": None, "jevbench": None}
INTENT_QUESTION = "Which intent does the user's message express?"


def _parquet(rel: str) -> list[dict]:
    import pandas as pd

    (path,) = (ROOT / rel).parent.glob((ROOT / rel).name + "*.parquet")
    return pd.read_parquet(path).to_dict("records")


def _row(name: str, i: int, parts: list[tuple[str, str]], question: str,
         labels: dict[str, str], label: str) -> dict:
    return {"id": f"{name}:{i}", "source": name,
            "parts": [{"name": n, "text": t} for n, t in parts if t],
            "question": question, "labels": labels, "label": label,
            "spans": []}


def _numbered(options) -> dict[str, str]:
    return {str(k): str(o) for k, o in enumerate(options, 1)}


def boolq() -> list[dict]:
    return [_row("boolq", i, [("Passage", r["passage"])],
                 r["question"][:1].upper() + r["question"][1:] + "?",
                 {"yes": "Yes", "no": "No"}, "yes" if r["answer"] else "no")
            for i, r in enumerate(_parquet("boolq/data/validation"))]


def mmlu() -> list[dict]:
    return [_row("mmlu", i, [], r["question"], _numbered(r["choices"]),
                 str(int(r["answer"]) + 1))
            for i, r in enumerate(_parquet("mmlu/all/test"))]


def mmlu_pro() -> list[dict]:
    return [_row("mmlu_pro", i, [], r["question"], _numbered(r["options"]),
                 str(int(r["answer_index"]) + 1))
            for i, r in enumerate(_parquet("mmlu_pro/data/test"))]


def arc_challenge() -> list[dict]:
    return [_row("arc_challenge", i, [], r["question"],
                 dict(zip(map(str, r["choices"]["label"]),
                          map(str, r["choices"]["text"]))), r["answerKey"])
            for i, r in enumerate(_parquet("arc_challenge/ARC-Challenge/test"))]


def winogrande() -> list[dict]:
    return [_row("winogrande", i, [("Sentence", r["sentence"])],
                 "Which option correctly fills the blank (_)?",
                 {"1": r["option1"], "2": r["option2"]}, str(r["answer"]))
            for i, r in enumerate(_parquet("winogrande/winogrande_xl/validation"))]


def sciq() -> list[dict]:
    rows = []
    for i, r in enumerate(_parquet("sciq/data/test")):
        # the correct answer at a seeded position among the distractors
        opts = [r["correct_answer"], r["distractor1"], r["distractor2"],
                r["distractor3"]]
        random.Random(f"{SEED}:sciq:{i}").shuffle(opts)
        rows.append(_row("sciq", i, [("Support", r["support"])], r["question"],
                         _numbered(opts),
                         str(opts.index(r["correct_answer"]) + 1)))
    return rows


def banking77() -> list[dict]:
    import pandas as pd

    d = pd.read_csv(ROOT / "banking77" / "test.csv")
    intents = sorted(d["category"].unique())
    return [_row("banking77", i, [("Message", r.text)], INTENT_QUESTION,
                 {c: c for c in intents}, r.category)
            for i, r in enumerate(d.itertuples())]


def jevbench() -> list[dict]:
    """The 231 public items, rendered as JevBench's own LLM adapters
    (qwen_flash_linear, openai_compat) build them: options in the item's
    labels order, each "<label>: <criterion>" (noul: no / yes described by
    the false / true criteria; score: levels 0..k by their criteria); dict
    states as compact JSON."""
    rows = []
    for tier in ("easy", "original", "hard"):
        with open(ROOT / "jevbench" / f"{tier}.jsonl") as f:
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
                rows.append({
                    "id": t["id"], "source": "jevbench", "tier": tier,
                    "parts": [{"name": "State", "text": state}],
                    "question": q["instructions"],
                    "labels": {k: rubric[k] if rubric[k] == k else f"{k}: {rubric[k]}"
                               for k in map(str, t["labels"])},
                    "label": str(t["expected"]), "spans": []})
    return rows


LOADERS = {"boolq": boolq, "mmlu": mmlu, "mmlu_pro": mmlu_pro,
           "arc_challenge": arc_challenge, "winogrande": winogrande,
           "sciq": sciq, "banking77": banking77, "jevbench": jevbench}


def load(name: str) -> list[dict]:
    """The rows of one benchmark: the paper's sample size, seeded."""
    rows = LOADERS[name]()
    n = SIZES[name]
    if n is not None:
        rows = random.Random(f"{SEED}:{name}").sample(rows, n)
    for r in rows:
        if r["label"] not in r["labels"]:
            raise ValueError(f"{r['id']}: gold {r['label']!r} not an option")
    return rows


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
    logger.info("{} rows: {}", len(rows), {n: sum(r["source"] == n for r in rows)
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
