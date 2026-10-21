"""Multiple choice: parts + question -> one of the release's options.

ReClor, LogiQA 2.0, CosmosQA, PIQA, alpha-NLI and QuALITY. The options keep
the release's order; the subset is the gold option's 1-based position.
"""

import csv
import json
import zipfile
from typing import Any

from llm_jev.dataloader import normalize
from llm_jev.dataloader import schema
from llm_jev.dataloader.sources import base
from llm_jev.dataloader.sources import download

COSMOS_NONE_SHARE = 0.16  # Appendix G: "we upsample these to 16%"


def mc_example(
    source: str, key: Any, group: Any, parts: list[tuple[str, str]],
    question: str, options: list[str], gold: int,
    fixed_question: bool = False
) -> schema.Example:
    """A multiple-choice example; gold is the 0-based index of the answer.

    Args:
      source: Dataset name.
      key: Identifies the row inside the dataset.
      group: Raw identifier of the leakage unit.
      parts: (name, text) pairs in prompt order.
      question: The question text.
      options: The answer options, in release order.
      gold: 0-based index of the answer.
      fixed_question: Whether the question is our wording (PIQA, alpha-NLI),
        so the overlap filter skips it.

    Returns:
      The example.
    """
    return base.make_example(
        source, key, str(gold + 1), group, parts, question, options, gold,
        overlap_text=[text for _, text in parts]
        + ([] if fixed_question else [question]) + list(options))


@base.register("reclor", splits={"train": "train", "validation": "val"})
def reclor() -> base.Split:
    """ReClor: passage + question -> one of 4 options (train / val)."""
    out = {}
    for split, filename in (("train", "train.json"),
                            ("validation", "val.json")):
        data = json.loads(download.local_path("reclor", filename).read_text())
        out[split] = [
            mc_example("reclor", x["id_string"], x["context"],
                       [("Passage", x["context"])], x["question"],
                       x["answers"], x["label"])
            for x in data
        ]
    return out


@base.register("logiqa2", splits={"train": "train", "validation": "dev"})
def logiqa2() -> base.Split:
    """LogiQA 2.0 (English MRC): passage + question -> 4 options."""
    out = {}
    for split, filename in (("train", "train.txt"), ("validation", "dev.txt")):
        with open(download.local_path("logiqa2", filename)) as f:
            data = [json.loads(line) for line in f if line.strip()]
        out[split] = [
            mc_example("logiqa2", (split, x["id"]), x["text"],
                       [("Passage", x["text"])], x["question"], x["options"],
                       x["answer"])
            for x in data
        ]
    return out


def is_none_answer(text: str) -> bool:
    """Whether text is CosmosQA's catch-all option ("None of the above ...")."""
    return " ".join(normalize.words(text)).startswith("none of the above")


def cosmosqa_stratum(example: schema.Example) -> str:
    """"none" for a "none of the above" gold, else "other"."""
    return "none" if is_none_answer(example.answer) else "other"


@base.register("cosmosqa", splits={"train": "train", "validation": "valid"},
               stratum=cosmosqa_stratum,
               shares={"none": COSMOS_NONE_SHARE,
                       "other": 1 - COSMOS_NONE_SHARE})
def cosmosqa() -> base.Split:
    """CosmosQA: context + question -> 4 options (train / valid)."""
    out = {}
    for split, filename in (("train", "train.csv"),
                            ("validation", "valid.csv")):
        with open(download.local_path("cosmosqa", filename),
                  newline="") as f:
            data = list(csv.DictReader(f))
        out[split] = [
            mc_example("cosmosqa", x["id"], x["context"],
                       [("Context", x["context"])], x["question"],
                       [x[f"answer{i}"] for i in range(4)], int(x["label"]))
            for x in data
        ]
    return out


@base.register("piqa", splits={"train": "train", "validation": "valid"})
def piqa() -> base.Split:
    """PIQA: goal -> the better of two solutions (train / valid)."""
    out = {}
    for split, stem in (("train", "train"), ("validation", "valid")):
        with open(download.local_path("piqa", f"{stem}.jsonl")) as f:
            data = [json.loads(line) for line in f if line.strip()]
        gold = download.local_path(
            "piqa", f"{stem}-labels.lst").read_text().split()
        out[split] = [
            mc_example("piqa", (split, i), x["goal"], [("Goal", x["goal"])],
                       "Which solution better achieves the goal?",
                       [x["sol1"], x["sol2"]], int(g), fixed_question=True)
            for i, (x, g) in enumerate(zip(data, gold, strict=True))
        ]
    return out


@base.register("anli", splits={"train": "train", "validation": "dev"})
def anli() -> base.Split:
    """alpha-NLI (ART): two observations -> the more plausible hypothesis."""
    out = {}
    path = download.local_path("anli", "alphanli-train-dev.zip")
    with zipfile.ZipFile(path) as z:
        for split, stem in (("train", "train"), ("validation", "dev")):
            data = [json.loads(line)
                    for line in z.read(f"{stem}.jsonl").decode().splitlines()
                    if line.strip()]
            gold = z.read(f"{stem}-labels.lst").decode().split()
            out[split] = [
                mc_example(
                    "anli", (split, i, x["story_id"]), x["story_id"],
                    [("Observation 1", x["obs1"]),
                     ("Observation 2", x["obs2"])],
                    "Which hypothesis best explains what happened between"
                    " the two observations?",
                    [x["hyp1"], x["hyp2"]], int(g) - 1, fixed_question=True)
                for i, (x, g) in enumerate(zip(data, gold, strict=True))
            ]
    return out


@base.register("quality", splits={
    "train": "v1.0.1 htmlstripped train",
    "validation": "v1.0.1 htmlstripped dev"})
def quality() -> base.Split:
    """QuALITY v1.0.1 (HTML-stripped): article + question -> 4 options.

    Gold: the release's gold_label (annotators' majority), 1-based.
    """
    out = {}
    path = download.local_path("quality", "QuALITY.v1.0.1.zip")
    with zipfile.ZipFile(path) as z:
        for split, stem in (("train", "train"), ("validation", "dev")):
            examples = []
            for line in z.read(
                    f"QuALITY.v1.0.1.htmlstripped.{stem}").decode().split("\n"):
                if not line.strip():
                    continue
                x = json.loads(line)
                article = x["article"].strip()
                for q in x["questions"]:
                    if not q.get("gold_label"):
                        continue
                    examples.append(mc_example(
                        "quality", q["question_unique_id"], x["article_id"],
                        [("Article", article)], q["question"].strip(),
                        q["options"], int(q["gold_label"]) - 1))
            out[split] = examples
    return out
