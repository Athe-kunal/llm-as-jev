"""Intent classification: utterance -> one of the dataset's intent names.

CLINC150, MASSIVE (en-US) and Bitext customer support. The options are the
intent names verbatim; the subset is the gold intent.
"""

import csv
import json
import random
import re
import tarfile

from llm_jev.dataloader import normalize
from llm_jev.dataloader import schema
from llm_jev.dataloader.sources import base
from llm_jev.dataloader.sources import download

INTENT_QUESTION = "Which intent does the user's message express?"
BITEXT_HELDOUT = 0.1  # share of distinct utterances held out (no split)


def intent_examples(
    source: str, split: str, items: list[tuple[str, str]], names: list[str]
) -> list[schema.Example]:
    """Utterance -> one of the dataset's intent names (verbatim).

    Args:
      source: Dataset name.
      split: Split name, part of each example's key.
      items: (utterance, intent name) pairs.
      names: All intent names, in option order.

    Returns:
      One example per item.
    """
    index = {name: i for i, name in enumerate(names)}
    return [
        base.make_example(
            source, (split, i, text), intent, text, [("Message", text)],
            INTENT_QUESTION, names, index[intent], overlap_text=[text])
        for i, (text, intent) in enumerate(items)
    ]


def clinc_names() -> list[str]:
    """Intent names of the clinc_oos "plus" config, by class index."""
    lines = download.local_path(
        "clinc150", "README.md").read_text().splitlines()
    start = lines.index("- config_name: plus")
    names: dict[int, str] = {}
    for line in lines[start + 1:]:
        m = re.fullmatch(r"\s+'(\d+)': (.+)", line)
        if m:
            name = m[2].strip()
            if len(name) > 1 and name[0] == name[-1] == "'":  # YAML-quoted
                name = name[1:-1].replace("''", "'")
            names[int(m[1])] = name
        elif names:
            break
    return [names[i] for i in range(len(names))]


@base.register("clinc150")
def clinc150() -> base.Split:
    """CLINC150 "plus" (150 intents + oos): train / validation."""
    import pandas as pd  # pylint: disable=import-outside-toplevel

    names = clinc_names()
    out = {}
    for split in ("train", "validation"):
        data = pd.read_parquet(download.local_path(
            "clinc150", f"plus/{split}-00000-of-00001.parquet"))
        out[split] = intent_examples(
            "clinc150", split,
            [(t, names[i]) for t, i in zip(data.text, data.intent,
                                           strict=True)],
            names)
    return out


@base.register("massive", splits={"train": "train", "validation": "dev"})
def massive() -> base.Split:
    """MASSIVE 1.1 en-US (60 intents): train / dev partitions."""
    path = download.local_path("massive", "amazon-massive-dataset-1.1.tar.gz")
    with tarfile.open(path) as t:
        data = [json.loads(line)
                for line in t.extractfile("1.1/data/en-US.jsonl")]
    names = sorted({x["intent"] for x in data})
    return {
        split: intent_examples(
            "massive", split,
            [(x["utt"], x["intent"]) for x in data if x["partition"] == part],
            names)
        for split, part in (("train", "train"), ("validation", "dev"))
    }


@base.register("bitext", splits={
    "train": "all (90% of distinct utterances)",
    "validation": "all (seeded 10% of distinct utterances)"})
def bitext() -> base.Split:
    """Bitext customer support (27 intents); one file, so a seeded split.

    BITEXT_HELDOUT of the distinct utterances (after normalization, so
    copies stay together) form the "validation" side.
    """
    path = download.local_path(
        "bitext", "Bitext_Sample_Customer_Support_Training_Dataset_27K_"
        "responses-v11.csv")
    with open(path, newline="") as f:
        data = list(csv.DictReader(f))
    names = sorted({x["intent"] for x in data})
    examples = intent_examples(
        "bitext", "all", [(x["instruction"], x["intent"]) for x in data],
        names)
    keys = [normalize.input_key(e) for e in examples]
    distinct = sorted(set(keys))
    random.Random(f"{base.SEED}:bitext-split").shuffle(distinct)
    held = set(distinct[: int(BITEXT_HELDOUT * len(distinct))])
    return {
        "train": [e for e, k in zip(examples, keys) if k not in held],
        "validation": [e for e, k in zip(examples, keys) if k in held],
    }
