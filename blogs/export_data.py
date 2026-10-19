"""Exports the examples the blog (a React app in blogs/) draws, as JSON.

Usage (from the repo root):
  PYTHONPATH=src .venv/bin/python blogs/export_data.py

Every example is rendered by answers.build_prompt with the real chat template
and cut into labelled segments; JavaScript only draws them. Writes
blogs/data/prompt_and_data.json (real dataset rows: git-ignored, local only).
"""

import collections
import json
import pathlib
import random
from typing import Any

import transformers
from loguru import logger

from dataloader import answers, data

_HERE = pathlib.Path(__file__).parent
_MODEL = "Qwen/Qwen3.5-4B"
_PER, _SEED = 2, 0
# Span-carrying sources: this many of the _PER rows come from the span pool.
_SPAN_SOURCES = {"proofwriter", "contractnli", "musique"}
_SPAN_ROWS = 2
_GROUPS = {"Intent": ("clinc150", "massive", "bitext"),
           "Long input": ("contractnli", "musique", "quality")}

# source -> (title, description); descriptions follow docs/data.md.
DATASETS: dict[str, tuple[str, str]] = {
    "clinc150": ("CLINC150", "Short user messages, each labelled with one of 151 "
                 "intents across 10 domains, including out-of-scope (oos)."),
    "massive": ("MASSIVE", "Voice-assistant utterances (en-US) labelled with one "
                "of 60 intents."),
    "bitext": ("Bitext", "E-commerce customer-support messages labelled with one "
               "of 27 intents. Messages keep the release's {{...}} placeholders."),
    "reclor": ("ReClor", "Argument passages with a 4-option question about the "
               "argument's logic."),
    "logiqa2": ("LogiQA 2.0", "Short civil-service-style logic passages with a "
                "4-option question."),
    "cosmosqa": ("CosmosQA", "A short everyday context, a commonsense question "
                 "and 4 options."),
    "piqa": ("PIQA", "Physical commonsense: a goal and two candidate solutions."),
    "anli": ("αNLI", "Two observations and two hypotheses: which one explains "
             "what happened in between?"),
    "proofwriter": ("ProofWriter", "Facts, rules and a statement: true, false or "
                    "unknown given the theory (OWA, depth 5)."),
    "contractnli": ("ContractNLI", "A long contract and a hypothesis: entailed, "
                    "contradicted or not mentioned."),
    "musique": ("MuSiQue", "Twenty Wikipedia paragraphs and a multi-hop "
                "question: are the paragraphs enough to answer it?"),
    "quality": ("QuALITY", "Long story articles with a 4-option reading "
                "question; gold is the annotators' majority."),
}


def pick_rows(
    rows: list[dict[str, Any]], per: int, seed: int
) -> dict[str, list[dict[str, Any]]]:
    """Seeded sample of `per` rows for each dataset, in DATASETS order."""
    rng = random.Random(seed)
    by_src: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        by_src[r["source"]].append(r)
    picked: dict[str, list[dict[str, Any]]] = {}
    for src in DATASETS:
        pool = by_src[src]
        chosen: list[dict[str, Any]] = []
        if src in _SPAN_SOURCES:
            with_spans = [r for r in pool if r["spans"]]
            chosen = rng.sample(with_spans, min(_SPAN_ROWS, per, len(with_spans)))
        taken = {r["id"] for r in chosen}
        rest = [r for r in pool if r["id"] not in taken]
        chosen += rng.sample(rest, min(per - len(chosen), len(rest)))
        picked[src] = chosen
    return picked


def segments(row: dict[str, Any], order: list[str], tok: Any) -> list[list]:
    """The exact build_prompt text cut into [kind, text] (options: [kind, text, k])."""
    text, where = answers.build_prompt(row, order, tok)
    segs: list[list] = []

    def add(kind: str, s: str, *extra) -> None:
        if s:
            segs.append([kind, s, *extra])

    st = text.index("State:")
    add("sys", text[:st])
    add("label", "State:")
    cur = st + len("State:")
    for s, e in sorted(where.values()):
        add("sep", text[cur:s])
        add("part", text[s:e])
        cur = e
    q = text.index(row["question"], cur)
    add("sep", text[cur:q])
    add("q", row["question"])
    cur = q + len(row["question"])
    opt = text.index("Options:", cur)
    add("sep", text[cur:opt])
    add("label", "Options:")
    add("sep", "\n")
    cur = opt + len("Options:\n")
    for i, k in enumerate(order):
        line = f"[{i + 1}] {row['labels'][k]}\n"
        if not text.startswith(line, cur):
            raise ValueError(f"option lines do not match the prompt of {row['id']}")
        add("opt", line, i)
        cur += len(line)
    tail = text.index("<|im_end|>", cur)
    add("ins", text[cur:tail])
    pre = text.index(answers.ANSWER_PREFIX, tail)
    add("sys", text[tail:pre])
    add("pre", text[pre:])
    return segs, len(tok(text, add_special_tokens=False)["input_ids"])


def example(row: dict[str, Any], tok: Any, rng: random.Random) -> dict[str, Any]:
    """One row as the page draws it; options shuffled as at training time."""
    order = list(row["labels"])
    rng.shuffle(order)
    gold = order.index(row["label"])
    segs, n_tokens = segments(row, order, tok)
    suffix = answers.suffix_ids(tok, len(order))[gold]
    return {"id": row["id"], "segs": segs, "k": len(order), "gold": gold,
            "goldText": row["labels"][row["label"]], "tokens": n_tokens,
            "suffix": [tok.decode([t]) for t in suffix]}


def main() -> None:
    """Writes data/prompt_and_data.json from the picked rows."""
    tok = transformers.AutoTokenizer.from_pretrained(_MODEL)
    rows = data.load_rows()
    counts = collections.Counter(r["source"] for r in rows)
    picked = pick_rows(rows, _PER, _SEED)
    rng = random.Random(_SEED)
    group = {s: g for g, srcs in _GROUPS.items() for s in srcs}
    payload = {
        "model": _MODEL,
        "datasets": [{"id": s, "title": t, "desc": d, "rows": counts[s],
                      "group": group.get(s, "Short reasoning"),
                      "examples": [example(r, tok, rng) for r in picked[s]]}
                     for s, (t, d) in DATASETS.items()],
    }
    out = _HERE / "data" / "prompt_and_data.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False))
    logger.info("wrote {} ({:.0f} KB), {} datasets", out, out.stat().st_size / 1024,
                len(payload["datasets"]))


if __name__ == "__main__":
    main()
