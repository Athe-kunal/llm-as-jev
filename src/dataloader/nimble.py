"""Nimble rows with Nimble's own verified evidence as their spans.

The rows (parts, question, labels, label) come from data/synth/nimble-{split}.jsonl,
already in this project's row format; their spans there are the earlier
8-sample LLM vote. Here the spans are replaced by the evidence Nimble verified
(data/external/nimble/{split}.jsonl, evidence_certificate.spec): the two
focus sentences, whose change flips the answer, and the policy quotes the
answer applies (docs/learnings.md section 2). A counterfactual row holds one
focus sentence negated (verified_pair.negative_left / negative_right), so a
row's focus spans are whichever of the pair's sentences occur in its text.
"""

import json
import pathlib
from typing import Any

from loguru import logger

SYNTH = pathlib.Path("data/synth")
EXTERNAL = pathlib.Path("data/external/nimble")
_ROW_KEYS = ("id", "source", "parts", "question", "labels", "label")


def verified_spans(row: dict[str, Any], certificate: dict[str, Any]) -> list[dict[str, Any]]:
    """Char spans of the focus and policy evidence in the row's parts
    ({"part", "start", "end", "text", "kind"}), each text at its first
    occurrence; texts that do not occur verbatim are skipped."""
    spec, pair = certificate["spec"], certificate["verified_pair"]
    texts = ([("focus", x["text"]) for x in spec["focus_evidence"]]
             + [("focus", pair[k]) for k in ("negative_left", "negative_right")]
             + [("policy", x["text"]) for x in spec["policy_evidence"]])
    spans, seen = [], set()
    for kind, text in texts:
        for p in row["parts"]:
            start = p["text"].find(text)
            if start >= 0 and (p["name"], start) not in seen:
                seen.add((p["name"], start))
                spans.append({"part": p["name"], "start": start, "end": start + len(text), "text": text, "kind": kind})
                break
    return spans


def load_rows(split: str = "eval") -> list[dict[str, Any]]:
    """Nimble's rows of one split (train, eval), source "nimble", with verified spans."""
    certificates = {}
    with open(EXTERNAL / f"{split}.jsonl") as f:
        for line in f:
            r = json.loads(line)
            certificates[r["id"]] = r["evidence_certificate"]
    rows = []
    with open(SYNTH / f"nimble-{split}.jsonl") as f:
        for line in f:
            r = {k: v for k, v in json.loads(line).items() if k in _ROW_KEYS}
            r["source"] = "nimble"
            r["spans"] = verified_spans(r, certificates[r["id"].split(":", 1)[1]])
            rows.append(r)
    logger.info("{} Nimble {} rows, {} with verified spans", len(rows), split, sum(bool(r["spans"]) for r in rows))
    return rows
