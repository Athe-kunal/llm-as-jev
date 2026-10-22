"""Reasoning over long inputs: ProofWriter, ContractNLI and MuSiQue.

These carry the datasets' own evidence spans.
"""

import json
import re
import zipfile

from llm_jev.dataloader.sources import base
from llm_jev.dataloader.sources import download

PROOFWRITER_CONFIG = "OWA/depth-5"
PROOFWRITER_LABELS = {
    "true": "True: the theory implies the statement.",
    "false": "False: the theory implies that the statement is false.",
    "unknown": "Unknown: the theory implies neither the statement nor its"
    " negation.",
}
CONTRACTNLI_LABELS = {
    "entailed": "The contract entails the statement.",
    "contradicted": "The contract contradicts the statement.",
    "not_mentioned": "The contract does not mention it.",
}
CONTRACTNLI_CHOICES = {
    "Entailment": "entailed",
    "Contradiction": "contradicted",
    "NotMentioned": "not_mentioned",
}
MUSIQUE_LABELS = {
    "yes": "Yes: the paragraphs contain enough information to answer the"
    " question.",
    "no": "No: the paragraphs do not contain enough information to answer"
    " the question.",
}


def sentence_offsets(theory: str, texts: list[tuple[str, str]]) -> dict:
    """(start, end) in the theory of each fact / rule, in theory order."""
    where, pos = {}, 0
    for key, text in texts:
        i = theory.find(text, pos)
        if i < 0:
            i = theory.find(text)
        if i >= 0:
            where[key] = (i, i + len(text))
            pos = i + len(text)
    return where


@base.register("proofwriter", splits={
    "train": f"{PROOFWRITER_CONFIG} train",
    "validation": f"{PROOFWRITER_CONFIG} dev"},
    stratification=base.Stratification(  # label-balanced
        lambda e: e.subset, "equal"))
def proofwriter() -> base.Split:
    """ProofWriter (OWA, depth <= 5): theory + statement -> T / F / Unknown.

    Spans: the facts and rules of the first proof (none for Unknown).
    """
    path = download.local_path(
        "proofwriter", "proofwriter-dataset-V2020.12.3.zip")
    names = list(PROOFWRITER_LABELS)
    out = {}
    with zipfile.ZipFile(path) as z:
        for split, stem in (("train", "train"), ("validation", "dev")):
            examples = []
            member = (f"proofwriter-dataset-V2020.12.3/{PROOFWRITER_CONFIG}/"
                      f"meta-{stem}.jsonl")
            with z.open(member) as f:
                for line in f:
                    x = json.loads(line)
                    theory = x["theory"]
                    where = sentence_offsets(theory, [
                        *((k, v["text"]) for k, v in x["triples"].items()),
                        *((k, v["text"]) for k, v in x["rules"].items())])
                    for qid, q in x["questions"].items():
                        label = str(q["answer"]).lower()
                        spans = []
                        proofs = q.get("proofsWithIntermediates") or []
                        if label != "unknown" and proofs:
                            used = re.findall(r"(?:triple|rule)\d+",
                                              proofs[0]["representation"])
                            spans = [("Theory", *where[k]) for k in used
                                     if k in where]
                        examples.append(base.make_example(
                            "proofwriter", (x["id"], qid), label, x["id"],
                            [("Theory", theory)],
                            f'Statement: "{q["question"]}"\nBased only on'
                            " the theory, is the statement true, false, or"
                            " unknown?",
                            PROOFWRITER_LABELS.values(), names.index(label),
                            spans, overlap_text=[theory, q["question"]]))
            out[split] = examples
    return out


@base.register("contractnli", splits={"train": "train", "validation": "dev"})
def contractnli() -> base.Split:
    """ContractNLI: NDA + hypothesis -> entailed / contradicted / not mentioned.

    Split by document. "Not mentioned" has no spans.
    """
    names = list(CONTRACTNLI_LABELS)
    out = {}
    with zipfile.ZipFile(
            download.local_path("contractnli", "contract-nli.zip")) as z:
        for split, stem in (("train", "train"), ("validation", "dev")):
            data = json.loads(z.read(f"contract-nli/{stem}.json"))
            examples = []
            for doc in data["documents"]:
                text = doc["text"]
                for key, a in doc["annotation_sets"][0]["annotations"].items():
                    label = CONTRACTNLI_CHOICES[a["choice"]]
                    spans = (
                        [] if label == "not_mentioned"
                        else [("Contract", *doc["spans"][i])
                              for i in a["spans"]])
                    hypothesis = data["labels"][key]["hypothesis"]
                    examples.append(base.make_example(
                        "contractnli", (doc["id"], key), label, doc["id"],
                        [("Contract", text)],
                        f'Statement: "{hypothesis}"\n'
                        "Does the contract entail the statement, contradict"
                        " it, or not mention it?",
                        CONTRACTNLI_LABELS.values(), names.index(label),
                        spans, overlap_text=[text, hypothesis]))
            out[split] = examples
    return out


@base.register("musique", splits={
    "train": "full_v1.0 train", "validation": "full_v1.0 dev"})
def musique() -> base.Split:
    """MuSiQue-Full: question + paragraphs -> answerable yes / no.

    Each answerable question has an unanswerable contrast copy with the same
    id (the group). Spans (answerable rows only): the supporting paragraphs.
    """
    names = list(MUSIQUE_LABELS)
    out = {}
    for split, stem in (("train", "train"), ("validation", "dev")):
        examples = []
        path = download.local_path("musique", f"musique_full_v1.0_{stem}.jsonl")
        with open(path) as f:
            for line in f:
                x = json.loads(line)
                text, spans, units = "", [], []
                for p in x["paragraphs"]:
                    block = f"{p['title']}\n{p['paragraph_text']}"
                    if text:
                        text += "\n\n"
                    if p["is_supporting"] and x["answerable"]:
                        spans.append(("Paragraphs", len(text),
                                      len(text) + len(block)))
                    text += block
                    units.append(block)
                label = "yes" if x["answerable"] else "no"
                question = (f"Question: {x['question']}\nDo the paragraphs"
                            " contain enough information to answer the"
                            " question?")
                examples.append(base.make_example(
                    "musique", (x["id"], label), label, x["id"],
                    [("Paragraphs", text)], question, MUSIQUE_LABELS.values(),
                    names.index(label), spans,
                    overlap_text=[*units, x["question"]]))
        out[split] = examples
    return out
