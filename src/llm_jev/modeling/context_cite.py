"""ContextCite for the decision interface: which context sentences an answer depends on.

Cohen-Wang et al., ContextCite (NeurIPS 2024), following
github.com/MadryLab/context-cite (MIT). Sources are the sentences of the
example's parts; ablations keep a random subset, the predicted answer is
re-scored on each, and a sparse linear surrogate gives one weight per
sentence. The target is logit q(answer) from answers.score_batch.
"""

import dataclasses
import functools
import re
from typing import Any

import nltk
import numpy as np
import torch
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from llm_jev.dataloader import answers
from llm_jev.dataloader import schema


@functools.cache
def _punkt() -> None:
    """Fetches nltk's sentence model once (the reference does it on import)."""
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)


_WORD = re.compile(r"[^\W\d_]{2,}")


def has_content(text: str, min_words: int = 2, min_alnum: float = 0.5) -> bool:
    """Whether a sentence carries content (enough words, mostly letters or digits).

    Blank form fields, checkbox lines and nltk abbreviation fragments do not.
    """
    alnum = sum(ch.isalnum() for ch in text) / max(len(text), 1)
    return len(_WORD.findall(text)) >= min_words and alnum >= min_alnum


def split_sources(example: schema.Example, min_words: int = 2,
                  min_alnum: float = 0.5) -> list[dict[str, Any]]:
    """The example's sources: the sentences of every part, in order, each with content.

    Sentences without content join a neighbouring source, so the text is
    unchanged. Each source is {"part", "start", "end" (char range in that
    part's text, the same offsets as spans), "sep" (the text since the previous
    source of its part)}.
    """
    _punkt()
    sources = []
    for i, part in enumerate(example.parts):
        text, cur, own, lead = part.text, 0, [], None
        for line in text.splitlines():
            for sentence in nltk.sent_tokenize(line):
                start = text.find(sentence, cur)
                if start < 0:
                    raise ValueError(f"{example.id}: sentence not found in part {i}")
                s = {"part": i, "start": start, "end": start + len(sentence), "sep": text[cur:start]}
                cur = s["end"]
                if not has_content(sentence, min_words, min_alnum):
                    if own:
                        own[-1]["end"] = s["end"]
                    elif lead:
                        lead["end"] = s["end"]
                    else:
                        lead = s
                    continue
                if lead:
                    s = {**s, "start": lead["start"], "sep": lead["sep"]}
                    lead = None
                own.append(s)
        sources += own or ([lead] if lead else [])
    return sources


def ablate(example: schema.Example, sources: list[dict[str, Any]],
           keep: list[bool]) -> schema.Example:
    """The example with only the kept sources (the reference's get_context).

    Text outside a part's sources is always kept, so keeping every source
    rebuilds the example exactly. Spans are dropped.
    """
    kept: list[list[str]] = [[] for _ in example.parts]
    for s, k in zip(sources, keep):
        if k:
            text = example.parts[s["part"]].text
            kept[s["part"]].append((s["sep"] if kept[s["part"]] else "") + text[s["start"]:s["end"]])
    parts = []
    for i, p in enumerate(example.parts):
        own = [s for s in sources if s["part"] == i]
        lead = p.text[: own[0]["start"]] if own else p.text
        tail = p.text[own[-1]["end"]:] if own else ""
        parts.append(schema.Part(p.name, lead + "".join(kept[i]) + tail))
    return dataclasses.replace(example, parts=tuple(parts), spans=())


def ablation_masks(n: int, d: int, keep_prob: float = 0.5, seed: int = 0) -> np.ndarray:
    """[n, d] bool: n random ablations of d sources, True = kept.

    Mask i uses RandomState(seed + i), as the reference's _create_mask does.
    """
    masks = np.zeros((n, d), dtype=bool)
    for i in range(n):
        rng = np.random.RandomState(seed + i)
        masks[i] = rng.choice([False, True], size=d, p=[1 - keep_prob, keep_prob])
    return masks


def logit_q(s: torch.Tensor, answer: int) -> float:
    """logit q(answer) = log q - log(1 - q) from the candidate scores s [K].

    Computed as s_answer - logsumexp(other s), which stays exact when q rounds to 1.
    """
    s = s.double()
    return (s[answer] - torch.cat([s[:answer], s[answer + 1:]]).logsumexp(0)).item()


@torch.no_grad()
def ablation_targets(model, tok, example: schema.Example, sources: list[dict[str, Any]],
                     masks: np.ndarray, answer: int, device: str = "cuda",
                     batch_tokens: int = 32768, batch_rows: int = 64) -> np.ndarray:
    """[n] float: logit q(answer | ablated prompt) for every mask.

    Options stay in release order, so `answer` is the same option under every mask.
    """
    ablated = [ablate(example, sources, keep) for keep in masks]
    scored = answers.score_examples(model, ablated, tok, device, batch_tokens, batch_rows)
    return np.array([logit_q(s, answer) for s, _, _ in scored])


def fit_surrogate(masks: np.ndarray, y: np.ndarray, alpha: float = 0.01,
                  num_output_tokens: int = 1) -> tuple[np.ndarray, float]:
    """Sparse linear surrogate y ~ b + w . mask: one weight per source.

    Lasso on standardized masks, with w and b mapped back to 0/1 masks. y is a
    single decision rather than a response, so num_output_tokens is 1.
    """
    X = masks.astype(np.float32)
    scaler = StandardScaler()
    lasso = Lasso(alpha=alpha, random_state=0, fit_intercept=True)
    make_pipeline(scaler, lasso).fit(X, y / num_output_tokens)
    weight = lasso.coef_ / scaler.scale_
    bias = lasso.intercept_ - (scaler.mean_ / scaler.scale_) @ lasso.coef_.T
    return weight * num_output_tokens, float(bias * num_output_tokens)


def predict(model, tok, example: schema.Example,
            device: str = "cuda") -> tuple[list[int], int, torch.Tensor]:
    """The prompt (options in release order), the predicted option, and the scores s [K]."""
    prompt = answers.prompt_ids(example, tok)
    s, _, _ = answers.score(model, prompt, tok, len(example.options), device)
    return prompt, int(s.argmax()), s


def answer_positions(tok, example: schema.Example, prompt: list[int],
                     answer: int) -> tuple[list[int], list[int]]:
    """Option `answer`'s suffix tokens ("k]") and the positions whose logits
    predict them, in the prompt followed by that suffix."""
    suffix = answers.candidates(tok, len(example.options))[0][answer]
    return suffix, list(range(len(prompt) - 1, len(prompt) - 1 + len(suffix)))


def attribute(model, tok, example: schema.Example, num_ablations: int = 64, keep_prob: float = 0.5,
              alpha: float = 0.01, seed: int = 0, device: str = "cuda") -> dict[str, Any]:
    """ContextCite on one example (the reference's get_attributions).

    The attributed answer is the model's prediction on the full example.
    Returns {"answer", "q" (its tree probability), "sources" (split_sources
    with each source's weight "w"), "bias"}. A weight is how much keeping that
    sentence adds to logit q(answer).
    """
    _, answer, s = predict(model, tok, example, device)
    sources = split_sources(example)
    if not sources:  # nothing to cite: the decision alone
        return no_sources_result(s, answer)
    masks = ablation_masks(num_ablations, len(sources), keep_prob, seed)
    y = ablation_targets(model, tok, example, sources, masks, answer, device)
    return result(s, answer, sources, masks, y, alpha)


def no_sources_result(s: torch.Tensor, answer: int) -> dict[str, Any]:
    """attribute's output for an example with no sentences: the decision
    (from the candidate scores s [K]) and no sources."""
    return {"answer": answer, "q": torch.softmax(s.float(), 0)[answer].item(),
            "sources": [], "bias": 0.0}


def result(s: torch.Tensor, answer: int, sources: list[dict[str, Any]], masks: np.ndarray,
           y: np.ndarray, alpha: float = 0.01) -> dict[str, Any]:
    """attribute's output from the full prompt's scores s [K] and the ablation targets y [n] for masks [n, S]."""
    w, b = fit_surrogate(masks, y, alpha)
    return {"answer": answer, "q": torch.softmax(s.float(), 0)[answer].item(),
            "sources": [{**src, "w": float(v)} for src, v in zip(sources, w)], "bias": b}


def to_spans(sources: list[dict[str, Any]]) -> tuple[schema.Span, ...]:
    """Sources as spans of the example, in document order."""
    return tuple(schema.Span(s["part"], s["start"], s["end"])
                 for s in sorted(sources, key=lambda s: (s["part"], s["start"])))


def select(result: dict[str, Any], k: int = 3) -> tuple[schema.Span, ...]:
    """The predicted evidence: the k highest-weighted sentences with w > 0,
    as spans of the example (to_spans)."""
    top = sorted((s for s in result["sources"] if s["w"] > 0), key=lambda s: -s["w"])[:k]
    return to_spans(top)


def gold_sentences(example: schema.Example, sources: list[dict[str, Any]]) -> set[int]:
    """Indices of the sources that overlap any of the example's gold spans."""
    return {i for i, s in enumerate(sources) for sp in example.spans
            if sp.part == s["part"] and s["start"] < sp.end and s["end"] > sp.start}


def span_metrics(example: schema.Example, result: dict[str, Any], k: int = 3,
                 beta: float = 1.0) -> dict[str, float] | None:
    """Sentence-level scores against the gold spans (None without gold spans).

    precision, recall and f_beta of the top-k selection (beta < 1 favours
    precision); ap: average precision of the full weight ranking; top1:
    whether the highest-weighted sentence is gold.
    """
    gold = gold_sentences(example, result["sources"])
    if not gold:
        return None
    ranked = sorted(range(len(result["sources"])), key=lambda i: -result["sources"][i]["w"])
    chosen = [i for i in ranked[:k] if result["sources"][i]["w"] > 0]
    hits = len(set(chosen) & gold)
    p = hits / len(chosen) if chosen else 0.0
    r = hits / len(gold)
    f = (1 + beta ** 2) * p * r / (beta ** 2 * p + r) if p + r else 0.0
    found, ap = 0, 0.0
    for rank, i in enumerate(ranked, 1):
        if i in gold:
            found += 1
            ap += found / rank
    return {"precision": p, "recall": r, "f_beta": f, "ap": ap / len(gold),
            "top1": float(ranked[0] in gold)}
