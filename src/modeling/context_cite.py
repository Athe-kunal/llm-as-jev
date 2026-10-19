"""ContextCite for the decision interface: which context sentences an answer depends on.

Cohen-Wang et al., "ContextCite: Attributing Model Generation to Context"
(NeurIPS 2024), following github.com/MadryLab/context-cite (MIT). Sources
are the sentences of the row's parts; ablated rows keep a random subset of
them, the model's predicted answer is re-scored on each, and a sparse linear
surrogate gives one weight per sentence. The target is the logit of the tree
probability q(answer | x) (answers.score_batch), not raw token
probabilities. See docs/architecture.md.
"""

import functools
import re
from typing import Any

import nltk
import numpy as np
import torch
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dataloader import answers


@functools.cache
def _punkt() -> None:
    """Fetches nltk's sentence model once (the reference does it on import)."""
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)


# A word for has_content: two or more letters (no digits or underscores).
_WORD = re.compile(r"[^\W\d_]{2,}")


def has_content(text: str, min_words: int = 2, min_alnum: float = 0.5) -> bool:
    """Whether a sentence carries content: at least min_words words and at
    least min_alnum of its characters letters or digits. Blank form fields
    ("Agent:____"), checkbox lines and fragments of abbreviations split by
    nltk ("(C.A.R.") do not."""
    alnum = sum(ch.isalnum() for ch in text) / max(len(text), 1)
    return len(_WORD.findall(text)) >= min_words and alnum >= min_alnum


def split_sources(row: dict[str, Any], min_words: int = 2,
                  min_alnum: float = 0.5) -> list[dict[str, Any]]:
    """The row's sources: the sentences of every part, in order, each with content.

    As the reference's split_text: each line of a part is split with
    nltk.sent_tokenize and each sentence is located after the previous one.
    Unlike the reference, a sentence without content (has_content) joins the
    previous source of its part, or the next one at the start of a part (a
    part with no content sentence is one source), so every source can carry
    evidence and the text is unchanged. A source is {"part": part index,
    "start", "end": its char range in that part's text (the row's spans use
    the same offsets), "sep": the text between the previous source of the
    part, or the part start, and it}.
    """
    _punkt()
    sources = []
    for i, part in enumerate(row["parts"]):
        text, cur, own, lead = part["text"], 0, [], None
        for line in text.splitlines():
            for sentence in nltk.sent_tokenize(line):
                start = text.find(sentence, cur)
                if start < 0:
                    raise ValueError(f"{row['id']}: sentence not found in part {i}")
                s = {"part": i, "start": start, "end": start + len(sentence), "sep": text[cur:start]}
                cur = s["end"]
                if not has_content(sentence, min_words, min_alnum):
                    if own:
                        own[-1]["end"] = s["end"]          # joins the previous source
                    elif lead:
                        lead["end"] = s["end"]
                    else:
                        lead = s                           # waits for the next source
                    continue
                if lead:
                    s = {**s, "start": lead["start"], "sep": lead["sep"]}
                    lead = None
                own.append(s)
        sources += own or ([lead] if lead else [])
    return sources


def ablate(row: dict[str, Any], sources: list[dict[str, Any]], keep: list[bool]) -> dict[str, Any]:
    """The row with only the kept sources (the reference's get_context).

    Each kept sentence is preceded by its own separator, except the first
    kept one of its part. Unlike the reference, the text before a part's
    first sentence and after its last (whitespace) is always kept, so keeping
    every source rebuilds the row exactly. Spans are dropped: their offsets
    no longer hold.
    """
    kept: list[list[str]] = [[] for _ in row["parts"]]
    for s, k in zip(sources, keep):
        if k:
            text = row["parts"][s["part"]]["text"]
            kept[s["part"]].append((s["sep"] if kept[s["part"]] else "") + text[s["start"]:s["end"]])
    parts = []
    for i, p in enumerate(row["parts"]):
        own = [s for s in sources if s["part"] == i]
        lead = p["text"][: own[0]["start"]] if own else p["text"]
        tail = p["text"][own[-1]["end"]:] if own else ""
        parts.append({**p, "text": lead + "".join(kept[i]) + tail})
    return {**row, "parts": parts, "spans": []}


def ablation_masks(n: int, d: int, keep_prob: float = 0.5, seed: int = 0) -> np.ndarray:
    """[n, d] bool: n random ablations of d sources, True = kept.

    The reference's _create_mask: mask i comes from its own
    np.random.RandomState(seed + i), each source kept with keep_prob, so the
    same seed gives the reference's masks.
    """
    masks = np.zeros((n, d), dtype=bool)
    for i in range(n):
        rng = np.random.RandomState(seed + i)
        masks[i] = rng.choice([False, True], size=d, p=[1 - keep_prob, keep_prob])
    return masks


def logit_q(s: torch.Tensor, answer: int) -> float:
    """logit q(answer) = log q - log(1 - q) from the candidate scores s [K]:
    s_answer - logsumexp(other s), exact even when q rounds to 1 (the
    reference's correct-minus-logsumexp-of-the-rest, per token there)."""
    s = s.double()
    return (s[answer] - torch.cat([s[:answer], s[answer + 1:]]).logsumexp(0)).item()


@torch.no_grad()
def ablation_targets(model, tok, row: dict[str, Any], sources: list[dict[str, Any]],
                     masks: np.ndarray, answer: int, device: str = "cuda",
                     batch_tokens: int = 32768, batch_rows: int = 64) -> np.ndarray:
    """[n] float: logit q(answer | ablated prompt) for every mask.

    The reference's get_masks_and_logit_probs + aggregate_logit_probs, with
    the tree probability q of answers.score_batch in place of the response
    tokens' probabilities. Options stay in release order (no shuffling), so
    `answer` is the same option under every mask; prompts are scored in
    length-sorted batches (answers.batches).
    """
    prompts = []
    for keep in masks:
        ex = answers.encode(ablate(row, sources, keep), tok, None)
        prompts.append(ex["input_ids"][: ex["path"][0][0] + 1])
    k = len(row["labels"])
    y = np.empty(len(prompts))
    for batch in answers.batches([len(p) for p in prompts], batch_tokens, batch_rows):
        scored = answers.score_batch(model, [prompts[i] for i in batch], tok, [k] * len(batch),
                                     device, batch_tokens, batch_rows)
        for i, (s, _, _) in zip(batch, scored):
            y[i] = logit_q(s, answer)
    return y


def fit_surrogate(masks: np.ndarray, y: np.ndarray, alpha: float = 0.01,
                  num_output_tokens: int = 1) -> tuple[np.ndarray, float]:
    """Sparse linear surrogate y ~ b + w . mask: one weight per source.

    The reference's LassoRegression.fit: masks standardised (StandardScaler),
    Lasso(alpha) with an intercept, then w and b mapped back to 0/1 masks.
    The reference divides y by the response length before the fit and
    multiplies w, b back; our y is one decision, so num_output_tokens = 1.
    """
    X = masks.astype(np.float32)
    scaler = StandardScaler()
    lasso = Lasso(alpha=alpha, random_state=0, fit_intercept=True)
    make_pipeline(scaler, lasso).fit(X, y / num_output_tokens)
    weight = lasso.coef_ / scaler.scale_
    bias = lasso.intercept_ - (scaler.mean_ / scaler.scale_) @ lasso.coef_.T
    return weight * num_output_tokens, float(bias * num_output_tokens)


def predict(model, tok, row: dict[str, Any], device: str = "cuda") -> tuple[list[int], int, torch.Tensor]:
    """The prompt (options in release order), the predicted option (argmax
    of answers.score) and the candidate scores s [K]."""
    ex = answers.encode(row, tok, None)
    prompt = ex["input_ids"][: ex["path"][0][0] + 1]
    s, _, _ = answers.score(model, prompt, tok, ex["n_options"], device)
    return prompt, int(s.argmax()), s


def attribute(model, tok, row: dict[str, Any], num_ablations: int = 64, keep_prob: float = 0.5,
              alpha: float = 0.01, seed: int = 0, device: str = "cuda") -> dict[str, Any]:
    """ContextCite on one row (the reference's get_attributions).

    The attributed answer is the model's prediction on the full row (argmax
    of answers.score, options in release order). Returns {"answer": its
    option index, "q": its tree probability, "sources": split_sources(row)
    with each source's weight "w", "bias": the surrogate's intercept}. A
    weight is how much keeping that sentence adds to logit q(answer).
    """
    _, answer, s = predict(model, tok, row, device)
    sources = split_sources(row)
    masks = ablation_masks(num_ablations, len(sources), keep_prob, seed)
    y = ablation_targets(model, tok, row, sources, masks, answer, device)
    w, b = fit_surrogate(masks, y, alpha)
    return {"answer": answer, "q": torch.softmax(s.float(), 0)[answer].item(),
            "sources": [{**src, "w": float(v)} for src, v in zip(sources, w)], "bias": b}


def select(row: dict[str, Any], result: dict[str, Any], k: int = 3) -> list[dict[str, Any]]:
    """The predicted evidence: the k highest-weighted sentences with w > 0,
    in the row's span format {"part": part name, "start", "end", "text"}."""
    top = sorted((s for s in result["sources"] if s["w"] > 0), key=lambda s: -s["w"])[:k]
    out = []
    for s in sorted(top, key=lambda s: (s["part"], s["start"])):
        part = row["parts"][s["part"]]
        out.append({"part": part["name"], "start": s["start"], "end": s["end"],
                    "text": part["text"][s["start"]:s["end"]]})
    return out


def gold_sentences(row: dict[str, Any], sources: list[dict[str, Any]]) -> set[int]:
    """Indices of the sources that overlap any of the row's gold spans."""
    names = [p["name"] for p in row["parts"]]
    return {i for i, s in enumerate(sources) for sp in row["spans"]
            if names.index(sp["part"]) == s["part"] and s["start"] < sp["end"] and s["end"] > sp["start"]}


def span_metrics(row: dict[str, Any], result: dict[str, Any], k: int = 3,
                 beta: float = 1.0) -> dict[str, float] | None:
    """Sentence-level scores of the attributions against the gold spans
    (None for a row without gold spans).

    precision, recall and f_beta = (1 + beta^2) P R / (beta^2 P + R) of the
    top-k selection (beta < 1 favours precision; beta = 1 is F1); ap: average
    precision of the full weight ranking (threshold-free; ties broken by
    sentence order); top1: whether the highest-weighted sentence is gold.
    """
    gold = gold_sentences(row, result["sources"])
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
