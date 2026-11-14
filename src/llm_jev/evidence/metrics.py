"""Scores of predicted evidence against gold spans: sentences, tokens, words."""

from collections.abc import Sequence
import re

import numpy as np

from llm_jev.data import schema
from llm_jev.evidence import sentences as sentences_lib


def span_metrics(
    example: schema.Example,
    sentences: Sequence[sentences_lib.Sentence],
    weights: Sequence[float],
    k: int = 3,
    beta: float = 1.0,
) -> dict[str, float] | None:
    """Sentence-level scores of a weighting against the gold spans.

    Args:
      example: The row, with gold spans.
      sentences: Its sentences.
      weights: One weight per sentence; higher = more evidence.
      k: Size of the top-k selection (positive weights only).
      beta: F-beta weight (< 1 favours precision).

    Returns:
      precision, recall and f_beta of the top-k selection; ap, the average
      precision of the full ranking; top1, whether the highest-weighted
      sentence is gold. None without gold sentences.
    """
    gold = sentences_lib.gold_indices(example, sentences)
    if not gold:
        return None
    ranked = sorted(range(len(sentences)), key=lambda i: -weights[i])
    chosen = [i for i in ranked[:k] if weights[i] > 0]
    hits = len(set(chosen) & gold)
    precision = hits / len(chosen) if chosen else 0.0
    recall = hits / len(gold)
    denominator = beta**2 * precision + recall
    f_beta = (
        (1 + beta**2) * precision * recall / denominator if denominator else 0.0
    )
    found, ap = 0, 0.0
    for rank, i in enumerate(ranked, 1):
        if i in gold:
            found += 1
            ap += found / rank
    return {
        "precision": precision,
        "recall": recall,
        "f_beta": f_beta,
        "ap": ap / len(gold),
        "top1": float(ranked[0] in gold),
    }


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """Average precision of `scores` against binary labels (NaN if none)."""
    from sklearn.metrics import average_precision_score

    if not labels.any():
        return float("nan")
    return float(average_precision_score(labels, scores))


def word_scores(
    text: str, tokens: Sequence[tuple[int, int, float]]
) -> list[tuple[int, int, float]]:
    """(start, end, score) per word (a run of non-space characters).

    A word's score is the highest score of the (start, end, score) tokens
    inside it; words without a token are skipped.
    """
    out = []
    ordered = sorted(tokens)
    i = 0
    for match in re.finditer(r"\S+", text):
        while i < len(ordered) and ordered[i][1] <= match.start():
            i += 1
        inside, j = [], i
        while j < len(ordered) and ordered[j][0] < match.end():
            inside.append(ordered[j][2])
            j += 1
        if inside:
            out.append((match.start(), match.end(), max(inside)))
    return out


def precision_recall_f1(
    scores: np.ndarray, labels: np.ndarray, threshold: float
) -> tuple[float, float, float]:
    """Precision, recall and F1 of `scores >= threshold` against labels."""
    selected = scores >= threshold
    true_positives = float((selected & labels).sum())
    precision = true_positives / max(selected.sum(), 1)
    recall = true_positives / max(labels.sum(), 1)
    return (
        precision,
        recall,
        2 * precision * recall / max(precision + recall, 1e-12),
    )
