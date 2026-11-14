"""ContextCite by deletion, for any decision model.

Cohen-Wang et al., ContextCite (NeurIPS 2024), following
github.com/MadryLab/context-cite (MIT). The attributed answer is the model's
prediction on the full row. Random ablations keep about half of the
sentences, each ablated row is re-scored, and a sparse linear surrogate of
logit q(answer) gives one weight per sentence: how much keeping that
sentence adds to logit q(answer).
"""

from collections.abc import Sequence
import dataclasses
import math

import numpy as np
import torch

from llm_jev.data import schema
from llm_jev.evidence import sentences as sentences_lib
from llm_jev.models import base


@dataclasses.dataclass(frozen=True)
class Attribution:
    """Per-sentence weights for a model's answer.

    Attributes:
      answer: The attributed option (release order).
      q: softmax(s)[answer] on the full row.
      sentences: The row's sentences.
      weights: One weight per sentence; higher = more support.
      bias: The surrogate's intercept (ContextCite only).
    """

    answer: int
    q: float
    sentences: list[sentences_lib.Sentence]
    weights: np.ndarray
    bias: float = 0.0


def ablation_count(num_sentences: int, low: int = 32, high: int = 256) -> int:
    """About two ablations per sentence, a power of two in [low, high]."""
    return min(high, max(low, 1 << math.ceil(math.log2(2 * num_sentences))))


def logit_q(s: torch.Tensor, answer: int) -> float:
    """Logit q(answer) = log q - log(1 - q) from the option scores s [K].

    Computed as s_answer - logsumexp(other s), exact when q rounds to 1.
    """
    s = s.double()
    others = torch.cat([s[:answer], s[answer + 1 :]])
    return (s[answer] - others.logsumexp(0)).item()


def ablation_targets(
    model: base.DecisionModel,
    example: schema.Example,
    sentences: Sequence[sentences_lib.Sentence],
    masks: np.ndarray,
    answer: int,
) -> np.ndarray:
    """[n] logit q(answer | ablated row) for every mask (True = kept).

    Options stay in release order, so `answer` is the same option under
    every mask.
    """
    ablated = [
        sentences_lib.ablate(example, sentences, list(keep)) for keep in masks
    ]
    return np.array([logit_q(s, answer) for s in model.score(ablated)])


def fit_surrogate(
    masks: np.ndarray, y: np.ndarray, alpha: float = 0.01
) -> tuple[np.ndarray, float]:
    """Sparse linear surrogate y ~ b + w . mask: one weight per sentence.

    Lasso on standardized masks, with w and b mapped back to 0/1 masks.
    """
    from sklearn.linear_model import Lasso
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    lasso = Lasso(alpha=alpha, random_state=0, fit_intercept=True)
    make_pipeline(scaler, lasso).fit(masks.astype(np.float32), y)
    weight = lasso.coef_ / scaler.scale_
    bias = lasso.intercept_ - (scaler.mean_ / scaler.scale_) @ lasso.coef_.T
    return weight, float(bias)


def attribute(
    model: base.DecisionModel,
    example: schema.Example,
    num_ablations: int | None = None,
    keep_prob: float = 0.5,
    alpha: float = 0.01,
    seed: int = 0,
) -> Attribution:
    """ContextCite on one row.

    Args:
      model: Any decision model.
      example: The row.
      num_ablations: Ablations (None: `ablation_count` of the sentences).
      keep_prob: Probability that an ablation keeps a sentence.
      alpha: Lasso regularisation.
      seed: Seeds the ablation masks.
    """
    s = model.score([example])[0]
    answer = int(s.argmax())
    q = torch.softmax(s.float(), 0)[answer].item()
    sentences = sentences_lib.split(example)
    if not sentences:
        return Attribution(answer, q, [], np.zeros(0))
    n = num_ablations or ablation_count(len(sentences))
    masks = sentences_lib.ablation_masks(n, len(sentences), keep_prob, seed)
    y = ablation_targets(model, example, sentences, masks, answer)
    weights, bias = fit_surrogate(masks, y, alpha)
    return Attribution(answer, q, sentences, weights, bias)
