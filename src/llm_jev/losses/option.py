"""Encoder recipes: how option scores s [K] become a training loss.

Every recipe normalises a per-option value over the K options and takes
-log p_gold, so the options always compete and the decision is argmax s:

- "listwise": p_k = softmax(s)_k.
- "sigmoid_norm": p_k = sigmoid(s_k) / sum_j sigmoid(s_j), the form of
  DeepSeek-V3's expert gate.
- "sqrt_softplus": p_k = h(s_k) / sum_j h(s_j), h(s) = sqrt(softplus(s)),
  unbounded above, so an accepted option keeps a slowly fading gradient.

The last two need s = logit(yes) - logit(no): raw MLM logits are ~200 in
size, where every sigmoid saturates at 1.
"""

from collections.abc import Callable

import torch
from torch import nn


def log_sqrt_softplus(s: torch.Tensor) -> torch.Tensor:
    """Log sqrt(softplus(s)), stable: softplus(s) ~ e^s for very negative s."""
    softplus = nn.functional.softplus(s)
    return 0.5 * torch.where(s < -20, s, softplus.clamp_min(1e-30).log())


def log_weights(s: torch.Tensor, kind: str) -> torch.Tensor:
    """Log g(s_k), whose normalisation over the options gives p [K].

    Args:
      s: Option scores [K].
      kind: A key of LOSSES.

    Raises:
      ValueError: On an unknown kind.
    """
    if kind == "listwise":
        return s
    if kind == "sigmoid_norm":
        return nn.functional.logsigmoid(s)
    if kind == "sqrt_softplus":
        return log_sqrt_softplus(s)
    raise ValueError(f"loss is one of {tuple(LOSSES)}, not {kind!r}")


def probabilities(s: torch.Tensor, kind: str) -> torch.Tensor:
    """The recipe's distribution p [K] over the options."""
    return torch.softmax(log_weights(s, kind), 0)


def _normalised_nll(kind: str) -> Callable[[torch.Tensor, int], torch.Tensor]:
    def nll(s: torch.Tensor, gold: int) -> torch.Tensor:
        log_g = log_weights(s, kind)
        return log_g.logsumexp(0) - log_g[gold]

    return nll


LOSSES: dict[str, Callable[[torch.Tensor, int], torch.Tensor]] = {
    name: _normalised_nll(name)
    for name in ("listwise", "sigmoid_norm", "sqrt_softplus")
}
NEEDS_NO_TOKEN = frozenset({"sigmoid_norm", "sqrt_softplus"})


def loss(s: torch.Tensor, gold: int, kind: str = "listwise") -> torch.Tensor:
    """-log p_gold under recipe `kind`, from scores s [K]. Returns [].

    Raises:
      ValueError: On an unknown kind.
    """
    if kind not in LOSSES:
        raise ValueError(f"loss is one of {tuple(LOSSES)}, not {kind!r}")
    return LOSSES[kind](s, gold)
