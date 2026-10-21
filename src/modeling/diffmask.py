"""DiffMask for the decision interface: which document tokens the decision needs.

De Cao et al., "How do Decisions Emerge across Layers in Neural Models?
Interpretation with Differentiable Masking" (EMNLP 2020). A probe on the
frozen decision model's hidden states gives every document token a gate z in
[0, 1]; the document is re-run with the closed gates' tokens hidden, and the
probe is trained to close as many gates as possible while the tree
distribution q over the options stays within a KL margin of the unmasked one.
No labels: the training signal is the model's own decision. See
docs/architecture.md.
"""

import functools
import math

import torch


class HardConcrete:
    """Louizos et al. (2018) gates: a stretched, clipped binary Concrete.

    A gate is s = sigmoid((logit(u) + log_alpha) / beta) for u ~ U(0, 1),
    stretched to (l, r) = (-0.1, 1.1) and clipped to [0, 1], so it is exactly
    0 or 1 with positive probability and differentiable in log_alpha in
    between (Louizos et al.'s defaults for beta, l and r).
    """

    def __init__(self, beta: float = 2 / 3, l: float = -0.1, r: float = 1.1):
        self.beta, self.l, self.r = beta, l, r

    def sample(self, log_alpha: torch.Tensor) -> torch.Tensor:
        """Gates z [...] in [0, 1], reparameterised (training)."""
        u = torch.rand_like(log_alpha).clamp(1e-6, 1 - 1e-6)
        s = torch.sigmoid((u.log() - (-u).log1p() + log_alpha) / self.beta)
        return (s * (self.r - self.l) + self.l).clamp(0, 1)

    def gate(self, log_alpha: torch.Tensor) -> torch.Tensor:
        """Deterministic gates (inference): the stretched sigmoid, clipped."""
        return (torch.sigmoid(log_alpha) * (self.r - self.l) + self.l).clamp(0, 1)

    def open_prob(self, log_alpha: torch.Tensor) -> torch.Tensor:
        """P(z > 0) per gate, in closed form; its sum is the expected L0 norm."""
        return torch.sigmoid(log_alpha - self.beta * math.log(-self.l / self.r))


class GateProbe(torch.nn.Module):
    """log_alpha per token from every layer's hidden state of the unmasked pass.

    As DiffMask's probe: one small MLP per hidden state (the embeddings and
    each decoder layer), outputs summed. Each state is RMS-normalised first:
    Qwen's hidden-state norms grow with depth and would otherwise swamp the
    early layers. The output bias starts at init_open, so every gate starts
    almost surely open (P(z > 0) ~ 0.99 at 4.0): training starts from the
    unmasked decision, where the KL is 0.
    """

    def __init__(self, num_states: int, hidden: int, width: int = 128, init_open: float = 4.0,
                 bound: float = 6.0):
        super().__init__()
        self.bound = bound
        self.w1 = torch.nn.Parameter(torch.randn(num_states, hidden, width) / hidden ** 0.5)
        self.b1 = torch.nn.Parameter(torch.zeros(num_states, 1, width))
        self.w2 = torch.nn.Parameter(torch.zeros(num_states, width))   # zero: log_alpha starts at the bias
        self.bias = torch.nn.Parameter(torch.tensor(init_open))

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """states [S, T, H] (S hidden states of T tokens) -> log_alpha [T]."""
        x = states.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)     # [S, T, H]
        h = torch.tanh(torch.einsum("sth,shw->stw", x, self.w1) + self.b1)  # [S, T, W]
        raw = torch.einsum("stw,sw->t", h, self.w2) + self.bias        # [T]
        # soft bound: every gate keeps P(z > 0) >= ~3%, so the KL gradient can still
        # reach it (a gate whose samples are all clipped to 0 gets no gradient)
        return self.bound * torch.tanh(raw / self.bound)


# ---- hiding gated tokens ------------------------------------------------------
# Two ways, compared in the experiment: "fade" scales every attention weight
# that targets a gated token by its gate z (log z added to the score, at
# every layer, so the token fades from all later tokens, the document
# included); "interp" replaces the token's input embedding by
# z * e + (1 - z) * b, b a learned vector (the paper's input masking).

ATTENTION = "flex_gated"


@functools.cache
def _flex():
    from torch.nn.attention.flex_attention import flex_attention

    return torch.compile(flex_attention, dynamic=True)


def gated_flex_attention(module, query, key, value, attention_mask, scaling=None, key_bias=None, **kwargs):
    """Flex attention under a BlockMask, plus key_bias [N] added to every score
    that targets key k (0 off the gated tokens). Registered as ATTENTION; the
    model's forward passes key_bias through to every layer. Returns [B, T, H, D]
    as transformers' attention functions do."""
    score_mod = None
    if key_bias is not None:
        def score_mod(score, b, h, q, k):
            return score + key_bias[k]
    out = _flex()(query, key, value, score_mod=score_mod, block_mask=attention_mask,
                  scale=scaling, enable_gqa=True)
    return out.transpose(1, 2).contiguous(), None


def use_gated_attention(model) -> None:
    """Switches the model to gated_flex_attention."""
    from transformers import AttentionInterface

    AttentionInterface.register(ATTENTION, gated_flex_attention)
    model.set_attn_implementation(ATTENTION)


def fade_bias(z: torch.Tensor, gated: torch.Tensor, length: int) -> torch.Tensor:
    """key_bias [length]: log z on the gated positions, 0 elsewhere. z = 0 is
    clamped to log 1e-6, a weight of 10^-6 (the gradient is 0 there anyway:
    HardConcrete clips)."""
    bias = z.new_zeros(length)
    return bias.index_put((gated,), z.clamp_min(1e-6).log())


def interpolate(embeddings: torch.Tensor, z: torch.Tensor, gated: torch.Tensor,
                baseline: torch.Tensor) -> torch.Tensor:
    """inputs_embeds [N, H]: the gated rows become z * e + (1 - z) * baseline."""
    e = embeddings[gated]
    return embeddings.index_put((gated,), z[:, None].to(e.dtype) * e + (1 - z[:, None]).to(e.dtype) * baseline.to(e.dtype))
