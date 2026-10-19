"""AT2: attribution with attention, learned from ContextCite-style ablations.

Cohen-Wang et al., "Learning to Attribute with Attention" (2025,
arXiv 2504.13752), following github.com/MadryLab/AT2 (MIT). A source's score
is a learned weighted sum, over every (layer, head), of the attention from
the answer's positions to the source's tokens; the weights are fitted so
that the summed scores of the kept sources correlate (Pearson) with the
answer's logit q under random ablations (context_cite.ablation_targets).
Written for Qwen3 (per-head q_norm / k_norm). See docs/architecture.md.
"""

import sys

import numpy as np
import torch
from torch import nn

from dataloader import answers
from modeling import context_cite


@torch.no_grad()
def attention_weights(model, input_ids: list[int], queries: list[int],
                      device: str = "cuda") -> torch.Tensor:
    """[L, H, Q, N] attention probabilities from the query positions to all N
    tokens, for every layer and head.

    One forward pass with the model's own attention (sdpa); hooks on every
    layer's q_norm / k_norm keep the normalised queries of the Q rows and
    all keys, then RoPE and a causal softmax give the Q rows of each
    attention matrix. The reference (get_attention_weights) recomputes q and
    k from the hidden states instead; the N x N matrix is never stored.
    """
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    n = len(input_ids)
    q_at = torch.tensor(queries, device=device)
    qs, ks, hooks = [], [], []
    for layer in base.model.layers:
        attn = layer.self_attn
        hooks.append(attn.q_norm.register_forward_hook(lambda m, i, o: qs.append(o[:, q_at])))
        hooks.append(attn.k_norm.register_forward_hook(lambda m, i, o: ks.append(o)))
    try:
        base.model(input_ids=torch.tensor([input_ids], device=device), use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    pos = torch.arange(n, device=device)[None]
    cos, sin = base.model.rotary_emb(ks[0], pos)                                # [1, N, d]
    causal = torch.arange(n, device=device)[None, :] > q_at[:, None]          # [Q, N]
    out = []
    for layer, q, k in zip(base.model.layers, qs, ks):
        attn = layer.self_attn
        helpers = sys.modules[type(attn).__module__]
        q, _ = helpers.apply_rotary_pos_emb(q.transpose(1, 2), q.transpose(1, 2),
                                            cos[:, q_at], sin[:, q_at])        # [1, H, Q, d]
        _, k = helpers.apply_rotary_pos_emb(k.transpose(1, 2), k.transpose(1, 2), cos, sin)
        k = helpers.repeat_kv(k, attn.num_key_value_groups)                     # [1, H, N, d]
        scores = (q[0] @ k[0].transpose(1, 2)).float() * attn.scaling           # [H, Q, N]
        out.append(scores.masked_fill(causal, -torch.inf).softmax(-1))
    return torch.stack(out)


def source_token_ranges(row: dict, tok, sources: list[dict]) -> list[tuple[int, int]]:
    """Each source's [start, end) token range in the row's prompt.

    The prompt is answers.build_prompt with options in release order (as
    answers.encode without shuffling); build_prompt's part offsets place each
    sentence in it. As the reference: from the token holding the sentence's
    first character to the one holding its last, so a token that straddles
    two sentences belongs to both.
    """
    text, where = answers.build_prompt(row, list(row["labels"]), tok)
    enc = tok(text, add_special_tokens=False)
    names = [p["name"] for p in row["parts"]]
    out = []
    for s in sources:
        offset = where[names[s["part"]]][0]
        out.append((enc.char_to_token(offset + s["start"]),
                    enc.char_to_token(offset + s["end"] - 1) + 1))
    return out


def source_features(attn: torch.Tensor, ranges: list[tuple[int, int]]) -> torch.Tensor:
    """[S, L*H]: each source's attention features (the reference's
    features_by_target_source).

    attn [L, H, Q, N] from attention_weights is averaged over the answer's Q
    positions and summed over each source's tokens; feature l*H + h is layer
    l, head h, as the reference orders them.
    """
    per_token = attn.mean(2).flatten(0, 1).T                                   # [N, L*H]
    cum = torch.cat([per_token.new_zeros(1, per_token.shape[1]), per_token.cumsum(0)])
    return torch.stack([cum[b] - cum[a] for a, b in ranges])                   # [S, L*H]


def example_data(model, tok, row: dict, num_masks: int = 32, seed: int = 0,
                 device: str = "cuda") -> tuple[torch.Tensor, np.ndarray]:
    """One training example: per-mask features [num_masks, L*H] and targets [num_masks].

    The reference's compute_features_and_outputs_for_example: the attributed
    answer is the model's prediction; features come from the unablated
    prompt followed by that answer's suffix; the targets are logit q of the
    answer under num_masks ablations (context_cite). As the reference stores
    them, mask j's features are (2 * mask_j - 1) @ source features, so the
    surrogate's prediction is features @ theta and the sentence count drops
    out.
    """
    prompt, answer, _ = context_cite.predict(model, tok, row, device)
    suffix = answers._candidates(tok, len(row["labels"]))[0][answer]
    queries = list(range(len(prompt) - 1, len(prompt) - 1 + len(suffix)))
    sources = context_cite.split_sources(row)
    feats = source_features(attention_weights(model, prompt + suffix, queries, device),
                            source_token_ranges(row, tok, sources))                # [S, L*H]
    masks = context_cite.ablation_masks(num_masks, len(sources), 0.5, seed)
    y = context_cite.ablation_targets(model, tok, row, sources, masks, answer, device)
    signed = torch.tensor(masks, device=feats.device, dtype=feats.dtype) * 2 - 1    # [M, S]
    return signed @ feats, y


class ScoreEstimator(nn.Module):
    """The reference's LinearScoreEstimator: a sentence's score is theta .
    phi_j, one weight per (layer, head), no bias, initialised to 1/F (the
    average-attention baseline)."""

    def __init__(self, num_features: int):
        super().__init__()
        self.linear = nn.Linear(num_features, 1, bias=False)
        nn.init.constant_(self.linear.weight, 1 / num_features)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """[..., F] features -> [...] scores."""
        return self.linear(features)[..., 0]

    def theta(self) -> torch.Tensor:
        """[F] theta, L1-normalised as the reference does after training (the
        scale is irrelevant under the Pearson loss)."""
        w = self.linear.weight.detach()[0]
        return w / w.abs().sum()


@torch.no_grad()
def attribute(model, tok, row: dict, theta: torch.Tensor, device: str = "cuda") -> dict:
    """AT2 on one row, in context_cite.attribute's format: {"answer", "q",
    "sources" each with its score "w"}. One forward pass: the attention from
    the predicted answer's positions, scored with theta."""
    prompt, answer, s = context_cite.predict(model, tok, row, device)
    suffix = answers._candidates(tok, len(row["labels"]))[0][answer]
    queries = list(range(len(prompt) - 1, len(prompt) - 1 + len(suffix)))
    sources = context_cite.split_sources(row)
    phi = source_features(attention_weights(model, prompt + suffix, queries, device),
                          source_token_ranges(row, tok, sources))                  # [S, L*H]
    w = (phi @ theta.to(phi)).tolist()
    return {"answer": answer, "q": torch.softmax(s.float(), 0)[answer].item(),
            "sources": [{**src, "w": v} for src, v in zip(sources, w)]}
