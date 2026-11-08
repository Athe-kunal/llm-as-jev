"""AT2 for encoders: attribution with attention from the option masks.

Cohen-Wang et al., "Learning to Attribute with Attention" (2025): a source's
score is a learned weighted sum, over (layer, head), of the attention from the
answer's positions to the source's tokens. For the encoder (dataloader.mlm)
the answer's positions are the option masks, inside the prompt, so the
attention comes from the same forward pass that scores the options.
"""

import contextlib

import numpy as np
import torch

from llm_jev.dataloader import mlm
from llm_jev.modeling import context_cite
from llm_jev.modeling import encoder_varlen

CAPTURE = "at2_capture"  # attention implementation name while capturing


def _capture(store: dict):
    """An attention function that records softmax(q k^T) for the rows in `at2_rows`,
    then runs encoder_varlen.attention unchanged."""

    def attention(module, query, key, value, attention_mask, scaling=None,
                  sliding_window=None, at2_rows=None, **kwargs):
        if at2_rows is not None:
            q = query[0][:, at2_rows].float()                       # [H, Q, D]
            k = key[0].float()                                      # [Hkv, N, D]
            k = k.repeat_interleave(q.shape[0] // k.shape[0], 0)    # [H, N, D] (GQA)
            s = (q @ k.transpose(1, 2)) * scaling                   # [H, Q, N]
            if sliding_window is not None:                          # |i - j| <= window (encoder_varlen.window)
                w = encoder_varlen.window(sliding_window)[0]
                j = torch.arange(s.shape[-1], device=s.device)
                s = s.masked_fill((at2_rows[:, None] - j[None]).abs() > w, float("-inf"))
            store[module.layer_idx] = s.softmax(-1)
        return encoder_varlen.attention(module, query, key, value, attention_mask,
                                        scaling=scaling, sliding_window=sliding_window, **kwargs)

    return attention


@contextlib.contextmanager
def capturing(model, store: dict):
    """Switches `model` to the capturing attention for the duration; restores it after.

    Requires the FA2 varlen backend: under a custom implementation transformers
    builds no attention masks, which varlen does not need (windows are kernel
    arguments) but SDPA would (its sliding layers would silently see everything).
    """
    from transformers import AttentionInterface

    before = model.config._attn_implementation
    if before != encoder_varlen.NAME:
        raise ValueError(f"AT2 capture needs model.attention: varlen, not {before!r}")
    AttentionInterface.register(CAPTURE, _capture(store))
    model.set_attn_implementation(CAPTURE)
    try:
        yield store
    finally:
        model.set_attn_implementation(before)


def source_token_ranges(example, tokenizer, fmt, sources: list[dict],
                        order: list[int] | None = None) -> tuple[list[int], list[tuple[int, int]]]:
    """The encoder prompt's token ids, and each source's [start, end) token range in it.

    The prompt is mlm.build_prompt with the options in `order` (default:
    release order), tokenized as mlm.encode does (with special tokens).
    sources come from context_cite.split_sources: character ranges in a part.
    A token that straddles a source boundary belongs to both sources (as in
    the AT2 reference); special tokens (offset (0, 0)) belong to none.
    """
    order = list(range(len(example.options))) if order is None else order
    text, where = mlm.build_prompt(example, order, tokenizer, fmt)
    enc = tokenizer(text, return_offsets_mapping=True)
    starts = torch.tensor([a for a, _ in enc["offset_mapping"]])
    ends = torch.tensor([b for _, b in enc["offset_mapping"]])
    ranges = []
    for s in sources:
        lo = where[s["part"]][0] + s["start"]                        # prompt characters [lo, hi)
        hi = where[s["part"]][0] + s["end"]
        first = int(torch.searchsorted(ends, lo, right=True))        # first token ending after lo
        stop = int(torch.searchsorted(starts, hi, right=False))      # first token starting at/after hi
        ranges.append((first, stop))
    return enc["input_ids"], ranges


def source_features(attn: torch.Tensor, ranges: list[tuple[int, int]]) -> torch.Tensor:
    """[S, L*H]: each source's attention features (AT2's phi_j).

    attn [L, H, Q, N] (mask_attention) is averaged over its Q query rows and
    summed over each source's tokens [start, end); feature l*H + h is layer l,
    head h. A running sum over tokens makes each source one subtraction.
    """
    per_token = attn.mean(2).flatten(0, 1).T                         # [N, L*H]
    cum = torch.cat([per_token.new_zeros(1, per_token.shape[1]), per_token.cumsum(0)])  # [N+1, L*H]
    return torch.stack([cum[b] - cum[a] for a, b in ranges])         # [S, L*H]


def document_tokens(example, tokenizer, fmt, order: list[int] | None = None) -> list[tuple[int, int, int, int]]:
    """(token index, part, start, end) for every prompt token inside a part's text.

    start / end are character offsets in that part (the same offsets as spans),
    clipped to the part; tokens of the template, question, options and special
    tokens are left out.
    """
    order = list(range(len(example.options))) if order is None else order
    text, where = mlm.build_prompt(example, order, tokenizer, fmt)
    out = []
    for t, (a, b) in enumerate(tokenizer(text, return_offsets_mapping=True)["offset_mapping"]):
        if a == b:
            continue
        for p, (lo, hi) in enumerate(where):
            if a < hi and b > lo:
                out.append((t, p, max(a, lo) - lo, min(b, hi) - lo))
                break
    return out


@torch.no_grad()
def ablation_targets(scorer, tokenizer, fmt, example, sources: list[dict], masks,
                     answer: int, device: str = "cuda", budget: int = 32768) -> np.ndarray:
    """[n] float: logit q(answer | ablated row) for every mask (AT2's training targets).

    The encoder counterpart of context_cite.ablation_targets: each mask keeps
    some sources and deletes the rest (context_cite.ablate, the reference
    method); options stay in release order, so `answer` is the same option
    under every mask. Ablated rows are scored with evaluate_mlm.score_encoded,
    so on FA2 varlen they are packed.
    """
    from llm_jev.evaluation import evaluate_mlm  # evaluation imports modeling, not the reverse

    encoded = [mlm.encode(context_cite.ablate(example, sources, list(keep)), tokenizer, fmt)
               for keep in masks]
    scores = evaluate_mlm.score_encoded(scorer, encoded, device, float("inf"), budget)
    return np.array([context_cite.logit_q(s, answer) for s in scores])


@torch.no_grad()
def mask_attention(scorer, ex: dict, rows: list[int] | None = None,
                   device: str = "cuda") -> tuple[torch.Tensor, torch.Tensor]:
    """Option scores s [K] and attention [L, H, Q, N] from query rows to every token.

    One forward pass of the encoded row `ex` (mlm.encode). rows: the query
    positions (default: every option mask, ex["masks"]). Attention is
    softmax(q k^T * scale) of each layer and head for those rows, with the
    layer's sliding window, in fp32.
    """
    rows_t = torch.tensor(ex["masks"] if rows is None else rows, device=device)
    store: dict = {}
    with capturing(scorer.model, store), torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
        s = scorer(torch.tensor([ex["input_ids"]], device=device),
                   torch.tensor(ex["masks"], device=device), at2_rows=rows_t)
    attn = torch.stack([store[i] for i in sorted(store)])            # [L, H, Q, N]
    return s, attn


def example_data(scorer, tokenizer, fmt, example, num_masks: int = 32, seed: int = 0,
                 device: str = "cuda", budget: int = 32768) -> tuple[torch.Tensor, np.ndarray, int]:
    """One AT2 training example: features [num_masks, L*H], targets [num_masks], the answer.

    The attributed answer is the model's prediction (options in release
    order). Features come from one capture pass on the full row: the answer
    mask's attention, summed per source (source_features). Targets are
    logit q(answer) under num_masks ablations (ablation_targets, masks as
    context_cite.ablation_masks). As the reference stores them, mask m's
    features are (2 * mask_m - 1) @ phi, so the surrogate's prediction for
    ablation m is features[m] @ theta and the source count drops out.
    """
    ex = mlm.encode(example, tokenizer, fmt)
    sources = context_cite.split_sources(example)
    ids, ranges = source_token_ranges(example, tokenizer, fmt, sources)
    if ids != ex["input_ids"]:
        raise ValueError(f"{example.id}: source ranges and the scored prompt differ")
    s, attn = mask_attention(scorer, ex, device=device)                  # [K], [L, H, K, N]
    answer = int(s.argmax())
    phi = source_features(attn[:, :, answer:answer + 1], ranges)        # [S, L*H]
    masks = context_cite.ablation_masks(num_masks, len(sources), 0.5, seed)
    y = ablation_targets(scorer, tokenizer, fmt, example, sources, masks, answer, device, budget)
    signed = torch.tensor(masks, device=phi.device, dtype=phi.dtype) * 2 - 1   # [M, S]
    return signed @ phi, y, answer


class ScoreEstimator(torch.nn.Module):
    """The reference's LinearScoreEstimator: a source's score is theta . phi_j,
    one weight per (layer, head), no bias, initialised to 1/F (the
    average-attention baseline)."""

    def __init__(self, num_features: int):
        super().__init__()
        self.linear = torch.nn.Linear(num_features, 1, bias=False)
        torch.nn.init.constant_(self.linear.weight, 1 / num_features)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """[..., F] features -> [...] scores."""
        return self.linear(features)[..., 0]

    def theta(self) -> torch.Tensor:
        """[F] theta, L1-normalised (the scale is irrelevant under the Pearson loss)."""
        w = self.linear.weight.detach()[0]
        return w / w.abs().sum()


@torch.no_grad()
def attribute(scorer, tokenizer, fmt, example, theta: torch.Tensor,
              device: str = "cuda") -> dict:
    """AT2 on one row: sentence weights and per-token scores from one forward pass.

    Returns context_cite.attribute's format ({"answer", "q", "sources" each with
    its weight "w"}) plus "tokens" (each document token's part, character range
    in it and score "w"), "token_scores" [N] (theta . attention from the answer's
    mask to every prompt token; a sentence's weight is the sum over its tokens),
    "token_ranges" (each source's [start, end) tokens) and "input_ids". theta
    uniform (1/F) gives the average-attention baseline.
    """
    ex = mlm.encode(example, tokenizer, fmt)
    sources = context_cite.split_sources(example)
    s, attn = mask_attention(scorer, ex, device=device)                  # [K], [L, H, K, N]
    answer = int(s.argmax())
    a = attn[:, :, answer].flatten(0, 1)                                # [L*H, N]
    token_scores = theta.to(a) @ a                                       # [N]
    _, ranges = source_token_ranges(example, tokenizer, fmt, sources)
    cum = torch.cat([token_scores.new_zeros(1), token_scores.cumsum(0)])
    w = [float(cum[b] - cum[a_]) for a_, b in ranges]
    return {"answer": answer, "q": torch.softmax(s.float(), 0)[answer].item(),
            "sources": [{**src, "w": v} for src, v in zip(sources, w)],
            "token_scores": token_scores.tolist(), "token_ranges": ranges,
            "input_ids": ex["input_ids"],
            "tokens": [{"index": t, "part": p, "start": a_, "end": b_, "w": float(token_scores[t])}
                       for t, p, a_, b_ in document_tokens(example, tokenizer, fmt)]}
