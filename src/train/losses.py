"""Training losses: the tree-factorized listwise loss and the KL anchors.

Li & Wagle, "LLM-as-Jev" (arXiv 2610.02076v3, §3.4, App. I):
loss = L_tree + lambda * (L_mass + L_out + L_pos). See docs/architecture.md.
"""

import math

import torch

from modeling import modeling

def tree_loss(logits: torch.Tensor, path: list) -> torch.Tensor:
    """L_tree = -log q(gold | x), Eq. 2.

    Each gold-path node contributes -log(p(gold token) / legal mass), which
    is logsumexp over the legal tokens' logits minus the gold token's logit
    (the full-vocabulary normalizer cancels). logits: [len(path), vocab],
    row t at path[t]'s position.
    """
    loss = logits.new_zeros((), dtype=torch.float32)
    for t, (_, legal, gold) in enumerate(path):
        z = logits[t].float()
        loss = loss + torch.logsumexp(z[legal], 0) - z[gold]
    return loss


# Anchored distributions keep the reference's top TOP tokens plus one bucket
# for the rest (App. I).
TOP = 64


def _log1mexp(x: torch.Tensor) -> torch.Tensor:
    """log(1 - exp(x)) for log-probabilities x <= 0 (-50 instead of -inf).

    Clamped inside the log: clamping log(0) = -inf afterwards gives a NaN
    gradient (0 * inf) when the top tokens hold all the mass.
    """
    return torch.log((-torch.expm1(x)).clamp_min(math.exp(-50.0)))


def _split(logits: torch.Tensor, legal: list[int]):
    """Log-probabilities, log legal mass and log illegal mass at a node.

    Both masses are summed directly: 1 - m rounds to 0 in fp32 when the
    legal mass is within ~1e-8 of 1, as at most trie nodes.
    """
    logp = torch.log_softmax(logits.float(), -1)
    is_legal = torch.zeros_like(logp, dtype=torch.bool)
    is_legal[legal] = True
    illegal = logp.masked_fill(is_legal, -torch.inf)
    return logp, torch.logsumexp(logp[legal], 0), torch.logsumexp(illegal, 0), illegal


def _kl_top(ref_lp: torch.Tensor, lp: torch.Tensor) -> torch.Tensor:
    """KL(ref || model) over the reference's top tokens plus a remainder bucket.

    ref_lp, lp: the reference's and the model's log-probabilities of the
    same top tokens, each within its own (normalized) distribution.
    """
    ref = torch.cat([ref_lp, _log1mexp(torch.logsumexp(ref_lp, 0))[None]])
    mod = torch.cat([lp, _log1mexp(torch.logsumexp(lp, 0))[None]])
    return (ref.exp() * (ref - mod)).sum()


def node_stats(logits: torch.Tensor, legal: list[int], top: int = TOP):
    """At a trie node: log legal and illegal mass, and the top illegal tokens
    with their log-probabilities within the illegal mass (p-bar of L_out)."""
    _, log_m, log_r, illegal = _split(logits, legal)
    lp, idx = (illegal - log_r).topk(top)
    return log_m, log_r, idx, lp


def pos_stats(logits: torch.Tensor, top: int = TOP):
    """At a context position: the top tokens and their log-probabilities."""
    lp, idx = torch.log_softmax(logits.float(), -1).topk(top)
    return idx, lp


def node_anchor(logits: torch.Tensor, legal: list[int], ref) -> tuple:
    """(L_mass, L_out) at one trie node, against ref = node_stats(p0's logits).

    L_mass = binary KL(m0 || m), the legal vs illegal mass; L_out = KL over
    the illegal tokens, each side renormalized within its illegal mass.
    """
    ref_log_m, ref_log_r, idx, ref_lp = ref
    logp, log_m, log_r, _ = _split(logits, legal)
    mass = (ref_log_m.exp() * (ref_log_m - log_m)
            + ref_log_r.exp() * (ref_log_r - log_r))
    out = _kl_top(ref_lp, logp[idx] - log_r)
    return mass, out


def pos_anchor(logits: torch.Tensor, ref) -> torch.Tensor:
    """L_pos at one context position, against ref = pos_stats(p0's logits)."""
    idx, ref_lp = ref
    return _kl_top(ref_lp, torch.log_softmax(logits.float(), -1)[idx])


def _logits(model, ex: dict, seqs: list[list[int]], device: str, pad: int):
    """Gold-path logits, reply-start logits, and each sequence's last logits.

    Row 0 is the gold sequence, whose last position is after "]"; the other
    rows are the off-path sequences, whose last position is their node.
    """
    hs, last = modeling.batched_hidden(model, seqs, device, pad)
    head = model.get_base_model().lm_head
    pos = [p for p, _, _ in ex["path"]] + [ex["reply"]]
    at = head(hs[0][pos].to(head.weight.dtype)).float()
    return at[:-1], at[-1], last


def compute(model, ex: dict, device: str, pad: int) -> dict[str, torch.Tensor]:
    """L_tree and the anchors L_mass, L_out, L_pos (Eqs. 2-3) for one example.

    One batch: the gold sequence and, per off-path node, the prompt + the
    node's tokens. p_theta is the LoRA model; p0 is the same model with the
    adapter disabled, run without gradients. Weights: the gold-path nodes
    share 0.5 and the off-path nodes 0.5; the reply start and the position
    after "]" get 0.5 each. Total loss: tree + lambda * (mass + out + pos).
    """
    prompt = ex["input_ids"][: ex["path"][0][0] + 1]
    seqs = [ex["input_ids"]] + [prompt + n for n, _ in ex["off_path"]]
    with torch.no_grad(), model.disable_adapter():
        ref_path, ref_reply, ref_last = _logits(model, ex, seqs, device, pad)
    path, reply, last = _logits(model, ex, seqs, device, pad)
    nodes = [
        [(path[t], ref_path[t], legal) for t, (_, legal, _) in enumerate(ex["path"])],
        [(last[i], ref_last[i], legal)
         for i, (_, legal) in enumerate(ex["off_path"], 1)],
    ]
    mass = out = path.new_zeros(())
    for group in nodes:
        for lg, ref, legal in group:
            m, o = node_anchor(lg, legal, node_stats(ref, legal))
            mass = mass + 0.5 / len(group) * m
            out = out + 0.5 / len(group) * o
    pos = 0.5 * pos_anchor(reply, pos_stats(ref_reply)) + 0.5 * pos_anchor(
        last[0], pos_stats(ref_last[0])
    )
    return {"tree": tree_loss(path, ex["path"]), "mass": mass, "out": out,
            "pos": pos}
