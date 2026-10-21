"""Training losses: the tree-factorized listwise loss and the KL anchors.

Li & Wagle, "LLM-as-Jev" (arXiv 2610.02076v3, §3.4, App. I):
loss = L_tree + lambda * (L_mass + L_out + L_pos). See docs/architecture.md.

Shapes in the comments:
  V  vocabulary size
  T  gold-path nodes (suffix tokens of the gold option, "k]")
  O  sampled off-path nodes (answers.OFF_PATH, fewer for small tries)
  N  trie nodes scored together: T + O
  S  sequences in the forward batch: 1 gold + O off-path
  K  anchored top tokens (TOP)
Per-node quantities are TensorDicts with batch_size [N], so indexing or
slicing one slices every field together.
"""

import math

import torch
from tensordict import TensorDict

from modeling import modeling

# Anchored distributions keep the reference's top TOP tokens plus one bucket
# for the rest (App. I).
TOP = 64


def legal_mask(legal: list[list[int]], vocab: int, device) -> torch.Tensor:
    """[N, V] bool: True at each node's legal tokens."""
    mask = torch.zeros(len(legal), vocab, dtype=torch.bool, device=device)  # [N, V]
    for i, tokens in enumerate(legal):
        mask[i, tokens] = True
    return mask


def tree_loss(logits: torch.Tensor, legal: torch.Tensor,
              gold: torch.Tensor) -> torch.Tensor:
    """L_tree = -log q(gold | x), Eq. 2.

    Each gold-path node contributes -log(p(gold token) / legal mass), which
    is logsumexp over the legal tokens' logits minus the gold token's logit
    (the full-vocabulary normalizer cancels).
    logits [T, V], legal [T, V] bool, gold [T] -> [].
    """
    z = logits.float()                                                  # [T, V]
    lse = torch.logsumexp(z.masked_fill(~legal, -torch.inf), -1)        # [T]
    return (lse - z.gather(-1, gold[:, None])[:, 0]).sum()              # []


def _log1mexp(x: torch.Tensor) -> torch.Tensor:
    """log(1 - exp(x)) for log-probabilities x <= 0 (-50 instead of -inf).

    Elementwise, any shape. Clamped inside the log: clamping log(0) = -inf
    afterwards gives a NaN gradient (0 * inf) when the top tokens hold all
    the mass.
    """
    return torch.log((-torch.expm1(x)).clamp_min(math.exp(-50.0)))


def _split(logits: torch.Tensor, legal: torch.Tensor) -> TensorDict:
    """Log-probabilities, log legal mass and log illegal mass at each node.

    logits [N, V], legal [N, V] bool -> TensorDict [N]. Both masses are
    summed directly: 1 - m rounds to 0 in fp32 when the legal mass is within
    ~1e-8 of 1, as at most trie nodes.
    """
    logp = torch.log_softmax(logits.float(), -1)                        # [N, V]
    illegal = logp.masked_fill(legal, -torch.inf)                       # [N, V]
    return TensorDict(
        logp=logp,                                                      # [N, V]
        illegal=illegal,                                                # [N, V]
        log_m=torch.logsumexp(logp.masked_fill(~legal, -torch.inf), -1),  # [N]
        log_r=torch.logsumexp(illegal, -1),                             # [N]
        batch_size=logits.shape[:1],
    )


def _kl_top(ref_lp: torch.Tensor, lp: torch.Tensor) -> torch.Tensor:
    """KL(ref || model) over the reference's top tokens plus a remainder bucket.

    ref_lp, lp [N, K]: the reference's and the model's log-probabilities of
    the same top tokens, each within its own (normalized) distribution.
    Returns [N].
    """
    rest = _log1mexp(torch.logsumexp(ref_lp, -1))                       # [N]
    ref = torch.cat([ref_lp, rest[:, None]], -1)                        # [N, K+1]
    rest = _log1mexp(torch.logsumexp(lp, -1))                           # [N]
    mod = torch.cat([lp, rest[:, None]], -1)                            # [N, K+1]
    return (ref.exp() * (ref - mod)).sum(-1)                            # [N]


def node_stats(logits: torch.Tensor, legal: torch.Tensor,
               top: int = TOP) -> TensorDict:
    """At each trie node: log legal and illegal mass, and the top illegal
    tokens with their log-probabilities within the illegal mass (p-bar of
    L_out). logits [N, V], legal [N, V] -> TensorDict [N]."""
    s = _split(logits, legal)
    lp, idx = (s["illegal"] - s["log_r"][:, None]).topk(top, -1)        # [N, K] x2
    return TensorDict(log_m=s["log_m"], log_r=s["log_r"],               # [N] x2
                      idx=idx, lp=lp, batch_size=s.batch_size)          # [N, K] x2


def pos_stats(logits: torch.Tensor, top: int = TOP) -> TensorDict:
    """At each context position: the top tokens and their log-probabilities.
    logits [P, V] -> TensorDict [P]."""
    lp, idx = torch.log_softmax(logits.float(), -1).topk(top, -1)       # [P, K] x2
    return TensorDict(idx=idx, lp=lp, batch_size=logits.shape[:1])


def node_anchor(logits: torch.Tensor, legal: torch.Tensor,
                ref: TensorDict) -> TensorDict:
    """L_mass and L_out at each trie node, against ref = node_stats(p0's logits).

    L_mass = binary KL(m0 || m), the legal vs illegal mass; L_out = KL over
    the illegal tokens, each side renormalized within its illegal mass.
    logits [N, V], legal [N, V], ref [N] -> TensorDict [N] (mass, out).
    """
    s = _split(logits, legal)
    mass = (ref["log_m"].exp() * (ref["log_m"] - s["log_m"])           # [N]
            + ref["log_r"].exp() * (ref["log_r"] - s["log_r"]))
    lp = s["logp"].gather(-1, ref["idx"]) - s["log_r"][:, None]         # [N, K]
    return TensorDict(mass=mass, out=_kl_top(ref["lp"], lp),            # [N] x2
                      batch_size=s.batch_size)


def pos_anchor(logits: torch.Tensor, ref: TensorDict) -> torch.Tensor:
    """L_pos at each context position, against ref = pos_stats(p0's logits).
    logits [P, V], ref [P] -> [P]."""
    lp = torch.log_softmax(logits.float(), -1).gather(-1, ref["idx"])   # [P, K]
    return _kl_top(ref["lp"], lp)                                       # [P]


def _logits(model, ex: dict, seqs: list[list[int]], device: str,
            pad: int) -> TensorDict:
    """Gold-path logits, reply-start logits, and each sequence's last logits.

    Row 0 of last is the gold sequence, whose last position is after "]";
    the other rows are the off-path sequences, whose last position is their
    node. Fields have different leading sizes, so the TensorDict has no
    batch dimension.
    """
    hs, last = modeling.batched_hidden(model, seqs, device, pad)        # S x [L_i, H], [S, V]
    head = model.get_base_model().lm_head
    pos = [p for p, _, _ in ex["path"]] + [ex["reply"]]                 # T + 1 positions
    at = head(hs[0][pos].to(head.weight.dtype)).float()                 # [T+1, V]
    return TensorDict(path=at[:-1], reply=at[-1], last=last,            # [T, V], [V], [S, V]
                      batch_size=[])


def compute(model, ex: dict, device: str, pad: int) -> TensorDict:
    """L_tree and the anchors L_mass, L_out, L_pos (Eqs. 2-3) for one example.

    One batch: the gold sequence and, per off-path node, the prompt + the
    node's tokens. p_theta is the LoRA model; p0 is the same model with the
    adapter disabled, run without gradients. All N trie nodes (T gold-path,
    then O off-path) are scored as one [N, V] batch. Weights: the gold-path
    nodes share 0.5 and the off-path nodes 0.5; the reply start and the
    position after "]" get 0.5 each. Total loss: tree + lambda * (mass + out
    + pos). Returns a TensorDict [] of four scalars.
    """
    prompt = ex["input_ids"][: ex["path"][0][0] + 1]
    seqs = [ex["input_ids"]] + [prompt + n for n, _ in ex["off_path"]]  # S sequences
    with torch.no_grad(), model.disable_adapter():
        ref = _logits(model, ex, seqs, device, pad)                     # p0
    cur = _logits(model, ex, seqs, device, pad)                         # p_theta
    T, O = len(ex["path"]), len(ex["off_path"])
    legal = legal_mask([l for _, l, _ in ex["path"]] + [l for _, l in ex["off_path"]],
                       cur["path"].shape[-1], device)                   # [N, V]
    gold = torch.tensor([g for _, _, g in ex["path"]], device=device)   # [T]
    weight = torch.tensor([0.5 / T] * T + [0.5 / max(O, 1)] * O,
                          device=device)                                # [N]
    nodes = torch.cat([cur["path"], cur["last"][1:]])                   # [N, V]
    ref_nodes = torch.cat([ref["path"], ref["last"][1:]])               # [N, V]
    anchor = node_anchor(nodes, legal, node_stats(ref_nodes, legal))    # TensorDict [N]
    sites = torch.stack([cur["reply"], cur["last"][0]])                 # [2, V]
    ref_sites = torch.stack([ref["reply"], ref["last"][0]])             # [2, V]
    return TensorDict(
        tree=tree_loss(cur["path"], legal[:T], gold),                   # []
        mass=(weight * anchor["mass"]).sum(),                           # []
        out=(weight * anchor["out"]).sum(),                             # []
        pos=0.5 * pos_anchor(sites, pos_stats(ref_sites)).sum(),        # []
        batch_size=[],
    )


def pearson_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """AT2's loss: the negative Pearson correlation between surrogate
    predictions and ablation targets, per row, averaged over rows (the
    reference's PearsonCorrelationLoss). pred, target [B, M] -> [].

    Invariant to a per-row affine map of pred, so the surrogate needs no
    intercept or scale.
    """
    p = pred - pred.mean(1, keepdim=True)                                       # [B, M]
    t = target - target.mean(1, keepdim=True)                                   # [B, M]
    corr = (p * t).mean(1) / ((p ** 2).mean(1).sqrt() * (t ** 2).mean(1).sqrt() + 1e-8)  # [B]
    return -corr.mean()


def diffmask_loss(open_prob: torch.Tensor, s: torch.Tensor, s_masked: torch.Tensor,
                  lam: float, margin: float) -> TensorDict:
    """DiffMask's constrained objective for one row, as a Lagrangian.

    Minimise the expected fraction of open gates subject to
    KL(q || q~) <= margin, where q = softmax(s) is the tree distribution of
    the unmasked decision and q~ = softmax(s_masked) that of the masked one;
    lam >= 0 is the multiplier (updated by dual ascent in the trainer).
    open_prob [T] = P(z_t > 0) per gated token, s, s_masked [K] -> loss and
    its parts, all [].
    """
    l0 = open_prob.mean()                                                       # []
    lq = torch.log_softmax(s.detach().float(), -1)                              # [K] the target, fixed
    kl = (lq.exp() * (lq - torch.log_softmax(s_masked.float(), -1))).sum()      # []
    return TensorDict({"loss": l0 + lam * (kl - margin), "l0": l0, "kl": kl})
