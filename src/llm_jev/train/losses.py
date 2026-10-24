"""Training losses: the tree-factorized listwise loss and the KL anchors.

Li & Wagle, arXiv 2610.02076v3 (§3.4, App. I):
loss = L_tree + lambda * (L_mass + L_out + L_pos).

Shape legend: V vocabulary, T gold-path nodes, O off-path nodes, N = T + O
trie nodes, S sequences, K anchored top tokens.
"""

import math

import torch
from tensordict import TensorDict

from llm_jev.modeling import modeling

# Anchors keep the reference's top TOP tokens plus one bucket for the rest (App. I).
TOP = 64


def legal_mask(legal: list[list[int]], vocab: int, device) -> torch.Tensor:
    """[N, V] bool: True at each node's legal tokens."""
    mask = torch.zeros(len(legal), vocab, dtype=torch.bool, device=device)  # [N, V]
    for i, tokens in enumerate(legal):
        mask[i, tokens] = True
    return mask


def tree_loss(logits: torch.Tensor, legal: torch.Tensor,
              gold: torch.Tensor) -> torch.Tensor:
    """L_tree = -log q(gold | x), Eq. 2, summed over the gold-path nodes.

    The full-vocabulary normalizer cancels. logits [T, V], legal [T, V] bool,
    gold [T] -> [].
    """
    z = logits.float()                                                  # [T, V]
    lse = torch.logsumexp(z.masked_fill(~legal, -torch.inf), -1)        # [T]
    return (lse - z.gather(-1, gold[:, None])[:, 0]).sum()              # []


def _log1mexp(x: torch.Tensor) -> torch.Tensor:
    """log(1 - exp(x)) for log-probabilities x <= 0 (-50 instead of -inf).

    Clamped inside the log: clamping log(0) = -inf afterwards gives a NaN
    gradient (0 * inf) when the top tokens hold all the mass.
    """
    return torch.log((-torch.expm1(x)).clamp_min(math.exp(-50.0)))


def _split(logits: torch.Tensor, legal: torch.Tensor) -> TensorDict:
    """Log-probabilities, log legal mass and log illegal mass at each node.

    logits [N, V], legal [N, V] bool -> TensorDict [N]. The illegal mass is
    summed directly, not as 1 - m: that rounds to 0 in fp32 when the legal
    mass is within ~1e-8 of 1, as at most trie nodes.
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

    ref_lp, lp [N, K] -> [N].
    """
    rest = _log1mexp(torch.logsumexp(ref_lp, -1))                       # [N]
    ref = torch.cat([ref_lp, rest[:, None]], -1)                        # [N, K+1]
    rest = _log1mexp(torch.logsumexp(lp, -1))                           # [N]
    mod = torch.cat([lp, rest[:, None]], -1)                            # [N, K+1]
    return (ref.exp() * (ref - mod)).sum(-1)                            # [N]


def node_stats(logits: torch.Tensor, legal: torch.Tensor,
               top: int = TOP) -> TensorDict:
    """Log legal and illegal mass at each trie node, and the top illegal tokens (p-bar of L_out).

    logits [N, V], legal [N, V] -> TensorDict [N].
    """
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

    L_mass = binary KL(m0 || m) over legal vs illegal mass; L_out = KL over the
    illegal tokens. logits [N, V], legal [N, V], ref [N] -> TensorDict [N]
    (mass, out).
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


def _logits(model, ex: dict, seqs: list[list[int]], device,
            pad: int) -> TensorDict:
    """Gold-path logits, reply-start logits, and each sequence's last logits.

    last[0] is the gold sequence (its last position is after "]"); the other
    rows are the off-path sequences. The TensorDict has no batch dimension,
    since its fields have different leading sizes.
    """
    pos = [p for p, _, _ in ex["path"]] + [ex["reply"]]                 # T + 1 positions
    at, last = modeling.batched_logits(model, seqs, pos, device, pad)   # [T+1, V], [S, V]
    return TensorDict(path=at[:-1], reply=at[-1], last=last,            # [T, V], [V], [S, V]
                      batch_size=[])


def _loss_terms(
    cur_path: torch.Tensor, cur_last: torch.Tensor, cur_reply: torch.Tensor,
    ref_path: torch.Tensor, ref_last: torch.Tensor, ref_reply: torch.Tensor,
    legal: torch.Tensor, gold: torch.Tensor, weight: torch.Tensor,
) -> torch.Tensor:
    """The four losses from the current (cur_*) and reference (ref_*) logits.

    Shapes as `_logits` returns them: path [T, V], reply [V], last [S, V];
    legal [N, V] bool, gold [T], weight [N]. Returns [4]: tree, mass, out, pos.
    """
    T = cur_path.shape[0]
    nodes = torch.cat([cur_path, cur_last[1:]])                         # [N, V]
    ref_nodes = torch.cat([ref_path, ref_last[1:]])                     # [N, V]
    anchor = node_anchor(nodes, legal, node_stats(ref_nodes, legal))    # TensorDict [N]
    sites = torch.stack([cur_reply, cur_last[0]])                       # [2, V]
    ref_sites = torch.stack([ref_reply, ref_last[0]])                   # [2, V]
    return torch.stack([
        tree_loss(cur_path, legal[:T], gold),                           # []
        (weight * anchor["mass"]).sum(),                                # []
        (weight * anchor["out"]).sum(),                                 # []
        0.5 * pos_anchor(sites, pos_stats(ref_sites)).sum(),            # []
    ])


# Node counts vary per example, so compile with dynamic shapes.
loss_terms = torch.compile(_loss_terms, dynamic=True)


def compute(model, ex: dict, device, pad: int) -> TensorDict:
    """L_tree and the anchors L_mass, L_out, L_pos (Eqs. 2-3) for one example.

    p0 is the same model with the adapter disabled, run without gradients.
    Weights: gold-path nodes and off-path nodes share 0.5 each; the reply start
    and the position after "]" get 0.5 each in L_pos. Returns a TensorDict []
    of four scalars.
    """
    prompt = ex["input_ids"][: ex["path"][0][0] + 1]
    seqs = [ex["input_ids"]] + [prompt + n for n, _ in ex["off_path"]]  # S sequences
    peft = getattr(model, "module", model)                              # unwrap DDP
    # cur runs first: FSDP2 initializes on the first forward, and it must see the
    # adapter enabled (disable_adapter clears requires_grad on the LoRA weights).
    cur = _logits(model, ex, seqs, device, pad)                         # p_theta
    with torch.no_grad(), peft.disable_adapter():
        ref = _logits(peft, ex, seqs, device, pad)                      # p0
    T, O = len(ex["path"]), len(ex["off_path"])
    legal = legal_mask([l for _, l, _ in ex["path"]] + [l for _, l in ex["off_path"]],
                       cur["path"].shape[-1], device)                   # [N, V]
    gold = torch.tensor([g for _, _, g in ex["path"]], device=device)   # [T]
    weight = torch.tensor([0.5 / T] * T + [0.5 / max(O, 1)] * O,
                          device=device)                                # [N]
    tree, mass, out, pos = loss_terms(
        cur["path"], cur["last"], cur["reply"],
        ref["path"], ref["last"], ref["reply"], legal, gold, weight)
    return TensorDict(tree=tree, mass=mass, out=out, pos=pos, batch_size=[])
