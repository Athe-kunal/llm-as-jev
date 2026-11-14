"""Training losses: the tree-factorized listwise loss and the KL anchors.

Li & Wagle, arXiv 2610.02076v3 (§3.4, App. I):
loss = L_tree + lambda * (L_mass + L_out + L_pos).

Shape legend: V vocabulary, T gold-path nodes, O off-path nodes, N = T + O
trie nodes, S sequences, K anchored top tokens.
"""

from collections.abc import Sequence
import contextlib
import math

from tensordict import TensorDict
import torch
from torch import nn

from llm_jev.models import decoder

# Anchors keep the reference's top TOP tokens plus one bucket for the rest
# (App. I).
TOP = 64


def legal_mask(
    legal: Sequence[Sequence[int]], vocab: int, device: torch.device | str
) -> torch.Tensor:
    """[N, V] bool: True at each node's legal tokens."""
    mask = torch.zeros(
        len(legal), vocab, dtype=torch.bool, device=device
    )  # [N, V]
    for i, tokens in enumerate(legal):
        mask[i, tokens] = True
    return mask


def tree_loss(
    logits: torch.Tensor, legal: torch.Tensor, gold: torch.Tensor
) -> torch.Tensor:
    """L_tree = -log q(gold | x), Eq. 2, summed over the gold-path nodes.

    The full-vocabulary normalizer cancels. logits [T, V], legal [T, V] bool,
    gold [T] -> [].
    """
    z = logits.float()  # [T, V]
    lse = torch.logsumexp(z.masked_fill(~legal, -torch.inf), -1)  # [T]
    return (lse - z.gather(-1, gold[:, None])[:, 0]).sum()  # []


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
    logp = torch.log_softmax(logits.float(), -1)  # [N, V]
    illegal = logp.masked_fill(legal, -torch.inf)  # [N, V]
    return TensorDict(
        logp=logp,  # [N, V]
        illegal=illegal,  # [N, V]
        log_m=torch.logsumexp(logp.masked_fill(~legal, -torch.inf), -1),  # [N]
        log_r=torch.logsumexp(illegal, -1),  # [N]
        batch_size=logits.shape[:1],
    )


def _kl_top(ref_lp: torch.Tensor, lp: torch.Tensor) -> torch.Tensor:
    """KL(ref || model) over the reference's top tokens plus a remainder bucket.

    ref_lp, lp [N, K] -> [N].
    """
    rest = _log1mexp(torch.logsumexp(ref_lp, -1))  # [N]
    ref = torch.cat([ref_lp, rest[:, None]], -1)  # [N, K+1]
    rest = _log1mexp(torch.logsumexp(lp, -1))  # [N]
    mod = torch.cat([lp, rest[:, None]], -1)  # [N, K+1]
    return (ref.exp() * (ref - mod)).sum(-1)  # [N]


def node_stats(
    logits: torch.Tensor, legal: torch.Tensor, top: int = TOP
) -> TensorDict:
    """Legal and illegal log-mass at each node, and the top illegal tokens.

    The top illegal tokens are p-bar of L_out.

    logits [N, V], legal [N, V] -> TensorDict [N].
    """
    s = _split(logits, legal)
    lp, idx = (s["illegal"] - s["log_r"][:, None]).topk(top, -1)  # [N, K] x2
    return TensorDict(
        log_m=s["log_m"],
        log_r=s["log_r"],  # [N] x2
        idx=idx,
        lp=lp,
        batch_size=s.batch_size,
    )  # [N, K] x2


def pos_stats(logits: torch.Tensor, top: int = TOP) -> TensorDict:
    """At each context position: the top tokens and their log-probabilities.

    logits [P, V] -> TensorDict [P].
    """
    lp, idx = torch.log_softmax(logits.float(), -1).topk(top, -1)  # [P, K] x2
    return TensorDict(idx=idx, lp=lp, batch_size=logits.shape[:1])


def node_anchor(
    logits: torch.Tensor, legal: torch.Tensor, ref: TensorDict
) -> TensorDict:
    """L_mass and L_out at each trie node, against ref = node_stats(p0).

    L_mass = binary KL(m0 || m) over legal vs illegal mass; L_out = KL over the
    illegal tokens. logits [N, V], legal [N, V], ref [N] -> TensorDict [N]
    (mass, out).
    """
    s = _split(logits, legal)
    mass = (
        ref["log_m"].exp() * (ref["log_m"] - s["log_m"])  # [N]
        + ref["log_r"].exp() * (ref["log_r"] - s["log_r"])
    )
    lp = s["logp"].gather(-1, ref["idx"]) - s["log_r"][:, None]  # [N, K]
    return TensorDict(
        mass=mass,
        out=_kl_top(ref["lp"], lp),  # [N] x2
        batch_size=s.batch_size,
    )


def pos_anchor(logits: torch.Tensor, ref: TensorDict) -> torch.Tensor:
    """L_pos at each context position, against ref = pos_stats(p0).

    logits [P, V], ref [P] -> [P].
    """
    lp = torch.log_softmax(logits.float(), -1).gather(-1, ref["idx"])  # [P, K]
    return _kl_top(ref["lp"], lp)  # [P]


def _logits(
    model: nn.Module,
    ex: decoder.TrieEncoding,
    seqs: Sequence[Sequence[int]],
    device: torch.device | str,
    pad: int,
) -> TensorDict:
    """Gold-path logits, reply-start logits, and each sequence's last logits.

    last[0] is the gold sequence (its last position is after "]"); the other
    rows are the off-path sequences. The TensorDict has no batch dimension,
    since its fields have different leading sizes.
    """
    positions = [p for p, _, _ in ex.path] + [ex.reply]  # T + 1 positions
    at, last = decoder.batched_logits(model, seqs, positions, device, pad)
    return TensorDict(path=at[:-1], reply=at[-1], last=last, batch_size=[])


def _loss_terms(
    cur_path: torch.Tensor,
    cur_last: torch.Tensor,
    cur_reply: torch.Tensor,
    ref_path: torch.Tensor,
    ref_last: torch.Tensor,
    ref_reply: torch.Tensor,
    legal: torch.Tensor,
    gold: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """The four losses from the current (cur_*) and reference (ref_*) logits.

    Shapes as `_logits` returns them: path [T, V], reply [V], last [S, V];
    legal [N, V] bool, gold [T], weight [N]. Returns [4]: tree, mass, out,
    pos.
    """
    t = cur_path.shape[0]
    nodes = torch.cat([cur_path, cur_last[1:]])  # [N, V]
    ref_nodes = torch.cat([ref_path, ref_last[1:]])  # [N, V]
    anchor = node_anchor(nodes, legal, node_stats(ref_nodes, legal))  # [N]
    sites = torch.stack([cur_reply, cur_last[0]])  # [2, V]
    ref_sites = torch.stack([ref_reply, ref_last[0]])  # [2, V]
    return torch.stack(
        [
            tree_loss(cur_path, legal[:t], gold),
            (weight * anchor["mass"]).sum(),
            (weight * anchor["out"]).sum(),
            0.5 * pos_anchor(sites, pos_stats(ref_sites)).sum(),
        ]
    )


# Node counts vary per example, so compile with dynamic shapes.
loss_terms = torch.compile(_loss_terms, dynamic=True)


def compute(
    model: nn.Module,
    ex: decoder.TrieEncoding,
    device: torch.device | str,
    pad: int,
    reference: nn.Module | None = None,
) -> TensorDict:
    """L_tree and the anchors L_mass, L_out, L_pos (Eqs. 2-3) for one row.

    p0 is `reference` (a frozen copy of the starting model, for full
    fine-tuning) or, when None, the same model with its LoRA adapter
    disabled; either way it runs without gradients. Gold-path and off-path
    nodes share 0.5 of the node weight each; the reply start and the
    position after "]" get 0.5 each in L_pos.

    Args:
      model: The model being trained (possibly DDP / FSDP wrapped).
      ex: The encoded row, with sampled off-path nodes.
      device: Where the inputs go.
      pad: The tokenizer's pad id.
      reference: p0 for full fine-tuning; None for LoRA.

    Returns:
      A TensorDict [] of four scalars: tree, mass, out, pos.
    """
    seqs = [ex.input_ids] + [ex.prompt + n for n, _ in ex.off_path]
    peft_model = getattr(model, "module", model)  # Unwraps DDP.
    # p_theta runs first: FSDP2 initializes on the first forward, and it must
    # see the adapter enabled (disable_adapter clears requires_grad on LoRA).
    cur = _logits(model, ex, seqs, device, pad)
    if reference is not None:
        p0_model, p0_context = reference, contextlib.nullcontext()
    else:
        p0_model, p0_context = peft_model, peft_model.disable_adapter()
    with torch.no_grad(), p0_context:
        ref = _logits(p0_model, ex, seqs, device, pad)
    n_path, n_off = len(ex.path), len(ex.off_path)
    legal = legal_mask(
        [x for _, x, _ in ex.path] + [x for _, x in ex.off_path],
        cur["path"].shape[-1],
        device,
    )  # [N, V]
    gold = torch.tensor([g for _, _, g in ex.path], device=device)  # [T]
    weight = torch.tensor(
        [0.5 / n_path] * n_path + [0.5 / max(n_off, 1)] * n_off, device=device
    )  # [N]
    tree, mass, out, pos = loss_terms(
        cur["path"],
        cur["last"],
        cur["reply"],
        ref["path"],
        ref["last"],
        ref["reply"],
        legal,
        gold,
        weight,
    )
    return TensorDict(tree=tree, mass=mass, out=out, pos=pos, batch_size=[])
