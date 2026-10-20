"""ContextCite targets from one pass: the document once, one reader branch per ablation.

context_cite.ablation_targets deletes sentences and re-scores every ablated
prompt in full. Here the prompt is split into the document (the row's parts)
and the reader (question, options, "Best answer: [" and the trie suffixes).
The document is run once with causal attention; each ablation is a copy of the
reader that attends only to the kept sentences, all copies packed into one
sequence under a FlexAttention block mask. Branch 0 keeps every sentence: it
is the decision itself.

This is reader-side masking, not deletion: document tokens still read the
"removed" sentences in every layer, so their information can leak into the
kept ones. Positions stay fixed (as the AT2 paper's masking), unlike deletion.
"""

from typing import Any

import numpy as np
import torch

from dataloader import answers
from modeling import context_cite


def token_owners(tok, row: dict[str, Any], sources: list[dict[str, Any]]) -> tuple[list[int], np.ndarray, int]:
    """The prompt's token ids, each token's source and where the reader starts.

    The prompt is answers.build_prompt with options in release order (as
    answers.encode without shuffling). owner[t] is the first source that
    token t's characters reach, its separator included (ablate drops a
    source with its separator), so " CAL" before "CALIFORNIA" and ".\n" after
    a sentence go to that sentence; -1 outside every source (chat template,
    "State:", the text around a part's sentences, the reader). The reader is
    every token from the first one starting after the last part.
    """
    text, where = answers.build_prompt(row, list(row["labels"]), tok)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    names = [p["name"] for p in row["parts"]]
    by_char = np.full(len(text) + 1, -1)                                # [chars + 1]
    for i, s in enumerate(sources):
        offset = where[names[s["part"]]][0]
        by_char[offset + s["start"] - len(s["sep"]): offset + s["end"]] = i
    owner = np.array([next((o for o in by_char[a:b] if o >= 0), -1)    # [P]
                      for a, b in enc["offset_mapping"]])
    starts = np.array([a for a, _ in enc["offset_mapping"]])            # [P]
    reader = int(np.searchsorted(starts, where[names[-1]][1]))
    return enc["input_ids"], owner, reader


def pack(ids: list[int], reader: int, n: int,
         nodes: list[tuple[int, ...]]) -> dict[str, torch.Tensor | list]:
    """One sequence: the document (tokens before `reader`) once, then n copies of reader + trie.

    A copy is the reader followed by one token per inner trie node (every
    node of answers.trie but the root, by depth): the node's last token, at
    the position it has after the prompt (P - 1 + depth). Every copy gets
    the full prompt's positions, as if it alone followed the document
    (tree-attention decoding). branch is -1 on the document and i on copy i;
    node is j on the token of trie node nodes[j] and -1 elsewhere. at[i, j]
    is the token of copy i whose logits are node nodes[j]'s next-token
    distribution: the reader's last token for the root, the node's own token
    otherwise. With P prompt tokens, R = P - reader and T inner nodes:
    D + n*(R+T) tokens instead of n*P (plus the trie steps).
    """
    nodes = [()] + sorted((x for x in nodes if x), key=len)             # root first, by depth
    P = len(ids)
    doc = torch.tensor(ids[:reader])                                    # [D]
    copy = torch.tensor(ids[reader:] + [x[-1] for x in nodes[1:]])      # [C = R + T]
    pos = torch.tensor(list(range(reader, P)) + [P - 1 + len(x) for x in nodes[1:]])  # [C]
    D, R, C = reader, P - reader, len(copy)
    start = D + torch.arange(n)[:, None] * C                            # [n, 1]
    return {
        "input_ids": torch.cat([doc, copy.repeat(n)]),                  # [D + n*C]
        "position_ids": torch.cat([torch.arange(D), pos.repeat(n)]),    # [D + n*C]
        "branch": torch.cat([torch.full((D,), -1),
                             torch.arange(n).repeat_interleave(C)]),    # [D + n*C]
        "node": torch.cat([torch.full((D,), -1),                         # [D + n*C]
                           torch.cat([torch.full((R,), -1), torch.arange(1, C - R + 1)]).repeat(n)]),
        "at": start + torch.cat([torch.tensor([R - 1]), torch.arange(R, C)]),  # [n, 1 + T]
        "nodes": nodes,
    }


def block_mask(packed: dict[str, torch.Tensor], owner: np.ndarray, keep: np.ndarray, device: str = "cuda"):
    """The FlexAttention mask of a packed sequence: copy i reads only what mask i keeps.

    Query q may attend to key k iff k <= q and either both are in the same
    branch (the document causally, or copy i within itself), or q is in a
    copy i and k is a document token outside every source or in a source
    kept by keep[i]. A copy never sees another copy. Within a copy a trie
    token is seen only by itself and its descendants (tree attention), so
    each node reads the prompt and its own prefix, not its siblings. keep
    [n, S] bool, one row per copy (row 0 all True gives the full prompt,
    i.e. the decision).
    """
    from torch.nn.attention.flex_attention import create_block_mask

    nodes = packed["nodes"]
    anc = torch.tensor([[a[: len(b)] == b for b in nodes] for a in nodes], device=device)  # [1+T, 1+T]
    node = packed["node"].to(device)                                    # [N], -1 off the trie
    branch = packed["branch"].to(device)                                # [N]
    keep = torch.as_tensor(keep, device=device)                         # [n, S]
    S = keep.shape[1]
    kept = torch.cat([keep, keep.new_ones(len(keep), 1)], 1)            # [n, S+1]: column S = no source
    col = torch.full_like(branch, S)                                    # [N]: each key's column in kept
    D = int((branch < 0).sum())
    doc = torch.as_tensor(owner[:D], device=device)                     # [D]
    col[:D] = torch.where(doc >= 0, doc, S)

    def mask_mod(b, h, q, k):
        # a trie key is seen only from its own node or a node below it
        tree = (node[k] < 0) | ((node[q] >= 0) & anc[node[q].clamp(min=0), node[k].clamp(min=0)])
        same = (branch[q] == branch[k]) & tree
        reads = (branch[q] >= 0) & (branch[k] < 0) & kept[branch[q].clamp(min=0), col[k]]
        return (k <= q) & (same | reads)

    n = len(branch)
    # compiled, the mask is evaluated block by block instead of as one dense [N, N] grid
    return create_block_mask(mask_mod, B=None, H=None, Q_LEN=n, KV_LEN=n, device=device, _compile=True)


@torch.no_grad()
def score_copies(model, packed: dict[str, torch.Tensor | list], mask, tok, k: int,
                 device: str = "cuda") -> list[tuple]:
    """answers.score_batch's (s, root legal mass, decoded) for every copy, from one forward.

    The packed sequence runs once under the block mask (the model must use
    attn_implementation="flex_attention"); the LM head runs only at the
    `at` positions, each copy's trie nodes. Copy i's results are those of
    the full prompt with the sources dropped by mask i hidden from its
    reader (block_mask).
    """
    at = packed["at"]                                                   # [n, 1+T]
    logits = model(input_ids=packed["input_ids"][None].to(device),
                   position_ids=packed["position_ids"][None].to(device),
                   attention_mask=mask, logits_to_keep=at.flatten().to(device),
                   use_cache=False).logits[0].view(*at.shape, -1)       # [n, 1+T, V]
    suffixes, nodes = answers.candidates(tok, k)
    out = []
    for row in logits:                                                  # one copy: [1+T, V]
        lp = torch.log_softmax(row.float(), -1)
        out.append(answers.read_trie(dict(zip(packed["nodes"], lp)), suffixes, nodes))
    return out


def attribute(model, tok, row: dict[str, Any], num_ablations: int = 64, keep_prob: float = 0.5,
              alpha: float = 0.01, seed: int = 0, device: str = "cuda") -> dict[str, Any]:
    """context_cite.attribute with the decision and every ablation from one packed forward.

    Copy 0 keeps every source: its argmax is the attributed answer (the
    model's prediction). Copies 1..n use context_cite.ablation_masks, so the
    masks are the same as context_cite.attribute's for the same seed; each
    target is logit q(answer) with the dropped sources hidden from the
    reader instead of deleted. The model must use flex_attention.
    """
    sources = context_cite.split_sources(row)
    masks = context_cite.ablation_masks(num_ablations, len(sources), keep_prob, seed)
    ids, owner, reader = token_owners(tok, row, sources)
    keep = np.vstack([np.ones(len(sources), bool), masks])              # [1 + n, S]
    k = len(row["labels"])
    packed = pack(ids, reader, len(keep), list(answers.candidates(tok, k)[1]))
    s = [x[0] for x in score_copies(model, packed, block_mask(packed, owner, keep, device), tok, k, device)]
    answer = int(s[0].argmax())
    y = np.array([context_cite.logit_q(x, answer) for x in s[1:]])
    return context_cite.result(s[0], answer, sources, masks, y, alpha)
