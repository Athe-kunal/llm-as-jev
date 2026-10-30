"""ContextCite targets from one pass: the document once, one reader branch per ablation.

context_cite.ablation_targets deletes sentences and re-scores every ablated
prompt. Here the document runs once, and each ablation is a copy of the reader
that attends only to its kept sentences; all copies are packed into one
FlexAttention sequence. Branch 0 keeps every sentence: it is the decision.

This masks the reader only, not the document, so document tokens still see the
removed sentences and their information can leak. Positions stay fixed.
"""

import math
from typing import Any

import numpy as np
import torch

from llm_jev.dataloader import answers
from llm_jev.dataloader import schema
from llm_jev.modeling import context_cite


def token_owners(tok, example: schema.Example,
                 sources: list[dict[str, Any]]) -> tuple[list[int], np.ndarray, int]:
    """The prompt's token ids, each token's source (-1 if none), and where the reader starts.

    owner[t] is the first source whose characters reach token t, separator
    included. Tokens outside every source (chat template, text around
    sentences, the reader) get -1. The reader is every token from the first
    one starting after the last part.
    """
    text, where = answers.build_prompt(example, range(len(example.options)), tok)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    by_char = np.full(len(text) + 1, -1)                                # [chars + 1]
    for i, s in enumerate(sources):
        offset = where[s["part"]][0]
        by_char[offset + s["start"] - len(s["sep"]): offset + s["end"]] = i
    owner = np.array([next((o for o in by_char[a:b] if o >= 0), -1)    # [P]
                      for a, b in enc["offset_mapping"]])
    starts = np.array([a for a, _ in enc["offset_mapping"]])            # [P]
    reader = int(np.searchsorted(starts, where[-1][1]))
    return enc["input_ids"], owner, reader


def pack(ids: list[int], reader: int, n: int,
         nodes: list[tuple[int, ...]]) -> dict[str, torch.Tensor | list]:
    """One sequence: the document (tokens before `reader`) once, then n copies of reader + trie.

    Each copy is the reader followed by one token per inner trie node; all
    copies take the full prompt's positions (tree-attention decoding). branch
    is -1 on the document and i on copy i; node is j on trie node nodes[j]'s
    token, -1 elsewhere. at[i, j] is the copy-i token whose logits give node
    nodes[j]'s next-token distribution. Costs D + n*(R+T) tokens instead of
    n*P, with R = P - reader and T inner trie nodes.
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
    """The FlexAttention mask of a packed sequence: copy i reads only what keep[i] keeps.

    A copy never sees another copy. Within a copy, a trie token is seen only
    by its own node's descendants (tree attention). keep [n, S] bool, one row
    per copy; row 0 all True gives the full prompt, i.e. the decision.
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
    # Compiled: evaluated block by block, not as a dense [N, N] grid.
    return create_block_mask(mask_mod, B=None, H=None, Q_LEN=n, KV_LEN=n, device=device, _compile=True)


def score_copies(model, packed: dict[str, torch.Tensor | list], mask, tok, k: int,
                 device: str = "cuda", return_output: bool = False, **forward):
    """answers.score_batch's (s, root legal mass, decoded) for every copy, from one forward.

    The model must use flex attention. Differentiable; extra keyword arguments
    go to the model (e.g. inputs_embeds). With return_output, also the model's output.
    """
    at = packed["at"]                                                   # [n, 1+T]
    if "inputs_embeds" not in forward:
        forward["input_ids"] = packed["input_ids"][None].to(device)
    output = model(position_ids=packed["position_ids"][None].to(device),
                   attention_mask=mask, logits_to_keep=at.flatten().to(device),
                   use_cache=False, **forward)
    logits = output.logits[0].view(*at.shape, -1)                       # [n, 1+T, V]
    suffixes, nodes = answers.candidates(tok, k)
    out = []
    for row in logits:                                                  # one copy: [1+T, V]
        lp = torch.log_softmax(row.float(), -1)
        out.append(answers.read_trie(dict(zip(packed["nodes"], lp)), suffixes, nodes))
    return (out, output) if return_output else out


def full_batch(ids: list[int], owner: np.ndarray, keep: np.ndarray, nodes: list[tuple[int, ...]],
               device: str = "cuda") -> dict[str, Any]:
    """Full masking as FA2 varlen inputs (modeling.fa2), shared prefixes computed once.

    A dropped source is hidden from every later token, so copy i is just the
    prompt's kept tokens at their original positions. Copies are ordered by
    their keep rows, so each shares its longest common prefix with the previous
    one and only the new tokens are computed (a prefix tree). Each trie node
    token attends to its copy's path and its trie ancestors. at [n, 1+T] as in pack.
    """
    nodes = [()] + sorted((x for x in nodes if x), key=len)
    inner, P = nodes[1:], len(ids)
    anc = [[a for a, y in enumerate(inner) if x[: len(y)] == y] for x in inner]   # ancestors, by depth
    kept = [np.where((owner < 0) | keep[i][np.maximum(owner, 0)])[0] for i in range(len(keep))]
    order = sorted(range(len(keep)), key=lambda i: tuple(keep[i]), reverse=True)  # keep before drop
    path = [None] * len(keep)              # packed index of every kept token of each copy
    tok_pos, segs, prev, n_new = [], [], None, 0
    for i in order:
        lcp = 0
        if prev is not None:
            m = min(len(kept[i]), len(kept[prev]))
            diff = np.nonzero(kept[i][:m] != kept[prev][:m])[0]
            lcp = int(diff[0]) if len(diff) else m
        new = kept[i][lcp:]
        idx = np.arange(n_new, n_new + len(new))
        path[i] = np.concatenate([path[prev][:lcp], idx]) if prev is not None else idx
        tok_pos.append(new)
        segs.append((len(new), len(path[i]), path[i]))                # queries, keys, key indices
        n_new += len(new)
        prev = i
    main, T = n_new, len(inner)
    node_base = {i: main + r * T for r, i in enumerate(order)}        # node tokens per copy, in order
    for i in order:
        for j in range(T):
            keys = np.concatenate([path[i], node_base[i] + np.array(anc[j])])
            segs.append((1, len(keys), keys))
    q_lens = torch.tensor([q for q, _, _ in segs])
    k_lens = torch.tensor([k for _, k, _ in segs])
    node_tok = torch.tensor([x[-1] for x in inner], dtype=torch.long)
    node_pos = torch.tensor([P - 1 + len(x) for x in inner], dtype=torch.long)
    ids_t = torch.tensor(ids)
    pos = torch.cat([torch.from_numpy(x) for x in tok_pos] + [node_pos.repeat(len(keep))])
    last = torch.tensor([int(path[i][-1]) for i in range(len(keep))])
    nodes_at = torch.stack([node_base[i] + torch.arange(T) for i in range(len(keep))]) if T else \
        torch.zeros(len(keep), 0, dtype=torch.long)
    cu = lambda x: torch.cat([torch.zeros(1, dtype=torch.long), x.cumsum(0)]).to(device, torch.int32)
    return {
        "input_ids": torch.cat([ids_t[torch.from_numpy(x)] for x in tok_pos] + [node_tok.repeat(len(keep))]),
        "position_ids": pos,
        "at": torch.cat([last[:, None], nodes_at], 1),                  # [n, 1+T]
        "nodes": nodes,
        "naive_tokens": int(sum(len(x) for x in kept)), "tree_tokens": main,
        "fa2": {"fa2_cu_q": cu(q_lens), "fa2_cu_k": cu(k_lens), "fa2_max_q": int(q_lens.max()),
                "fa2_max_k": int(k_lens.max()),
                "fa2_gather": torch.from_numpy(np.concatenate([g for _, _, g in segs])).to(device)},
    }


def score_full(model, batch: dict[str, Any], tok, k: int, device: str = "cuda") -> list[tuple]:
    """answers.score_batch's (s, root legal mass, decoded) per sequence of a full_batch.

    The model must use modeling.fa2.
    """
    at = batch["at"]
    with torch.no_grad():
        logits = model(input_ids=batch["input_ids"][None].to(device),
                       position_ids=batch["position_ids"][None].to(device),
                       logits_to_keep=at.flatten().to(device), use_cache=False,
                       **batch["fa2"]).logits[0].view(*at.shape, -1)    # [n, 1+T, V]
    suffixes, nodes = answers.candidates(tok, k)
    return [answers.read_trie(dict(zip(batch["nodes"], torch.log_softmax(row.float(), -1))), suffixes, nodes)
            for row in logits]


def score_keep(model, tok, ids: list[int], owner: np.ndarray, keep: np.ndarray, k: int,
               budget: int = 65536, device: str = "cuda") -> list[torch.Tensor]:
    """s [K] for every keep row [n, S] under full masking, in FA2 passes of at most `budget` tokens.

    The model must use modeling.fa2.
    """
    nodes = list(answers.candidates(tok, k)[1])
    kept = ((owner[None] < 0) | keep[:, np.maximum(owner, 0)]).sum(1)  # [n] tokens per copy
    s, c = [], 0
    while c < len(keep):                                                # chunks of at most `budget` tokens
        e = c + 1
        while e < len(keep) and kept[c: e + 1].sum() <= budget:
            e += 1
        s += [x[0] for x in score_full(model, full_batch(ids, owner, keep[c:e], nodes, device), tok, k, device)]
        c = e
    return s


def decide(model, tok, example: schema.Example, device: str = "cuda") -> torch.Tensor:
    """s [K] of the full prompt as one FA2 sequence (score_keep with nothing to drop).

    For a model on modeling.fa2, whose attention ignores padding masks, so
    context_cite.predict's batched scoring does not apply.
    """
    ex = answers.encode(example, tok, None)
    ids = ex["input_ids"][: ex["path"][0][0] + 1]
    return score_keep(model, tok, ids, np.full(len(ids), -1), np.ones((1, 1), bool), len(example.options),
                      device=device)[0]


def predict(model, tok, example: schema.Example,
            device: str = "cuda") -> tuple[int, torch.Tensor]:
    """The decision on the full prompt through the packed scorer: the predicted option and s [K].

    Equal to copy 0 of attribute. The model must use flex attention.
    """
    sources = context_cite.split_sources(example)
    if not sources:  # nothing to mask: the plain decision
        s = decide(model, tok, example, device)
        return int(s.argmax()), s
    ids, owner, reader = token_owners(tok, example, sources)
    k = len(example.options)
    packed = pack(ids, reader, 1, list(answers.candidates(tok, k)[1]))
    keep = np.ones((1, len(sources)), bool)
    with torch.no_grad():
        s = score_copies(model, packed, block_mask(packed, owner, keep, device), tok, k, device)[0][0]
    return int(s.argmax()), s


def ablation_count(num_sources: int, low: int = 32, high: int = 256) -> int:
    """The ablation count for a row: about two per source, rounded up to a power of two, clipped to [low, high].

    Powers of two keep the packed sequences to a few shapes for a compiled model.
    """
    return min(high, max(low, 1 << math.ceil(math.log2(2 * num_sources))))


def attribute(model, tok, example: schema.Example, num_ablations: int | None = None, keep_prob: float = 0.5,
              alpha: float = 0.01, seed: int = 0, removal: str = "reader", budget: int = 65536,
              device: str = "cuda") -> dict[str, Any]:
    """context_cite.attribute with the decision and every ablation from packed forwards.

    Copy 0 keeps every source and gives the attributed answer. Copies 1..n use
    the same masks as context_cite.attribute. removal "reader" hides dropped
    sources from the reader only (one flex pass); "full" hides them from every
    later token (FA2, in passes of at most `budget` tokens). The model must use
    modeling.fa2. num_ablations None uses ablation_count.
    """
    if removal not in ("reader", "full"):
        raise ValueError(f"removal must be 'reader' or 'full', not {removal!r}")
    sources = context_cite.split_sources(example)
    if not sources:  # nothing to cite: the decision alone
        s = decide(model, tok, example, device)
        return context_cite.no_sources_result(s, int(s.argmax()))
    n = num_ablations or ablation_count(len(sources))
    masks = context_cite.ablation_masks(n, len(sources), keep_prob, seed)
    ids, owner, reader = token_owners(tok, example, sources)
    keep = np.vstack([np.ones(len(sources), bool), masks])              # [1 + n, S]
    k = len(example.options)
    nodes = list(answers.candidates(tok, k)[1])
    if removal == "reader":
        packed = pack(ids, reader, len(keep), nodes)
        with torch.no_grad():
            s = [x[0] for x in score_copies(model, packed, block_mask(packed, owner, keep, device), tok, k, device)]
    else:
        s = score_keep(model, tok, ids, owner, keep, k, budget, device)
    answer = int(s[0].argmax())
    y = np.array([context_cite.logit_q(x, answer) for x in s[1:]])
    return context_cite.result(s[0], answer, sources, masks, y, alpha)
