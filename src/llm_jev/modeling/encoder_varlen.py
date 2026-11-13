"""FlashAttention-2 varlen for encoders: packed rows, bidirectional, sliding windows.

An encoder row attends to itself in both directions; packing several rows
into one sequence without padding needs only their boundaries (cu_seqlens),
which FA2's varlen kernel takes instead of an N x N mask. The kernel is
modeling.fa2.kernel() (the Hub build the decoder path already uses).

How it plugs into transformers: use(model) registers `attention` under NAME
and switches the model to it. transformers then builds no attention masks
(masking_utils skips implementations it has no mask function for), and every
extra keyword argument given to the model's forward reaches `attention`
through each layer's **kwargs: that is how cu_seqlens / max_seqlen arrive.
"""

import torch

from llm_jev.modeling import fa2

NAME = "varlen_bidirectional"


def window(sliding_window: int | None) -> tuple[int, int]:
    """FA2's (left, right) window for transformers' `sliding_window`.

    transformers stores the window as w + 1 (UltraBERT: 129) and its own FA2
    path uses (w, w): each token sees w keys on each side and itself. None
    (a global layer) is unlimited, (-1, -1).
    """
    return (-1, -1) if sliding_window is None else (sliding_window - 1, sliding_window - 1)


@torch.compiler.disable  # one kernel call; keeps compiled graphs around it intact
def attention(module, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
              attention_mask: torch.Tensor | None, scaling: float | None = None,
              sliding_window: int | None = None, dropout: float = 0.0,
              cu_seqlens: torch.Tensor | None = None, max_seqlen: int | None = None,
              **kwargs) -> tuple[torch.Tensor, None]:
    """transformers attention function: query/key/value [1, H, N, D] -> ([1, N, H, D], None).

    cu_seqlens [R + 1] (int32, cumulative row lengths) and max_seqlen mark
    the packed rows; without them the whole sequence is one row. FA2 runs in
    bf16, so fp32 inputs (fp32 weights outside autocast) are cast and the
    output is cast back.
    """
    if attention_mask is not None:
        raise ValueError(f"{NAME} takes row boundaries (cu_seqlens), not an attention mask")
    if query.shape[0] != 1:
        raise ValueError(f"{NAME} expects one packed sequence (batch 1), got {query.shape[0]}")
    n = query.shape[2]
    if cu_seqlens is None:
        cu_seqlens = torch.tensor([0, n], dtype=torch.int32, device=query.device)
        max_seqlen = n
    q, k, v = (x[0].transpose(0, 1).to(torch.bfloat16) for x in (query, key, value))  # [N, H, D]
    out = fa2.kernel().flash_attn_varlen_func(
        q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen,
        dropout_p=dropout, softmax_scale=scaling, causal=False,
        window_size=window(sliding_window))                                    # [N, H, D]
    return out[None].to(query.dtype), None


def groups(lengths: list[int], budget: int = 32768, max_rows: int = 256) -> list[list[int]]:
    """Row indices in packs of at most `budget` tokens in total (and max_rows rows).

    A packed batch costs the sum of its row lengths, not rows x longest as a
    padded one does (answers.batches), so packs can hold more rows. Rows are
    taken shortest first, so a pack holds rows of similar length; a row longer
    than the budget gets a pack of its own.
    """
    out: list[list[int]] = []
    total = 0
    for i in sorted(range(len(lengths)), key=lengths.__getitem__):
        if out and len(out[-1]) < max_rows and total + lengths[i] <= budget:
            out[-1].append(i)
            total += lengths[i]
        else:
            out.append([i])
            total = lengths[i]
    return out


def pack(encoded: list[dict], device: str | torch.device = "cuda") -> dict:
    """Several mlm.encode outputs as one packed sequence for `attention`.

    Returns:
      input_ids [1, T]: the rows concatenated, no padding (T = total tokens).
      position_ids [1, T]: restarting at 0 in every row, so RoPE sees each
        row as if it were alone.
      cu_seqlens [R + 1] int32: row boundaries, [0, n_0, n_0 + n_1, ...].
      max_seqlen: the longest row (sizes the kernel's work grid).
      masks [sum K]: every row's mask positions, shifted by the row's offset.
      splits: each row's K, to cut the flat scores back into rows.
    """
    lengths = [len(e["input_ids"]) for e in encoded]
    starts = [0]
    for n in lengths:
        starts.append(starts[-1] + n)
    return {
        "input_ids": torch.tensor([t for e in encoded for t in e["input_ids"]], device=device)[None],
        "position_ids": torch.cat([torch.arange(n, device=device) for n in lengths])[None],
        "cu_seqlens": torch.tensor(starts, dtype=torch.int32, device=device),
        "max_seqlen": max(lengths),
        "masks": torch.tensor([s + m for e, s in zip(encoded, starts) for m in e["masks"]],
                              device=device),
        "splits": [len(e["masks"]) for e in encoded],
    }


def score_packed(scorer, packed: dict) -> list[torch.Tensor]:
    """s [K] for every row of a pack() batch, in order: one forward pass.

    The scorer's model must be on this attention (use()). position_ids,
    cu_seqlens and max_seqlen travel through the encoder's **kwargs into
    `attention` in every layer.
    """
    s = scorer(packed["input_ids"], packed["masks"], position_ids=packed["position_ids"],
               cu_seqlens=packed["cu_seqlens"], max_seqlen=packed["max_seqlen"])   # [sum K]
    return list(s.split(packed["splits"]))


def use(model) -> None:
    """Switches the model's attention to NAME (registers it once)."""
    from transformers import AttentionInterface

    AttentionInterface.register(NAME, attention)
    model.set_attn_implementation(NAME)
