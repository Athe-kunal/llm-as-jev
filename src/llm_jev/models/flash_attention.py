"""FlashAttention-2 varlen for encoders: packed rows, bidirectional, windows.

An encoder row attends to itself in both directions; packing several rows
into one sequence without padding needs only their boundaries (cu_seqlens),
which FA2's varlen kernel takes instead of an N x N mask.

flash-attn has no wheel for every torch build, so the kernel is the Hugging
Face Hub build kernels-community/flash-attn.

How it plugs into transformers: `use(model)` registers `attention` under
NAME and switches the model to it. transformers then builds no attention
masks, and every extra keyword argument given to the model's forward reaches
`attention` through each layer's **kwargs: that is how cu_seqlens and
max_seqlen arrive.
"""

import dataclasses
import functools
import importlib.util
import sys
from collections.abc import Sequence
from types import ModuleType
from typing import Any

import torch
from torch import nn

NAME = "varlen_bidirectional"
_KERNEL_REPO = "kernels-community/flash-attn"


@functools.cache
def kernel() -> ModuleType:
    """The FA2 module for this torch / CUDA build, from the Hub snapshot."""
    from huggingface_hub import snapshot_download

    major, minor = torch.__version__.split(".")[:2]
    cuda = torch.version.cuda.replace(".", "")
    build = f"build/torch{major}{minor}-cxx11-cu{cuda}-x86_64-linux"
    root = snapshot_download(
        _KERNEL_REPO, revision="main", allow_patterns=[build + "/*"]
    )
    path = f"{root}/{build}"
    spec = importlib.util.spec_from_file_location(
        "flash_attn_hub", f"{path}/__init__.py", submodule_search_locations=[path]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["flash_attn_hub"] = module
    spec.loader.exec_module(module)
    return module


def window(sliding_window: int | None) -> tuple[int, int]:
    """FA2's (left, right) window for transformers' `sliding_window`.

    transformers stores the window as w + 1 (UltraBERT: 129) and its own FA2
    path uses (w, w): each token sees w keys on each side and itself. None
    (a global layer) is unlimited, (-1, -1).
    """
    if sliding_window is None:
        return (-1, -1)
    return (sliding_window - 1, sliding_window - 1)


@torch.compiler.disable  # One kernel call; keeps graphs around it intact.
def attention(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float | None = None,
    sliding_window: int | None = None,
    dropout: float = 0.0,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: int | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """A transformers attention function over packed rows.

    Args:
      module: The attention layer (unused).
      query: [1, H, N, D].
      key: [1, H, N, D].
      value: [1, H, N, D].
      attention_mask: Must be None: rows are given by cu_seqlens.
      scaling: Softmax scale.
      sliding_window: transformers' window of this layer.
      dropout: Attention dropout.
      cu_seqlens: [R + 1] int32 cumulative row lengths; without it the whole
        sequence is one row.
      max_seqlen: The longest row.
      **kwargs: Ignored.

    Returns:
      ([1, N, H, D], None). FA2 runs in bf16; fp32 inputs are cast and the
      output is cast back.

    Raises:
      ValueError: Given an attention mask or a batch of more than one.
    """
    del module, kwargs
    if attention_mask is not None:
        raise ValueError(f"{NAME} takes row boundaries, not an attention mask")
    if query.shape[0] != 1:
        raise ValueError(f"{NAME} expects one packed sequence, not a batch")
    n = query.shape[2]
    if cu_seqlens is None:
        cu_seqlens = torch.tensor([0, n], dtype=torch.int32, device=query.device)
        max_seqlen = n
    q, k, v = (
        x[0].transpose(0, 1).to(torch.bfloat16) for x in (query, key, value)
    )  # [N, H, D]
    out = kernel().flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens,
        cu_seqlens,
        max_seqlen,
        max_seqlen,
        dropout_p=dropout,
        softmax_scale=scaling,
        causal=False,
        window_size=window(sliding_window),
    )  # [N, H, D]
    return out[None].to(query.dtype), None


def use(model: nn.Module) -> None:
    """Switches the model's attention to NAME (registered once)."""
    from transformers import AttentionInterface

    AttentionInterface.register(NAME, attention)
    model.set_attn_implementation(NAME)


def is_used(model: nn.Module) -> bool:
    """Whether the model runs this attention, so rows can be packed."""
    return model.config._attn_implementation == NAME


def groups(
    lengths: Sequence[int], budget: int = 32768, max_rows: int = 256
) -> list[list[int]]:
    """Row indices in packs of at most `budget` tokens (and `max_rows` rows).

    A packed batch costs the sum of its row lengths, not rows x longest as a
    padded one does. Rows are taken shortest first, so a pack holds rows of
    similar length; a row longer than the budget gets a pack of its own.
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


@dataclasses.dataclass(frozen=True)
class Packed:
    """Several rows as one packed sequence.

    Attributes:
      input_ids: [1, T], the rows concatenated, no padding.
      position_ids: [1, T], restarting at 0 in every row.
      cu_seqlens: [R + 1] int32 row boundaries.
      max_seqlen: The longest row.
      masks: [sum K] every row's mask positions, shifted by its offset.
      splits: Each row's K, to cut the flat scores back into rows.
    """

    input_ids: torch.Tensor
    position_ids: torch.Tensor
    cu_seqlens: torch.Tensor
    max_seqlen: int
    masks: torch.Tensor
    splits: list[int]

    def model_kwargs(self) -> dict[str, Any]:
        """The extra forward arguments `attention` reads."""
        return {
            "position_ids": self.position_ids,
            "cu_seqlens": self.cu_seqlens,
            "max_seqlen": self.max_seqlen,
        }


def pack(
    rows: Sequence[tuple[Sequence[int], Sequence[int]]],
    device: torch.device | str,
) -> Packed:
    """Packs (input ids, mask positions) rows into one sequence."""
    lengths = [len(ids) for ids, _ in rows]
    starts = [0]
    for n in lengths:
        starts.append(starts[-1] + n)
    return Packed(
        input_ids=torch.tensor(
            [t for ids, _ in rows for t in ids], device=device
        )[None],
        position_ids=torch.cat(
            [torch.arange(n, device=device) for n in lengths]
        )[None],
        cu_seqlens=torch.tensor(starts, dtype=torch.int32, device=device),
        max_seqlen=max(lengths),
        masks=torch.tensor(
            [
                s + m
                for (_, masks), s in zip(rows, starts, strict=False)
                for m in masks
            ],
            device=device,
        ),
        splits=[len(masks) for _, masks in rows],
    )
