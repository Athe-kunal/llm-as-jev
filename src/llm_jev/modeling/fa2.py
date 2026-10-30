"""FlashAttention-2 (variable-length) for packed ablations, next to flex attention.

flash-attn has no wheel for torch 2.9, so the kernel is the Hugging Face Hub build
kernels-community/flash-attn. use() registers one attention implementation that
dispatches on a FlexAttention BlockMask (masked_cite's reader-only packing) or on
fa2_* keyword arguments (masked_cite.full_batch: one varlen call, exact, no mask).
"""

import functools
import importlib.util
import sys

import torch

ATTENTION = "packed"
_REPO = "kernels-community/flash-attn"


@functools.cache
def kernel():
    """The FA2 module for this torch / CUDA build, from the Hub snapshot."""
    from huggingface_hub import snapshot_download

    major, minor = torch.__version__.split(".")[:2]
    cuda = torch.version.cuda.replace(".", "")
    build = f"build/torch{major}{minor}-cxx11-cu{cuda}-x86_64-linux"
    path = f"{snapshot_download(_REPO, revision='main', allow_patterns=[build + '/*'])}/{build}"
    spec = importlib.util.spec_from_file_location("flash_attn_hub", f"{path}/__init__.py",
                                                  submodule_search_locations=[path])
    module = importlib.util.module_from_spec(spec)
    sys.modules["flash_attn_hub"] = module
    spec.loader.exec_module(module)
    return module


@torch.compiler.disable  # one kernel already; keeps modeling.fuse's graphs on either side of it
def attention(module, query, key, value, attention_mask, scaling=None, fa2_gather=None, **kwargs):
    """transformers attention function: query [1, H, N, D], key / value
    [1, Hkv, N, D] -> [1, N, H, D]."""
    if fa2_gather is None:
        from transformers.integrations.flex_attention import flex_attention_forward

        return flex_attention_forward(module, query, key, value, attention_mask, scaling=scaling, **kwargs)
    q = query[0].transpose(0, 1).contiguous()                          # [N, H, D]
    k = key[0].transpose(0, 1)[fa2_gather]                             # [sum of key lengths, Hkv, D]
    v = value[0].transpose(0, 1)[fa2_gather]
    out = kernel().flash_attn_varlen_func(q, k, v, kwargs["fa2_cu_q"], kwargs["fa2_cu_k"], kwargs["fa2_max_q"],
                                          kwargs["fa2_max_k"], softmax_scale=scaling, causal=True)
    return out[None], None


def dense_mask(fa2_cu_q, fa2_cu_k, fa2_gather, **_) -> torch.Tensor:
    """The attention of fa2_* inputs as a [N, N] bool mask (True = attends), for SDPA.

    Segment j's queries see its gathered keys, causally aligned to the end
    (flash-attn's causal=True when there are more keys than queries).
    """
    cu_q, cu_k, gather = fa2_cu_q.tolist(), fa2_cu_k.tolist(), fa2_gather.cpu()
    mask = torch.zeros(cu_q[-1], cu_q[-1], dtype=torch.bool)
    for q0, q1, k0, k1 in zip(cu_q, cu_q[1:], cu_k, cu_k[1:]):
        for i in range(q1 - q0):
            mask[q0 + i, gather[k0: k1 - (q1 - q0) + i + 1]] = True
    return mask


def use(model) -> None:
    """Switches the model to ATTENTION (flex for BlockMasks, FA2 for fa2_* inputs)."""
    from transformers import AttentionInterface

    AttentionInterface.register(ATTENTION, attention)
    model.set_attn_implementation(ATTENTION)
