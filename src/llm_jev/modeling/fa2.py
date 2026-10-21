"""FlashAttention-2 (variable-length) for packed ablations, next to flex attention.

flash-attn has no wheel for torch 2.9; the kernel is the Hugging Face Hub
build kernels-community/flash-attn (main, build/torch29-cxx11-cu128-...),
imported straight from the snapshot. use() registers one attention
implementation, ATTENTION, that dispatches on its inputs:

- a FlexAttention BlockMask as attention_mask: flex attention
  (masked_cite's reader-only packing);
- fa2_* keyword arguments (masked_cite.full_batch): one FA2 varlen call.
  The packed tokens are split into query segments (fa2_cu_q); segment i
  attends, causally and aligned to its end, to keys gathered by index
  (fa2_gather, fa2_cu_k): a prefix-tree node sees its whole root-to-node
  path, a trie node its copy's path and its trie ancestors. Exact: no
  attention mask is needed anywhere.
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


def use(model) -> None:
    """Switches the model to ATTENTION (flex for BlockMasks, FA2 for fa2_* inputs)."""
    from transformers import AttentionInterface

    AttentionInterface.register(ATTENTION, attention)
    model.set_attn_implementation(ATTENTION)
