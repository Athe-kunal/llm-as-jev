"""AttnLRP for the decision interface: every context sentence's relevance from one backward pass.

Achtibat et al., AttnLRP (ICML 2024), in the Gradient*Input form of
github.com/rachtibat/LRP-eXplains-Transformers (lxt.efficient, Clear BSD).
The forward is unchanged; the backward follows LRP rules: RMSNorm passes
relevance through its scale (identity rule), the SwiGLU activation uses the
identity rule and its gate * up product the uniform rule, and attention's two
matmuls the uniform rule, which leaves q and k with 1/4 and v with 1/2 of
their gradient. Any attention kernel works, so attention runs on FA2.
A token's relevance is inputs_embeds * grad summed over the hidden size; a
source's is the sum over its tokens (masked_cite.token_owners).
"""

import contextlib
import types
from typing import Any

import numpy as np
import torch

from llm_jev.dataloader import answers
from llm_jev.dataloader import schema
from llm_jev.modeling import context_cite
from llm_jev.modeling import fa2
from llm_jev.modeling import masked_cite

ATTENTION = "lrp"


class _Identity(torch.autograd.Function):
    """y = fn(x) with the identity rule: the gradient is grad * y / x."""

    @staticmethod
    def forward(ctx, fn, x):
        y = fn(x)
        ctx.save_for_backward(torch.where(x == 0, 0, y.float() / x.float()).to(x.dtype))
        return y

    @staticmethod
    def backward(ctx, grad):
        return None, grad * ctx.saved_tensors[0]


class _Scale(torch.autograd.Function):
    """The identity forward, the gradient times c (the uniform rule's split)."""

    @staticmethod
    def forward(ctx, x, c):
        ctx.c = c
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.c, None


def _rms_norm(self, x):
    """Qwen3RMSNorm.forward with the scale held constant (identity rule)."""
    h = x.float()
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.variance_epsilon).detach()
    return self.weight * h.to(x.dtype)


def _mlp(self, x):
    """Qwen3MLP.forward: identity rule on the activation, uniform rule on gate * up."""
    gate = _Identity.apply(self.act_fn, self.gate_proj(x))
    return self.down_proj(_Scale.apply(gate * self.up_proj(x), 0.5))


def attention(module, query, key, value, attention_mask, **kwargs):
    """The uniform rule on QK^T and AV around FA2 (fa2_* inputs) or SDPA (a dense mask)."""
    query, key, value = _Scale.apply(query, 0.25), _Scale.apply(key, 0.25), _Scale.apply(value, 0.5)
    if kwargs.get("fa2_gather") is not None:
        return fa2.attention(module, query, key, value, attention_mask, **kwargs)
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    return sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)


@contextlib.contextmanager
def rules(model):
    """The model with AttnLRP's backward and frozen weights; restored on exit. Qwen3 only."""
    from transformers import AttentionInterface
    from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

    patch = {Qwen3RMSNorm: _rms_norm, Qwen3MLP: _mlp}
    patched = [m for m in model.modules() if type(m) in patch]
    if not any(isinstance(m, Qwen3MLP) for m in patched):
        raise ValueError("AttnLRP rules are implemented for Qwen3 only")
    AttentionInterface.register(ATTENTION, attention)
    impl = model.config._attn_implementation
    grads = [p.requires_grad for p in model.parameters()]
    for m in patched:
        m.forward = types.MethodType(patch[type(m)], m)
    model.requires_grad_(False)
    model.set_attn_implementation(ATTENTION)
    try:
        yield model
    finally:
        model.set_attn_implementation(impl)
        for m in patched:
            del m.forward                                               # back to the class's forward
        for p, g in zip(model.parameters(), grads):
            p.requires_grad_(g)


def goal(z: dict[tuple[int, ...], torch.Tensor], suffix: list[int],
         nodes: dict[tuple[int, ...], list[int]], target: str = "contrast") -> torch.Tensor:
    """The explained scalar: answer suffix logits z[node][token] along its trie path.

    "logit" sums them; "contrast" subtracts, at each node with other legal
    tokens, their mean logit (still linear in the logits, so LRP applies).
    """
    total = torch.zeros((), device=z[()].device)
    for t, x in enumerate(suffix):
        row = z[tuple(suffix[:t])].float()
        total = total + row[x]
        others = [u for u in nodes[tuple(suffix[:t])] if u != x]
        if target == "contrast" and others:
            total = total - row[others].mean()
    return total


def attribute(model, tok, example: schema.Example, target: str = "contrast", kernel: str = "fa2",
              device: str = "cuda") -> dict[str, Any]:
    """AttnLRP on one example, in context_cite.attribute's format.

    One forward over the prompt and its trie nodes (masked_cite.full_batch
    with every source kept) gives the decision; the backward from goal(answer)
    gives each source's relevance "w". bias is 0; "relevance" is the total over
    every input token (conservation: close to "goal"), "context" its sum over
    the sources. kernel "fa2" needs modeling.fa2's kernel; "sdpa" uses a dense
    mask (any device).
    """
    if target not in ("logit", "contrast"):
        raise ValueError(f"target must be 'logit' or 'contrast', not {target!r}")
    sources = context_cite.split_sources(example)
    ids, owner, _ = masked_cite.token_owners(tok, example, sources)
    k = len(example.options)
    suffixes, nodes = answers.candidates(tok, k)
    # one copy that keeps everything (a dummy column when there are no sources)
    batch = masked_cite.full_batch(ids, owner, np.ones((1, max(len(sources), 1)), bool), list(nodes), device)
    attn = batch["fa2"] if kernel == "fa2" else \
        {"attention_mask": fa2.dense_mask(**batch["fa2"])[None, None].to(device)}
    with rules(model), torch.enable_grad():
        embeds = model.get_input_embeddings()(batch["input_ids"][None].to(device)).detach().requires_grad_()
        logits = model(inputs_embeds=embeds, position_ids=batch["position_ids"][None].to(device),
                       logits_to_keep=batch["at"][0].to(device), use_cache=False, **attn).logits[0]  # [1+T, V]
        z = dict(zip(batch["nodes"], logits))
        s = answers.read_trie({n: torch.log_softmax(v.detach().float(), -1) for n, v in z.items()},
                              suffixes, nodes)[0]
        answer = int(s.argmax())
        value = goal(z, suffixes[answer], nodes, target)
        value.backward()
    r = (embeds.detach() * embeds.grad).float().sum(-1)[0].cpu().numpy()  # [N]: the prompt's tokens first
    on = owner >= 0
    w = np.zeros(len(sources))
    np.add.at(w, owner[on], r[: len(ids)][on])
    return {"answer": answer, "q": torch.softmax(s, 0)[answer].item(),
            "sources": [{**src, "w": float(v)} for src, v in zip(sources, w)], "bias": 0.0,
            "goal": value.item(), "relevance": float(r.sum()), "context": float(w.sum())}
