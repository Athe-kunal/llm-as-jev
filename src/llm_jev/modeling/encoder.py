"""A masked-language-model encoder as a decision model (see dataloader.mlm).

Works for any transformers MLM whose forward is
`logits = model.decoder(model.head(model.model(...).last_hidden_state))`:
UltraBERT and ModernBERT (and so Ettin) are. Logits are computed only at the
K mask positions, not over the vocabulary at every token.
"""

from typing import Any

import torch
from torch import nn

from llm_jev.dataloader import mlm


def load(model_id: str, revision: str | None = None,
         attention: str = "sdpa", device: str = "cuda",
         dtype: torch.dtype = torch.bfloat16):
    """The encoder with its MLM head, in eval mode on device.

    Args:
      model_id: Hugging Face id or local path.
      revision: Hub commit, branch or tag (None: latest).
      attention: transformers attn_implementation ("sdpa", "eager", ...).
      dtype: bf16 for inference; fp32 for full fine-tuning (bf16 weights
        round away updates of ~lr = 1e-5), with bf16 autocast for speed.
    """
    from transformers import AutoModelForMaskedLM

    model = AutoModelForMaskedLM.from_pretrained(
        model_id, revision=revision, dtype=dtype,
        attn_implementation=attention)
    for name in ("model", "head", "decoder"):
        if not hasattr(model, name):
            raise ValueError(f"{type(model).__name__} has no .{name}; "
                             "encoder.py expects decoder(head(model(...)))")
    return model.to(device).eval()


def _fp32(module, x: torch.Tensor) -> torch.Tensor:
    """module(x) in fp32: fp32 copies of its parameters (gradients reach the
    originals), with autocast off so its matmuls are not cast back to bf16."""
    params = {n: p.float() for n, p in module.named_parameters()}
    with torch.autocast(x.device.type, enabled=False):
        return torch.func.functional_call(module, params, (x.float(),))


def mask_logits(model, hidden: torch.Tensor) -> torch.Tensor:
    """Vocabulary logits [K, V] in fp32 from the final hidden states [K, d] at the masks.

    The head runs in fp32: MLM logits here are ~200 in size, where bf16 steps
    by 1-2, which ties most of 151 options and can change the argmax.
    """
    return _fp32(model.decoder, _fp32(model.head, hidden))


def option_scores(logits: torch.Tensor, yes: int, no: int | None) -> torch.Tensor:
    """s [K] from mask logits [K, V]: logit(yes), minus logit(no) if set."""
    s = logits[:, yes]
    return s if no is None else s - logits[:, no]


class Scorer(nn.Module):
    """The encoder and its verbalizer: input ids and mask positions -> s [K].

    The module to train and to hand to Accelerate: scoring inside forward()
    keeps DDP's gradient sync and mixed-precision autocast, which calling the
    encoder's submodules from outside would bypass.
    """

    def __init__(self, model, tokenizer: Any, fmt: mlm.Format):
        super().__init__()
        self.model = model
        self.yes, self.no = mlm.verbalizer_ids(tokenizer, fmt)

    def forward(self, input_ids: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """input_ids [1, N], masks [K] positions -> s [K] in fp32."""
        hidden = self.model.model(input_ids=input_ids).last_hidden_state[0]  # [N, d]
        return option_scores(mask_logits(self.model, hidden[masks]), self.yes, self.no)


def score(scorer, ex: dict[str, Any], device: str = "cuda") -> torch.Tensor:
    """s [K] for one encoded example (mlm.encode), in its mask order. One forward."""
    return scorer(torch.tensor([ex["input_ids"]], device=device),
                  torch.tensor(ex["masks"], device=device))


def loss(scorer, ex: dict[str, Any], device: str = "cuda"
         ) -> tuple[torch.Tensor, torch.Tensor]:
    """Cross-entropy of the gold option over the K masks (L_tree with one
    level), and whether the argmax is gold. Returns ([] loss, [] bool)."""
    s = score(scorer, ex, device)                                      # [K]
    gold = torch.tensor(ex["gold"], device=s.device)
    return nn.functional.cross_entropy(s, gold), s.argmax() == gold
