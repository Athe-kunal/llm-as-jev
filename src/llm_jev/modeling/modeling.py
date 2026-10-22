"""The base model: loading, LoRA, and a batched forward pass."""

import torch
from torch import nn


def load_base(model_id: str):
    """The base LLM in bf16 on the CPU (Qwen3.5 uses its multimodal class)."""
    from transformers import AutoModelForCausalLM
    from transformers import Qwen3_5ForConditionalGeneration

    cls = (Qwen3_5ForConditionalGeneration if "Qwen3.5" in model_id
           else AutoModelForCausalLM)
    return cls.from_pretrained(model_id, dtype=torch.bfloat16,
                               attn_implementation="sdpa")


def add_lora(model, rank: int = 16):
    """A fresh LoRA adapter on the decoder's linear layers (not the vision tower or LM head)."""
    from peft import LoraConfig
    from peft import get_peft_model

    targets = [n for n, m in model.named_modules()
               if isinstance(m, nn.Linear) and ".layers." in n
               and "visual" not in n]
    return get_peft_model(model, LoraConfig(
        r=rank, lora_alpha=2 * rank, lora_dropout=0.0, target_modules=targets,
        bias="none", task_type="CAUSAL_LM"))


def load_trained(model_id: str, adapter: str, device: str = "cuda"):
    """The base model with a saved LoRA adapter, in eval mode."""
    from peft import PeftModel

    base = load_base(model_id).to(device)
    return PeftModel.from_pretrained(base, adapter).eval()


def load(model_id: str, adapter: str | None = None, merge: bool = True, device: str = "cuda"):
    """A model for inference, in eval mode on device: the base model, or with
    a saved LoRA adapter, merged into the weights unless merge=False."""
    if adapter is None:
        return load_base(model_id).to(device).eval()
    model = load_trained(model_id, adapter, device)
    return model.merge_and_unload() if merge else model


def batched_hidden(model, seqs: list[list[int]], device: str, pad: int):
    """Last-layer states and last-position logits for several sequences.

    Sequences are left-padded; each row matches a batch-of-one forward up to
    bf16 rounding. Returns per-row hidden states (padding removed) and logits.
    """
    n = max(len(x) for x in seqs)
    ids = torch.full((len(seqs), n), pad, dtype=torch.long)
    mask = torch.zeros((len(seqs), n), dtype=torch.long)
    for i, x in enumerate(seqs):
        ids[i, n - len(x):] = torch.tensor(x)
        mask[i, n - len(x):] = 1
    ids, mask = ids.to(device), mask.to(device)
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    # Language model only, last layer only (no output_hidden_states); LoRA stays active.
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    out = base.model(input_ids=ids, attention_mask=mask, position_ids=pos,
                     use_cache=False)
    last = out.last_hidden_state
    logits = base.lm_head(last[:, -1]).float()
    last = last.float()
    return [last[i, n - len(x):] for i, x in enumerate(seqs)], logits
