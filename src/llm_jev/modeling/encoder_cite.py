"""ContextCite for encoders (dataloader.mlm), by deletion: the reference method.

The encoder counterpart of context_cite.attribute: delete random halves of the
sentences, score each ablated row (packed on FA2 varlen), fit the sparse linear
surrogate. The ablation scoring is shared with AT2's training targets
(encoder_at2.ablation_targets).
"""

import torch

from llm_jev.dataloader import mlm
from llm_jev.modeling import context_cite
from llm_jev.modeling import encoder_at2
from llm_jev.modeling import masked_cite


@torch.no_grad()
def attribute(scorer, tokenizer, fmt, example, num_ablations: int | None = None,
              keep_prob: float = 0.5, alpha: float = 0.01, seed: int = 0,
              device: str = "cuda", budget: int = 32768) -> dict:
    """ContextCite on one row, in context_cite.attribute's format, plus "ablations".

    num_ablations None uses masked_cite.ablation_count (about two per sentence,
    a power of two in [32, 256]), as the decoder's ContextCite does.
    """
    from llm_jev.evaluation import evaluate_mlm  # evaluation imports modeling, not the reverse

    s = evaluate_mlm.score_encoded(scorer, [mlm.encode(example, tokenizer, fmt)], device,
                                   float("inf"), budget)[0]
    answer = int(s.argmax())
    sources = context_cite.split_sources(example)
    if not sources:
        return {**context_cite.no_sources_result(s, answer), "ablations": 0}
    n = num_ablations or masked_cite.ablation_count(len(sources))
    masks = context_cite.ablation_masks(n, len(sources), keep_prob, seed)
    y = encoder_at2.ablation_targets(scorer, tokenizer, fmt, example, sources, masks, answer,
                                     device, budget)
    return {**context_cite.result(s, answer, sources, masks, y, alpha), "ablations": n}
