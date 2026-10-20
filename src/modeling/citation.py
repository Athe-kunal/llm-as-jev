"""Citation head: scores context sentences from the decision model's own hidden states.

Option A of docs/architecture.md §9: the decision model is frozen; one
forward pass over the prompt followed by the predicted answer gives
last-layer states; a SpanHead lets the document's tokens attend to the
tokens after it (question, options, prefill, answer); token logits are
averaged per sentence (context_cite.split_sources). Trained like AT2 (§8):
the summed scores of the kept sentences must correlate with the answer's
logit q under ablations (losses.pearson_loss), with AT2's precomputed
targets.
"""

from typing import Any

import torch
from torch import nn

from modeling import at2, context_cite, modeling
from modeling.span_heads import SpanHead


class CitationHead(nn.Module):
    """Sentence scores from hidden states: SpanHead token logits, averaged
    over each sentence's tokens (the mean, so length does not inflate a
    sentence's score)."""

    def __init__(self, hidden: int, dim: int = 512, heads: int = 8):
        super().__init__()
        self.span = SpanHead(hidden, dim, heads)

    def forward(self, states: torch.Tensor, ranges: list[tuple[int, int]],
                query_start: int) -> torch.Tensor:
        """states [N, D] of one sequence, sentence token ranges, and the first
        token after the document -> [S] sentence scores."""
        lo = min(a for a, _ in ranges)
        hi = max(b for _, b in ranges)
        tok = self.span(states[lo:hi], states[query_start:])                  # [hi - lo]
        sums = at2.range_sums(tok, [(a - lo, b - lo) for a, b in ranges])      # [S]
        return sums / torch.tensor([b - a for a, b in ranges], device=sums.device)


@torch.no_grad()
def encode(model, tok, row: dict[str, Any], device: str = "cuda") -> dict[str, Any]:
    """The frozen decision model's view of a row: last-layer states [N, D] of
    the prompt followed by the predicted answer's suffix, the sentences'
    token ranges, and where the query (everything after the document) starts."""
    prompt, answer, s = context_cite.predict(model, tok, row, device)
    suffix, _ = context_cite.answer_positions(tok, row, prompt, answer)
    sources = context_cite.split_sources(row)
    ranges = at2.source_token_ranges(row, tok, sources)
    hidden, _ = modeling.batched_hidden(model, [prompt + suffix], device, tok.pad_token_id)
    return {"states": hidden[0], "ranges": ranges, "query_start": max(b for _, b in ranges),
            "answer": answer, "q": torch.softmax(s.float(), 0)[answer].item(), "sources": sources}


@torch.no_grad()
def attribute(model, tok, row: dict[str, Any], head: CitationHead, device: str = "cuda") -> dict[str, Any]:
    """The head on one row, in context_cite.attribute's format."""
    e = encode(model, tok, row, device)
    w = head(e["states"].float(), e["ranges"], e["query_start"]).tolist()
    return {"answer": e["answer"], "q": e["q"], "sources": [{**s, "w": v} for s, v in zip(e["sources"], w)]}
