"""A supervised evidence head on a frozen encoder decision model: which document tokens support an answer.

The encoder (dataloader.mlm, modeling.encoder) reads the row once; its final
hidden states are bidirectional, so every document token has already seen the
question and the options. The head scores token t for answer k from
[h_t ; h_k ; h_t * h_k], h_k being the hidden state at option k's mask: the
evidence it marks is evidence for that answer. The encoder stays frozen, so
its decisions are unchanged and one forward gives both the decision and the
evidence.

Unlike AT2 / ContextCite (which explain what the model used), the head is
trained on evidence spans (gold, or generated), so it predicts what an
annotator would mark.
"""

import torch
from torch import nn

from llm_jev.dataloader import mlm
from llm_jev.modeling import encoder_at2


class EvidenceHead(nn.Module):
    """MLP over [h_t ; h_answer ; h_t * h_answer] -> one logit per token."""

    def __init__(self, hidden: int, width: int = 512, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(3 * hidden, width), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(width, 1))

    def forward(self, tokens: torch.Tensor, answer: torch.Tensor) -> torch.Tensor:
        """tokens [T, d], answer [d] -> logits [T]."""
        a = answer.expand_as(tokens)
        return self.mlp(torch.cat([tokens, a, tokens * a], -1))[..., 0]


def document_labels(example, tokenizer, fmt, spans) -> tuple[list[int], list[tuple[int, int, int]], torch.Tensor]:
    """Document tokens of the release-order prompt: (indices, (part, start, end), labels).

    labels[i] is 1 when token i overlaps one of `spans` (objects with part,
    start, end: character ranges in a part, like schema.Span).
    """
    doc = encoder_at2.document_tokens(example, tokenizer, fmt)
    labels = torch.tensor([float(any(sp.part == p and st < sp.end and en > sp.start for sp in spans))
                           for _, p, st, en in doc])
    return [t for t, *_ in doc], [(p, st, en) for _, p, st, en in doc], labels


@torch.no_grad()
def encode_row(scorer, tokenizer, fmt, example, device: str = "cuda") -> dict:
    """One frozen forward: option scores s [K], hidden states [N, d] (fp32) and the encoding."""
    ex = mlm.encode(example, tokenizer, fmt)
    ids = torch.tensor([ex["input_ids"]], device=device)
    with torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
        hidden = scorer.model.model(input_ids=ids).last_hidden_state[0]
        s = scorer(ids, torch.tensor(ex["masks"], device=device))
    return {"ex": ex, "hidden": hidden.float(), "s": s}


@torch.no_grad()
def predict(scorer, head: EvidenceHead, tokenizer, fmt, example, answer: int | None = None,
            device: str = "cuda") -> dict:
    """Evidence probabilities for every document token, for `answer` (default: the predicted option).

    Returns {"answer", "q", "tokens": [{"index", "part", "start", "end", "p"}]}.
    """
    row = encode_row(scorer, tokenizer, fmt, example, device)
    answer = int(row["s"].argmax()) if answer is None else answer
    idx, where, _ = document_labels(example, tokenizer, fmt, ())
    h = row["hidden"]
    p = torch.sigmoid(head(h[idx], h[row["ex"]["masks"][answer]])) if idx else torch.zeros(0)
    return {"answer": answer, "q": torch.softmax(row["s"].float(), 0)[answer].item(),
            "tokens": [{"index": t, "part": pa, "start": st, "end": en, "p": float(v)}
                       for t, (pa, st, en), v in zip(idx, where, p.tolist())]}
