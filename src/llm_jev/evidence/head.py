"""An evidence head on a frozen encoder: which tokens support an answer.

The encoder reads the row once; its final hidden states are bidirectional,
so every document token has already seen the question and the options. The
head scores token t for answer k from [h_t ; h_k ; h_t * h_k], h_k being
the hidden state at option k's mask. The encoder stays frozen, so its
decisions are unchanged and one forward gives the decision and the evidence.

Unlike AT2 or ContextCite, which explain what the model used, the head is
trained on evidence spans (gold, or generated), so it predicts what an
annotator would mark.
"""

from collections.abc import Sequence
import dataclasses

import torch
from torch import nn

from llm_jev.data import schema
from llm_jev.models import encoder


class EvidenceHead(nn.Module):
    """MLP over [h_t ; h_answer ; h_t * h_answer] -> one logit per token."""

    def __init__(
        self, hidden: int, width: int = 512, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3 * hidden, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, 1),
        )

    def forward(
        self, tokens: torch.Tensor, answer: torch.Tensor
    ) -> torch.Tensor:
        """Tokens [T, d], answer [d] -> logits [T]."""
        a = answer.expand_as(tokens)
        return self.mlp(torch.cat([tokens, a, tokens * a], -1))[..., 0]


@dataclasses.dataclass(frozen=True)
class DocumentTokens:
    """The document tokens of a release-order prompt.

    Attributes:
      index: Token positions in the prompt.
      where: (part, start, end) of each token in its part's text.
      labels: [T] float, 1 where the token overlaps an evidence span.
    """

    index: list[int]
    where: list[tuple[int, int, int]]
    labels: torch.Tensor


def document_tokens(
    model: encoder.MaskedEncoder,
    example: schema.Example,
    spans: Sequence[schema.Span] = (),
) -> DocumentTokens:
    """The document tokens of `example`, labelled against `spans`."""
    tokens = encoder.document_tokens(example, model.tokenizer, model.fmt)
    labels = torch.tensor(
        [
            float(
                any(
                    s.part == p and start < s.end and end > s.start
                    for s in spans
                )
            )
            for _, (p, start, end) in tokens
        ]
    )
    return DocumentTokens(
        [t for t, _ in tokens], [w for _, w in tokens], labels
    )


@dataclasses.dataclass(frozen=True)
class Prediction:
    """The head's evidence for one answer.

    Attributes:
      answer: The option the evidence is for (release order).
      q: softmax(s)[answer].
      tokens: The document tokens.
      probabilities: [T] evidence probability per document token.
    """

    answer: int
    q: float
    tokens: DocumentTokens
    probabilities: torch.Tensor


@torch.no_grad()
def predict(
    model: encoder.MaskedEncoder,
    head: EvidenceHead,
    example: schema.Example,
    answer: int | None = None,
) -> Prediction:
    """Evidence probabilities for `answer` (default: the predicted option)."""
    ex, hidden, s = model.forward_row(example)
    answer = int(s.argmax()) if answer is None else answer
    doc = document_tokens(model, example)
    if doc.index:
        logits = head(hidden[doc.index], hidden[ex.masks[answer]])
        p = torch.sigmoid(logits).float().cpu()
    else:
        p = torch.zeros(0)
    q = torch.softmax(s, 0)[answer].item()
    return Prediction(answer, q, doc, p)
