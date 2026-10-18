"""Evidence-span heads, to be put on top of the decision model.

Kept from the earlier joint (letter-code) trainer, unchanged; see
docs/span_heads.md. Inputs are the language model's last-layer states:
SpanHead (question-conditioned document states), EvidenceHead (draft answer
-> spans -> strict final answer) and MatchHead (GLiNER / ColBERT-style
matching).
"""

import math

import torch
from torch import nn


class SpanHead(nn.Module):
    """Document tokens attend to question tokens; one evidence logit each."""

    def __init__(self, hidden: int, dim: int = 512, heads: int = 8):
        super().__init__()
        self.doc = nn.Linear(hidden, dim)
        self.q = nn.Linear(hidden, dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1)
        )

    def forward(
        self, doc: torch.Tensor, question: torch.Tensor
    ) -> torch.Tensor:
        """Logits for doc (n, hidden) given question (m, hidden)."""
        d = self.doc(doc).unsqueeze(0)
        q = self.q(question).unsqueeze(0)
        ctx, _ = self.attn(d, q, q, need_weights=False)
        return self.out(self.norm(d + ctx)).squeeze(0).squeeze(-1)


class _Attn(nn.Module):
    """Multi-head attention through scaled_dot_product_attention."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.h = heads
        self.q, self.k, self.v, self.o = (nn.Linear(dim, dim) for _ in range(4))

    def forward(self, x, mem, bias=None):
        n, m, d = x.shape[0], mem.shape[0], x.shape[1]
        split = lambda t, L: t.view(L, self.h, d // self.h).transpose(0, 1)
        q, k, v = split(self.q(x), n), split(self.k(mem), m), split(self.v(mem), m)
        mask = None if bias is None else bias.view(1, 1, m).expand(self.h, n, m)
        y = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o(y.transpose(0, 1).reshape(n, d))


class EvidenceHead(nn.Module):
    """Draft answer -> evidence spans -> final answer from the evidence.

    Document states come from a document-first prompt, so they do not depend
    on the question. Each layer: document tokens attend to the question-side
    states (question, choices, and the draft answer as the choices' mix
    weighted by the LM's draft probabilities), then to each other. The final
    answer is strict: its query is the question encoded without the
    document, and it reads the raw document states only through attention
    biased by log p(evidence); choices are scored against their
    document-free encodings. The gold answer is never an input.

    Reader: with reader_layers = 0, one read by the mean question vector;
    otherwise the question's tokens are the queries, and each reader layer
    reads the document (evidence-biased) and lets the question tokens attend
    to each other, a small decoder over the cached document states.
    """

    def __init__(self, hidden: int, dim: int = 512, heads: int = 8,
                 layers: int = 2, reader_layers: int = 0):
        super().__init__()
        self.doc, self.mem, self.iso, self.raw = (
            nn.Linear(hidden, dim) for _ in range(4))
        self.blocks = nn.ModuleList(
            nn.ModuleDict({
                "n1": nn.LayerNorm(dim), "cross": _Attn(dim, heads),
                "n2": nn.LayerNorm(dim), "self": _Attn(dim, heads),
                "n3": nn.LayerNorm(dim),
                "ff": nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(),
                                    nn.Linear(4 * dim, dim)),
            }) for _ in range(layers))
        self.span = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim),
                                  nn.GELU(), nn.Linear(dim, 1))
        self.read = _Attn(dim, heads)
        self.reader = nn.ModuleList(
            nn.ModuleDict({
                "n1": nn.LayerNorm(dim), "cross": _Attn(dim, heads),
                "n2": nn.LayerNorm(dim), "self": _Attn(dim, heads),
                "n3": nn.LayerNorm(dim),
                "ff": nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(),
                                    nn.Linear(4 * dim, dim)),
            }) for _ in range(reader_layers))
        self.final = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(),
                                   nn.Linear(dim, dim))

    def spans(self, h_doc, h_q, i_choices, draft_p):
        """Span logits (n_doc,) from the span blocks alone (no reader)."""
        c = self.iso(i_choices)  # (k, dim)
        mem = torch.cat([self.mem(h_q), (draft_p.unsqueeze(1) * c).sum(0, True)])
        x = self.doc(h_doc)
        for b in self.blocks:
            x = x + b["cross"](b["n1"](x), mem)
            y = b["n2"](x)
            x = x + b["self"](y, y)
            x = x + b["ff"](b["n3"](x))
        return self.span(x).squeeze(-1)

    def forward(self, h_doc, h_q, i_q, i_choices, draft_p):
        """Span logits (n_doc,) and final answer logits (n_choices,)."""
        c = self.iso(i_choices)  # (k, dim)
        span = self.spans(h_doc, h_q, i_choices, draft_p)
        q_tok = self.iso(i_q)
        query = q_tok.mean(0, keepdim=True)
        bias = nn.functional.logsigmoid(span)  # log p(evidence)
        doc = self.raw(h_doc)
        if self.reader:
            x_q = q_tok
            for b in self.reader:
                x_q = x_q + b["cross"](b["n1"](x_q), doc, bias)
                y = b["n2"](x_q)
                x_q = x_q + b["self"](y, y)
                x_q = x_q + b["ff"](b["n3"](x_q))
            read = x_q.mean(0, keepdim=True)
        else:
            read = self.read(query, doc, bias)
        z = self.final(torch.cat([read, query], -1))  # (1, dim)
        return span, (z @ c.T).squeeze(0) / c.shape[-1] ** 0.5


class MatchHead(nn.Module):
    """GLiNER-style matching with ColBERT-style late interaction.

    Document tokens (document-first pass, cacheable) get one bidirectional
    self-attention block inside the head; the question's tokens and the
    answer choices are encoded without the document. Evidence logit per
    document token = a learned mix of:
      match: soft maximum over question tokens of the cosine similarity
        (temperature-scaled logsumexp; ColBERT's MaxSim made soft);
      pick: log of how much attention the question tokens give it, each
        question token spreading a softmax over the document (each query
        token picks the document tokens it is interested in);
      answer: similarity to the answer choices weighted by the LM letter's
        probabilities (GLiNER's label matching, for the model's own answer);
      prior: a linear score of the token alone (lets error-finding tasks,
        whose evidence shares no words with the question, learn something).
    Everything is soft; the answer itself stays the LM letter.
    """

    def __init__(self, hidden: int, dim: int = 512, heads: int = 8):
        super().__init__()
        self.doc, self.q, self.c = (nn.Linear(hidden, dim) for _ in range(3))
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.self_attn = _Attn(dim, heads)
        self.ff = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(),
                                nn.Linear(4 * dim, dim))
        self.kd, self.kq, self.kg = (nn.Linear(dim, dim) for _ in range(3))
        self.scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.prior = nn.Linear(dim, 1)
        self.mix = nn.Linear(3, 1)

    def forward(self, h_doc, i_q, i_choices, draft_p):
        """Span logits (n_doc,)."""
        x = self.doc(h_doc)
        y = self.n1(x)
        x = x + self.self_attn(y, y)
        x = x + self.ff(self.n2(x))
        q = self.q(i_q)
        s = self.scale.exp()
        sim = (nn.functional.normalize(self.kd(x), dim=-1)
               @ nn.functional.normalize(self.kq(q), dim=-1).T) * s  # (n, m)
        match = torch.logsumexp(sim, 1) - math.log(sim.shape[1])
        pick = torch.log(torch.softmax(sim, 0).sum(1) + 1e-6)
        c = nn.functional.normalize(self.c(i_choices), dim=-1)  # (k, dim)
        g = (nn.functional.normalize(self.kg(x), dim=-1) @ c.T) * s  # (n, k)
        answer = g @ draft_p
        feats = torch.stack([match, pick, answer], -1)
        return (self.mix(feats) + self.prior(x)).squeeze(-1)


def build_head(name: str, hidden: int, args: dict) -> nn.Module:
    """The span head called name ("cross", "evidence", "evspan", "match")."""
    if name in ("evidence", "evspan"):
        return EvidenceHead(hidden, layers=args.get("head_layers", 2),
                            reader_layers=args.get("reader_layers", 0))
    if name == "match":
        return MatchHead(hidden)
    return SpanHead(hidden)
