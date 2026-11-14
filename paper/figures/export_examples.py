"""Exports the appendix examples: decision, option probabilities, evidence.

  .venv/bin/python paper/figures/export_examples.py

Rule, fixed in advance: the test rows of the evidence study
(llm_jev.evidence.sentences.rows_with_evidence, part test, all rows, split
order); for each dataset, the first row whose prompt has at most MAX_TOKENS
tokens. Token scores come from the evidence head on the sigmoid-normalised
encoder, with its dev-tuned threshold. Writes paper/figures/examples.json.
"""

import json
import pathlib

import torch

from llm_jev import config
from llm_jev.data import storage
from llm_jev.evidence import head as head_lib
from llm_jev.evidence import sentences
from llm_jev.losses import option
from llm_jev.models import encoder

MAX_TOKENS = 1400
ROOT = pathlib.Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "runs/evidence_ultrabert_380m_sigmoidnorm"
OUT = pathlib.Path(__file__).parent / "examples.json"


def main() -> None:
    """Writes examples.json."""
    cfg: config.EvidenceConfig = config.resolve(
        config.load(config.EvidenceConfig, EVIDENCE / "config_test.yaml")
    )
    model = encoder.MaskedEncoder.from_run(cfg.encoder.run, attention="sdpa")
    hidden = model.unwrapped.model.config.hidden_size
    head = head_lib.EvidenceHead(hidden, cfg.head.width, cfg.head.dropout)
    head.load_state_dict(torch.load(EVIDENCE / "head.pt"))
    head = head.to(model.device).eval()
    threshold = json.loads((EVIDENCE / "threshold_test.json").read_text())
    rows = sentences.rows_with_evidence(
        storage.load_examples(cfg.data.source, "eval"), 0, 0, "test"
    )
    chosen = {}
    for r in rows:
        n_tokens = model.num_tokens(r)
        if r.source in chosen or n_tokens > MAX_TOKENS:
            continue
        s = model.score([r])[0]
        pred = head_lib.predict(model, head, r)
        chosen[r.source] = {
            "row": {
                "id": r.id,
                "source": r.source,
                "question": r.question,
                "options": list(r.options),
                "gold": r.gold,
                "answer": int(s.argmax()),
                "q": option.probabilities(s, model.loss_kind).tolist(),
                "parts": [p.name for p in r.parts],
                "partTexts": [p.text for p in r.parts],
                "spans": [
                    {"part": x.part, "start": x.start, "end": x.end}
                    for x in r.spans
                ],
                "tokens": n_tokens,
            },
            "evidence": {
                "threshold": threshold["threshold"],
                "tokens": [
                    {"part": p, "start": a, "end": b, "p": float(v)}
                    for (p, a, b), v in zip(
                        pred.tokens.where, pred.probabilities, strict=True
                    )
                ],
            },
        }
    OUT.write_text(json.dumps({"run": cfg.encoder.run, "examples": chosen}))
    print({k: v["row"]["tokens"] for k, v in chosen.items()})


if __name__ == "__main__":
    main()
