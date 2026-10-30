"""Decisions on a dataset split, with ContextCite evidence behind --cite.

Without --cite, rows are scored in length-sorted batches (answers.score_batch,
SDPA). With --cite, every row runs masked_cite.attribute (full masking on FA2,
fused kernels unless --no-fuse) and the decision is its full-prompt copy.
Writes one JSON line per row: id, source, answer (0-based), option, q, gold,
correct, and with --cite the --top-k highest-weighted sentences ("evidence")
and the seconds the row took.

  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m llm_jev.serving.predict \
      --model Athekunal/llm-jev-qwen3-0.6b --data Athekunal/llm-jev-reason-long \
      --split eval --limit 100 --cite --out runs/predict/eval.jsonl
"""

import argparse
import json
import pathlib
import time
from typing import Any

from loguru import logger
import torch

from llm_jev.dataloader import answers
from llm_jev.dataloader import schema
from llm_jev.dataloader import storage
from llm_jev.modeling import fa2
from llm_jev.modeling import masked_cite
from llm_jev.modeling import modeling


def decide(model, tok, rows: list[schema.Example], device: str = "cuda", batch_tokens: int = 32768,
           batch_rows: int = 64) -> list[torch.Tensor]:
    """s [K] for every row, from batched scoring of the prompts (options in release order)."""
    encoded = [answers.encode(r, tok, None) for r in rows]
    prompts = [ex["input_ids"][: ex["path"][0][0] + 1] for ex in encoded]
    s: list = [None] * len(rows)
    with torch.no_grad():
        for batch in answers.batches([len(p) for p in prompts], batch_tokens, batch_rows):
            scored = answers.score_batch(model, [prompts[i] for i in batch], tok,
                                         [len(rows[i].options) for i in batch], device, batch_tokens, batch_rows)
            for i, (x, _, _) in zip(batch, scored):
                s[i] = x
    return s


def evidence(row: schema.Example, sources: list[dict[str, Any]], k: int = 3) -> list[dict[str, Any]]:
    """The k highest-weighted sources with w > 0, strongest first, with their text."""
    top = sorted((x for x in sources if x["w"] > 0), key=lambda x: -x["w"])[:k]
    return [{"part": x["part"], "start": x["start"], "end": x["end"], "w": x["w"],
             "text": row.parts[x["part"]].text[x["start"]: x["end"]]} for x in top]


def record(row: schema.Example, answer: int, q: float) -> dict[str, Any]:
    """One output line without evidence."""
    return {"id": row.id, "source": row.source, "answer": answer, "option": row.options[answer],
            "q": q, "gold": row.gold, "correct": answer == row.gold}


def main() -> None:
    """Writes the decisions (and with --cite, the evidence) for a split."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="model id or folder (merged, or the base of --adapter)")
    parser.add_argument("--adapter", help="LoRA adapter on --model (merged before use)")
    parser.add_argument("--data", default=storage.DEFAULT_SOURCE, help="local folder / .parquet, or a Hub dataset id")
    parser.add_argument("--split", default="eval")
    parser.add_argument("--source", nargs="*", help="only these datasets")
    parser.add_argument("--limit", type=int, help="the first N rows only")
    parser.add_argument("--cite", action="store_true", help="ContextCite evidence for every decision")
    parser.add_argument("--fuse", action=argparse.BooleanOptionalAction, default=True,
                        help="with --cite: fused kernels (modeling.fuse; compiles on the first row)")
    parser.add_argument("--top-k", type=int, default=3, help="with --cite: evidence sentences per row")
    parser.add_argument("--out", required=True, help="JSON lines file")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    rows = [r for r in storage.load_examples(args.data, args.split)
            if not args.source or r.source in args.source][: args.limit]
    model = modeling.load(args.model, args.adapter)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        if not args.cite:
            for row, s in zip(rows, decide(model, tok, rows)):
                answer = int(s.argmax())
                f.write(json.dumps(record(row, answer, torch.softmax(s, 0)[answer].item())) + "\n")
        else:
            fa2.use(model)
            if args.fuse:
                modeling.fuse(model)
            for i, row in enumerate(rows):
                t0 = time.time()
                res = masked_cite.attribute(model, tok, row, removal="full")
                line = {**record(row, res["answer"], res["q"]), "seconds": time.time() - t0,
                        "evidence": evidence(row, res["sources"], args.top_k)}
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
                if (i + 1) % 50 == 0 or i + 1 == len(rows):
                    logger.info("{}/{} rows", i + 1, len(rows))
    logger.info("wrote {} decisions to {}", len(rows), out)


if __name__ == "__main__":
    main()
