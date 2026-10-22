"""Evaluation of a decision model on the eval split.

Usage (a trained run, or the base model without --adapter):
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m llm_jev.evaluation.evaluate \
      --model Qwen/Qwen3-0.6B --adapter runs/qwen3_0.6b/adapter \
      --out runs/qwen3_0.6b --tag final
Metrics per dataset and their mean ("ALL"): accuracy of argmax s_k, mean
log P(gold), the root's legal mass, and on a subset, the validity and accuracy
of unguided greedy text. Writes eval_<tag>.json and eval_<tag>_examples.jsonl.
External benchmarks are in benchmarks.py.
"""

import argparse
import collections
import json
import pathlib
import random
import re
import time

from loguru import logger
import torch

from llm_jev.dataloader import answers
from llm_jev.dataloader import schema
from llm_jev.dataloader import storage
from llm_jev.dataloader.sources import base
from llm_jev.modeling import modeling

SEED = base.SEED


def subset(
    rows: list[schema.Example], per: int, seed: int
) -> list[schema.Example]:
    """A fixed sample of `per` examples per dataset."""
    by: dict[str, list] = collections.defaultdict(list)
    for r in rows:
        by[r.source].append(r)
    rng = random.Random(seed)
    out = []
    for name in sorted(by):
        out += rng.sample(by[name], min(per, len(by[name])))
    return out


def evaluate(model, tok, rows: list[schema.Example],
             gen_rows: list[schema.Example], device: str,
             batch_tokens: int = 32768, batch_rows: int = 64):
    """Per-dataset metrics, and one example record per text-mode row.

    Rows are scored in length-sorted batches bounded by batch_tokens and
    batch_rows; text mode runs one row at a time.
    """
    was_training = model.training
    model.eval()
    stats: dict = collections.defaultdict(lambda: collections.defaultdict(float))
    examples = []
    stop = [tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id]
    gen_ids = {id(r) for r in gen_rows}
    encoded = [answers.encode(r, tok, None) for r in rows]
    prompts = [ex["input_ids"][: ex["path"][0][0] + 1] for ex in encoded]
    scored: list = [None] * len(rows)
    start = last = time.time()
    done = 0
    with torch.no_grad():
        for batch in answers.batches([len(p) for p in prompts], batch_tokens, batch_rows):
            results = answers.score_batch(
                model, [prompts[i] for i in batch], tok,
                [encoded[i]["n_options"] for i in batch], device, batch_tokens,
                batch_rows)
            for i, res in zip(batch, results):
                scored[i] = res
            done += len(batch)
            if time.time() - last > 60 or done == len(rows):
                last, el = time.time(), time.time() - start
                logger.info("scored {}/{} rows, {:.1f}m, eta {:.1f}m", done,
                            len(rows), el / 60, el / done * (len(rows) - done) / 60)
        for r, ex, prompt, (s, mass, _) in zip(rows, encoded, prompts, scored):
            st = stats[r.source]
            st["rows"] += 1
            st["accuracy"] += float(s.argmax().item() == ex["gold"])
            st["log_p_gold"] += torch.log_softmax(s, 0)[ex["gold"]].item()
            st["root_legal_mass"] += mass
            if id(r) not in gen_ids:
                continue
            # Unguided greedy text, stopped at "]" and capped at the longest suffix.
            longest = max(map(len, answers.candidates(tok, ex["n_options"])[0]))
            out = model.generate(
                torch.tensor([prompt], device=device), max_new_tokens=longest,
                do_sample=False, eos_token_id=stop, stop_strings=["]"], tokenizer=tok,
                pad_token_id=tok.pad_token_id, use_cache=True,
            )[0, len(prompt):].tolist()
            text = tok.decode([t for t in out if t not in stop])
            m = re.match(r"(\d+)\]", text)
            ok = bool(m) and 1 <= int(m.group(1)) <= ex["n_options"]
            st["gen_rows"] += 1
            st["gen_valid"] += float(ok)
            st["gen_correct"] += float(ok and int(m.group(1)) == ex["gold"] + 1)
            options = list(r.options)
            top = torch.softmax(s, 0).topk(min(3, len(options)))
            examples.append({
                "source": r.source, "id": r.id,
                "state": " / ".join(part.text for part in r.parts)[:400],
                "question": r.question,
                "options": options if len(options) <= 10 else options[:10] + ["…"],
                "gold": f"[{ex['gold'] + 1}] {options[ex['gold']]}",
                "scored_top3": [f"[{k + 1}] {options[k]} ({v:.3f})"
                                for v, k in zip(top.values.tolist(),
                                                top.indices.tolist())],
                "generation": tok.decode(out),
                "valid": ok,
            })
    model.train(was_training)
    metrics: dict = {}
    for name, st in sorted(stats.items()):
        metrics[name] = {k: v / (st["gen_rows"] if k.startswith("gen_")
                                 else st["rows"])
                         for k, v in st.items() if k not in ("rows", "gen_rows")}
        metrics[name]["rows"] = st["rows"]
        metrics[name]["gen_rows"] = st["gen_rows"]
    keys = [k for k in next(iter(metrics.values()))
            if k not in ("rows", "gen_rows")]
    metrics["ALL"] = {k: sum(m[k] for m in metrics.values()) / len(metrics)
                      for k in keys}
    return metrics, examples


def save(out: pathlib.Path, tag: str, metrics: dict, examples: list) -> None:
    """Writes eval_<tag>.json and eval_<tag>_examples.jsonl under out."""
    (out / f"eval_{tag}.json").write_text(json.dumps(metrics, indent=2))
    with open(out / f"eval_{tag}_examples.jsonl", "w") as f:
        for e in examples:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def log_summary(tag: str, metrics: dict) -> None:
    """Logs the mean metrics, then accuracy per dataset."""
    logger.info("eval {}: {}", tag, " ".join(
        f"{k} {v:.4f}" for k, v in metrics["ALL"].items()))
    logger.info("eval {} accuracy per dataset: {}", tag, " ".join(
        f"{n} {v['accuracy']:.3f}" for n, v in metrics.items() if n != "ALL"))


def main() -> None:
    """Evaluates a base model or a trained adapter."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", help="a saved LoRA adapter (else the base)")
    parser.add_argument("--data", default=storage.DEFAULT_SOURCE,
                        help="local folder / .parquet, or a Hub dataset id")
    parser.add_argument("--split", default="eval")
    parser.add_argument("--out", required=True)
    parser.add_argument("--tag", default="base")
    parser.add_argument("--eval-per", type=int, default=100)
    parser.add_argument("--gen-per", type=int, default=20)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    rows = subset(storage.load_examples(args.data, args.split), args.eval_per,
                  args.seed)
    gen_rows = subset(rows, args.gen_per, args.seed)
    model = modeling.load(args.model, args.adapter, merge=False)
    metrics, examples = evaluate(model, tok, rows, gen_rows, "cuda")
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save(out, args.tag, metrics, examples)
    log_summary(args.tag, metrics)


if __name__ == "__main__":
    main()
