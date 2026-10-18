"""Evaluation of a decision model on the eval split.

Usage (a trained run, or the base model without --adapter):
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src .venv/bin/python -m evaluation.evaluate \
      --model Qwen/Qwen3-0.6B --adapter runs/qwen3_0.6b/adapter \
      --out runs/qwen3_0.6b --tag final
--data is a local folder / .jsonl file or a Hugging Face dataset id
(data.load_rows); external benchmarks are in benchmarks.py.

Metrics per dataset (and their mean, "ALL"): accuracy of argmax s_k (Eq. 1),
mean log P(gold), the root's legal mass, and text mode on a subset (the
paper's check: greedy generation after "Best answer: [", up to 32 tokens):
the share that starts with a valid "k]", is correct, and ends right after
the "]". Writes eval_<tag>.json and eval_<tag>_examples.jsonl (one record
per text-mode row, for reading the outputs by hand).
"""

import argparse
import collections
import json
import pathlib
import random
import re

from loguru import logger
import torch

from dataloader import answers
from dataloader import data
from modeling import modeling

SEED = 20260922


def subset(rows: list[dict], per: int, seed: int) -> list[dict]:
    """A fixed sample of `per` rows per dataset."""
    by: dict[str, list] = collections.defaultdict(list)
    for r in rows:
        by[r["source"]].append(r)
    rng = random.Random(seed)
    out = []
    for name in sorted(by):
        out += rng.sample(by[name], min(per, len(by[name])))
    return out


def evaluate(model, tok, rows: list[dict], gen_rows: list[dict], device: str):
    """Per-dataset metrics, and one example record per text-mode row."""
    was_training = model.training
    model.eval()
    stats: dict = collections.defaultdict(lambda: collections.defaultdict(float))
    examples = []
    stop = [tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id]
    gen_ids = {id(r) for r in gen_rows}
    with torch.no_grad():
        for r in rows:
            ex = answers.encode(r, tok, None)
            prompt = ex["input_ids"][: ex["path"][0][0] + 1]
            s, mass, _ = answers.score(model, prompt, tok, ex["n_options"], device)
            st = stats[r["source"]]
            st["rows"] += 1
            st["accuracy"] += float(s.argmax().item() == ex["gold"])
            st["log_p_gold"] += torch.log_softmax(s, 0)[ex["gold"]].item()
            st["root_legal_mass"] += mass
            if id(r) not in gen_ids:
                continue
            out = model.generate(
                torch.tensor([prompt], device=device), max_new_tokens=32,
                do_sample=False, eos_token_id=stop,
                pad_token_id=tok.pad_token_id, use_cache=True,
            )[0, len(prompt):].tolist()
            text = tok.decode([t for t in out if t not in stop])
            m = re.match(r"(\d+)\]", text)
            ok = bool(m) and 1 <= int(m.group(1)) <= ex["n_options"]
            st["gen_rows"] += 1
            st["gen_valid"] += float(ok)
            st["gen_ends_after_bracket"] += float(ok and text == m.group(0)
                                                  and out[-1] in stop)
            st["gen_correct"] += float(ok and int(m.group(1)) == ex["gold"] + 1)
            options = [r["labels"][label] for label in r["labels"]]
            top = torch.softmax(s, 0).topk(min(3, len(options)))
            examples.append({
                "source": r["source"], "id": r["id"],
                "state": " / ".join(part["text"] for part in r["parts"])[:400],
                "question": r["question"],
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
    parser.add_argument("--data", default=data.TRAIN_EVAL,
                        help="local folder / .jsonl, or a Hub dataset id")
    parser.add_argument("--split", default="eval")
    parser.add_argument("--out", required=True)
    parser.add_argument("--tag", default="base")
    parser.add_argument("--eval-per", type=int, default=100)
    parser.add_argument("--gen-per", type=int, default=20)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    rows = subset(data.load_rows(args.data, args.split), args.eval_per,
                  args.seed)
    gen_rows = subset(rows, args.gen_per, args.seed)
    if args.adapter:
        model = modeling.load_trained(args.model, args.adapter)
    else:
        model = modeling.load_base(args.model).to("cuda").eval()
    metrics, examples = evaluate(model, tok, rows, gen_rows, "cuda")
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save(out, args.tag, metrics, examples)
    log_summary(args.tag, metrics)


if __name__ == "__main__":
    main()
