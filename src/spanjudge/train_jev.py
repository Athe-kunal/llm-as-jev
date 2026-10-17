"""LLM-as-Jev fine-tuning: tree-factorized listwise loss + KL anchors, LoRA.

Usage (one GPU per run):
  CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src .venv/bin/python \
      -m spanjudge.train_jev --model Qwen/Qwen3-0.6B --out runs/jev_qwen3_0.6b

Recipe of Li & Wagle, arXiv 2610.02076v3 (§3.4, Table 2; docs/architecture.md):
loss = L_tree + lambda * (L_mass + L_out + L_pos) (jev.losses), LoRA rank 16,
alpha 32, dropout 0 on the decoder linear layers, AdamW (weight decay 0.01,
gradient clipping 1.0), peak learning rate 1e-4, linear warmup over 100
steps then linear decay to 10% of the peak, global batch 32, one epoch,
seed 20260922, BF16 weights with FP32 LoRA parameters and optimizer states,
gradient checkpointing. The reference model p0 is the same model with the
adapter disabled (no cache).

Evaluation before training and after it, on --eval-per rows per dataset of
the eval split: accuracy of argmax s_k (Eq. 1), mean log P(gold), the
root's legal mass, and text mode (greedy generation after "Best answer: [",
up to 32 tokens) on --gen-per rows per dataset: the share that starts with
a valid "k]", and the share that ends right after it.
"""

import argparse
import collections
import json
import math
import pathlib
import random
import re
import time

import torch
from torch import nn

from spanjudge import jev
from spanjudge import train_joint

REVISIONS = {
    "Qwen/Qwen3-0.6B": "c1899de289a04d12100db370d81485cdf75e47ca",
    "Qwen/Qwen3.5-4B": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
}


def evaluate(model, tok, rows, gen_rows, device: str, pad: int):
    """Per-dataset accuracy, log P(gold), root legal mass and text mode.

    Also returns one record per text-mode row (question, options, gold, the
    scored top-3 and the generation) for reading the outputs by hand.
    """
    model.eval()
    stats: dict = collections.defaultdict(lambda: collections.defaultdict(float))
    examples = []
    stop = [tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id]
    gen_ids = {id(r) for r in gen_rows}
    with torch.no_grad():
        for r in rows:
            ex = jev.encode(r, tok, None)
            prompt = ex["input_ids"][: ex["path"][0][0] + 1]
            s, mass = jev.score(model, prompt, tok, ex["n_options"], device, pad)
            st = stats[r["source"]]
            st["rows"] += 1
            st["accuracy"] += float(s.argmax().item() == ex["gold"])
            st["log_p_gold"] += torch.log_softmax(s, 0)[ex["gold"]].item()
            st["root_legal_mass"] += mass
            if id(r) not in gen_ids:
                continue
            out = model.generate(
                torch.tensor([prompt], device=device), max_new_tokens=32,
                do_sample=False, eos_token_id=stop, pad_token_id=pad,
                use_cache=True)[0, len(prompt):].tolist()
            text = tok.decode([t for t in out if t not in stop])
            m = re.match(r"(\d+)\]", text)
            ok = bool(m) and 1 <= int(m.group(1)) <= ex["n_options"]
            st["gen_rows"] += 1
            st["gen_valid"] += float(ok)
            st["gen_ends_after_bracket"] += float(ok and text == m.group(0)
                                                  and out[-1] in stop)
            st["gen_correct"] += float(ok and int(m.group(1)) == ex["gold"] + 1)
            options = [r["labels"][label] for label in r["labels"]]
            p = torch.softmax(s, 0)
            top = p.topk(min(3, len(options)))
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
    model.train()
    out: dict = {}
    for name, st in sorted(stats.items()):
        out[name] = {k: v / (st["gen_rows"] if k.startswith("gen_") and
                             k != "gen_rows" else st["rows"])
                     for k, v in st.items() if k not in ("rows", "gen_rows")}
        out[name]["rows"], out[name]["gen_rows"] = st["rows"], st["gen_rows"]
    keys = [k for k in next(iter(out.values())) if k not in ("rows", "gen_rows")]
    out["ALL"] = {k: sum(o[k] for o in out.values()) / len(out) for k in keys}
    return out, examples


def main() -> None:
    """Trains one epoch and evaluates before and after."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, choices=sorted(REVISIONS))
    parser.add_argument("--train", default="data/splits/reason_long_12k/train.jsonl")
    parser.add_argument("--eval", default="data/splits/reason_long_12k/eval.jsonl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lam", type=float, default=1.0, help="anchor weight")
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=10240)
    parser.add_argument("--eval-per", type=int, default=100,
                        help="eval rows per dataset")
    parser.add_argument("--gen-per", type=int, default=20,
                        help="text-mode rows per dataset (of the eval rows)")
    parser.add_argument("--max-steps", type=int, default=-1, help="smoke tests")
    parser.add_argument(
        "--eval-every", type=int, default=0,
        help="also evaluate (with examples) and save the adapter every N steps")
    parser.add_argument(
        "--eval-only", action="store_true",
        help="no training: evaluate the base model, or --adapter, as eval_<tag>")
    parser.add_argument("--adapter", help="a saved LoRA adapter (--eval-only)")
    parser.add_argument("--tag", default="base", help="--eval-only file tag")
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args()

    from peft import LoraConfig
    from peft import get_peft_model
    from torch.utils.tensorboard import SummaryWriter
    from transformers import AutoTokenizer

    device = "cuda"
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    revision = REVISIONS[args.model]
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(
        {**vars(args), "revision": revision, "off_path": jev.OFF_PATH,
         "top": jev.TOP}, indent=2))
    tok = AutoTokenizer.from_pretrained(args.model, revision=revision)
    pad = tok.pad_token_id
    with open(args.train) as f:
        train = [json.loads(line) for line in f]
    with open(args.eval) as f:
        evals = train_joint.eval_subset(
            [json.loads(line) for line in f], args.eval_per, args.seed)
    gen_rows = train_joint.eval_subset(evals, args.gen_per, args.seed)

    model = train_joint.load_base(args.model, revision).to(device)
    model.config.use_cache = False
    if args.eval_only and args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    # LoRA on the decoder's linear layers (not the vision tower or LM head)
    targets = [n for n, m in model.named_modules()
               if isinstance(m, nn.Linear) and ".layers." in n
               and "visual" not in n]
    if not (args.eval_only and args.adapter):  # a fresh adapter is the base
        model = get_peft_model(model, LoraConfig(
            r=args.lora_rank, lora_alpha=2 * args.lora_rank, lora_dropout=0.0,
            target_modules=targets, bias="none", task_type="CAUSAL_LM"))
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    lora = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable parameters: {sum(p.numel() for p in lora):,} "
          f"({lora[0].dtype})", flush=True)
    opt = torch.optim.AdamW(lora, lr=args.lr, weight_decay=0.01)
    total = len(train) // args.batch_size
    if args.max_steps > 0:
        total = min(total, args.max_steps)
    warm = min(args.warmup_steps, total)

    def lr_scale(step: int) -> float:
        if step < warm:
            return (step + 1) / warm
        return 1.0 - 0.9 * (step - warm) / max(1, total - warm)

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_scale)
    writer = SummaryWriter(str(out / "tb"))
    log = open(out / "train.log", "a")

    def say(line: str) -> None:
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    def run_eval(step: int, tag: str) -> None:
        t0 = time.time()
        m, examples = evaluate(model, tok, evals, gen_rows, device, pad)
        (out / f"eval_{tag}.json").write_text(json.dumps(m, indent=2))
        with open(out / f"eval_{tag}_examples.jsonl", "w") as f:
            for e in examples:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        for name, v in m.items():
            for k, x in v.items():
                writer.add_scalar(f"eval/{name}/{k}", x, step)
        a = m["ALL"]
        say(f"eval {tag} (step {step}, {(time.time() - t0) / 60:.1f}m): "
            + " ".join(f"{k} {v:.4f}" for k, v in a.items()))
        say("  per dataset accuracy: " + " ".join(
            f"{n} {v['accuracy']:.3f}" for n, v in m.items() if n != "ALL"))

    if args.eval_only:
        run_eval(0, args.tag)
        return
    run_eval(0, "base")
    model.train()
    order = list(range(len(train)))
    random.Random(args.seed).shuffle(order)
    start, skipped, too_long = time.time(), 0, 0
    for step in range(total):
        batch = order[step * args.batch_size : (step + 1) * args.batch_size]
        sums = collections.Counter()
        n = 0
        for i in batch:
            ex = jev.encode(train[i], tok, random.Random(f"{args.seed}:{i}"))
            if len(ex["input_ids"]) > args.max_tokens:
                too_long += 1
                continue
            L = jev.losses(model, ex, device, pad)
            loss = L["tree"] + args.lam * (L["mass"] + L["out"] + L["pos"])
            (loss / args.batch_size).backward()
            for k, v in L.items():
                sums[k] += v.item()
            sums["tokens"] += len(ex["input_ids"]) * (1 + len(ex["off_path"]))
            n += 1
        norm = torch.nn.utils.clip_grad_norm_(lora, 1.0)
        if math.isfinite(norm.item()):
            opt.step()
        else:
            skipped += 1
            say(f"step {step + 1}: skipped update, grad_norm {norm.item()}")
        opt.zero_grad(set_to_none=True)
        sched.step()
        for k in ("tree", "mass", "out", "pos"):
            writer.add_scalar(f"train/{k}", sums[k] / max(n, 1), step + 1)
        writer.add_scalar("train/grad_norm", norm.item(), step + 1)
        writer.add_scalar("train/lr", sched.get_last_lr()[0], step + 1)
        if (step + 1) % 5 == 0 or step == 0:
            el = time.time() - start
            say(f"step {step + 1}/{total} tree {sums['tree'] / max(n, 1):.4f} "
                f"mass {sums['mass'] / max(n, 1):.5f} out "
                f"{sums['out'] / max(n, 1):.4f} pos {sums['pos'] / max(n, 1):.4f}"
                f" grad_norm {norm.item():.3f} lr {sched.get_last_lr()[0]:.2e}"
                f" skipped {skipped} too_long {too_long} elapsed {el / 60:.1f}m"
                f" eta {el / (step + 1) * (total - step - 1) / 60:.0f}m")
        if args.eval_every and (step + 1) % args.eval_every == 0 \
                and step + 1 < total:
            model.save_pretrained(out / "checkpoints" / f"step-{step + 1:04d}")
            run_eval(step + 1, f"step{step + 1:04d}")
    model.save_pretrained(out / "adapter")
    run_eval(total, "final")
    say("done")
    writer.close()
    log.close()


if __name__ == "__main__":
    main()
