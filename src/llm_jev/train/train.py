"""Fine-tuning a decision model: tree-factorized listwise loss + KL anchors.

Usage (one GPU per run):
  CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m llm_jev.train.train \
      --model Qwen/Qwen3-0.6B --out runs/qwen3_0.6b

Recipe of Li & Wagle, arXiv 2610.02076v3 (§3.4, Table 2):
loss = L_tree + lambda * (L_mass + L_out + L_pos) (losses.compute). Defaults
are the paper's (Table 2): LoRA rank 16, alpha 32, dropout 0 on the decoder
linear layers; torch AdamW (weight decay 0.01, gradient clipping 1.0); peak
learning rate 1e-4, linear warmup over 100 steps then linear decay to 10%
of the peak (torch LinearLR -> LinearLR in a SequentialLR);
global batch 32, one epoch, seed 20260922; BF16 weights with FP32 LoRA
parameters and optimizer states; gradient checkpointing. Every setting is a
flag (scripts/train.sh). The reference model p0 is the same model with the
adapter disabled (no cache).

Evaluates (evaluate.py) before training, every --eval-every steps (also
saving the adapter there) and at the end. --data is a local folder of
Parquet splits (built by dataloader.build) or a Hugging Face dataset id with
train and eval splits of `Example`s (storage.load_examples).
"""

import argparse
import collections
import json
import math
import pathlib
import random
import time

from loguru import logger
import torch
from torch.optim.lr_scheduler import LinearLR
from torch.optim.lr_scheduler import SequentialLR

from llm_jev.dataloader import answers
from llm_jev.dataloader import storage
from llm_jev.evaluation import evaluate
from llm_jev.train import losses
from llm_jev.modeling import modeling


def main() -> None:
    """Trains one epoch and evaluates before and after."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", default=storage.DEFAULT_SOURCE,
                        help="local folder, or a Hub dataset id")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="eval")
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4, help="peak")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1,
                        help="final learning rate / peak (linear decay)")
    parser.add_argument("--lam", type=float, default=1.0, help="anchor weight")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=10240)
    parser.add_argument("--eval-per", type=int, default=100,
                        help="eval rows per dataset")
    parser.add_argument("--gen-per", type=int, default=20,
                        help="text-mode rows per dataset (of the eval rows)")
    parser.add_argument(
        "--eval-every", type=int, default=0,
        help="also evaluate and save the adapter every N steps")
    parser.add_argument("--max-steps", type=int, default=-1, help="smoke tests")
    parser.add_argument("--seed", type=int, default=evaluate.SEED)
    args = parser.parse_args()

    from torch.utils.tensorboard import SummaryWriter
    from transformers import AutoTokenizer

    device = "cuda"
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logger.add(out / "train.log")
    (out / "args.json").write_text(json.dumps(
        {**vars(args), "off_path": answers.OFF_PATH, "top": losses.TOP},
        indent=2))
    tok = AutoTokenizer.from_pretrained(args.model)
    pad = tok.pad_token_id
    train = storage.load_examples(args.data, args.train_split)
    evals = evaluate.subset(storage.load_examples(args.data, args.eval_split),
                            args.eval_per, args.seed)
    gen_rows = evaluate.subset(evals, args.gen_per, args.seed)

    model = modeling.load_base(args.model).to(device)
    model.config.use_cache = False
    model = modeling.add_lora(model, args.lora_rank)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    lora = [p for p in model.parameters() if p.requires_grad]
    logger.info("trainable parameters: {:,} ({})",
                sum(p.numel() for p in lora), lora[0].dtype)
    opt = torch.optim.AdamW(lora, lr=args.lr, weight_decay=args.weight_decay)
    total = len(train) // args.batch_size
    if args.max_steps > 0:
        total = min(total, args.max_steps)
    warm = min(args.warmup_steps, total)
    sched = SequentialLR(opt, [
        LinearLR(opt, start_factor=1 / warm, end_factor=1.0,
                 total_iters=max(1, warm - 1)),
        LinearLR(opt, start_factor=1.0, end_factor=args.min_lr_ratio,
                 total_iters=max(1, total - warm)),
    ], milestones=[warm])
    writer = SummaryWriter(str(out / "tb"))

    def run_eval(step: int, tag: str) -> None:
        t0 = time.time()
        metrics, examples = evaluate.evaluate(model, tok, evals, gen_rows,
                                              device)
        evaluate.save(out, tag, metrics, examples)
        for name, v in metrics.items():
            for k, x in v.items():
                writer.add_scalar(f"eval/{name}/{k}", x, step)
        evaluate.log_summary(tag, metrics)
        logger.info("eval {} at step {} took {:.1f}m", tag, step,
                    (time.time() - t0) / 60)

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
            ex = answers.encode(
                train[i], tok, random.Random(f"{args.seed}:{i}"))
            if len(ex["input_ids"]) > args.max_tokens:
                too_long += 1
                continue
            L = losses.compute(model, ex, device, pad)
            loss = L["tree"] + args.lam * (L["mass"] + L["out"] + L["pos"])
            (loss / args.batch_size).backward()
            for k, v in L.items():
                sums[k] += v.item()
            n += 1
        norm = torch.nn.utils.clip_grad_norm_(lora, args.max_grad_norm)
        if math.isfinite(norm.item()):
            opt.step()
        else:
            skipped += 1
            logger.warning("step {}: skipped update, grad_norm {}", step + 1,
                           norm.item())
        opt.zero_grad(set_to_none=True)
        sched.step()
        for k in ("tree", "mass", "out", "pos"):
            writer.add_scalar(f"train/{k}", sums[k] / max(n, 1), step + 1)
        writer.add_scalar("train/grad_norm", norm.item(), step + 1)
        writer.add_scalar("train/lr", sched.get_last_lr()[0], step + 1)
        if (step + 1) % 5 == 0 or step == 0:
            el = time.time() - start
            logger.info(f"step {step + 1}/{total} tree {sums['tree'] / max(n, 1):.4f} "
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
    logger.info("done")
    writer.close()


if __name__ == "__main__":
    main()
