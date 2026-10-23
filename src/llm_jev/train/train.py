"""Fine-tuning a decision model: tree-factorized listwise loss + KL anchors.

Usage (Accelerate, FSDP2 or DDP over the visible GPUs; see scripts/train.sh):
  CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/accelerate launch \
      --config_file configs/accelerate_fsdp2.yaml --num_processes 4 \
      -m llm_jev.train.train --model Qwen/Qwen3-0.6B --out runs/qwen3_0.6b

Follows Li & Wagle, arXiv 2610.02076v3 (§3.4); defaults are the paper's Table 2
(every setting is a flag, see scripts/train.sh). Evaluates before training, at
the end and every --eval-every steps. --data is a local folder of Parquet
splits or a Hugging Face dataset id.

--batch-size is the global batch: each process takes batch_size /
num_processes rows per step. Rows longer than --max-tokens are dropped before
training, since every process must run the same number of forward and backward
passes. Evaluation runs the same rows on every process; the main process
writes the results.
"""

import argparse
import contextlib
import json
import math
import pathlib
import random
import sys
import time

from loguru import logger
import torch
from torch.optim.lr_scheduler import LinearLR
from torch.optim.lr_scheduler import SequentialLR

from llm_jev.dataloader import answers
from llm_jev.dataloader import storage
from llm_jev.evaluation import evaluate
from llm_jev.modeling import modeling
from llm_jev.train import losses

TERMS = ("tree", "mass", "out", "pos")


def fitting_rows(acc, train: list, tok, args) -> list[int]:
    """Indices of the training rows of at most --max-tokens tokens.

    Each process encodes a share of the rows, with the same per-row rng the
    training loop uses.
    """
    from accelerate.utils import gather_object

    mine = [(i, len(answers.encode(
        train[i], tok, random.Random(f"{args.seed}:{i}"))["input_ids"]))
        for i in range(acc.process_index, len(train), acc.num_processes)]
    lengths = dict(gather_object(mine))
    return [i for i in range(len(train)) if lengths[i] <= args.max_tokens]


def save_adapter(acc, model, path: pathlib.Path) -> None:
    """Saves the LoRA weights from the main process; every process must call it.

    Only the adapter's shards are gathered, not the base model's.
    """
    from torch.distributed.tensor import DTensor

    peft = acc.unwrap_model(model)  # DDP's state_dict keys carry "module."
    state = {k: (v.full_tensor() if isinstance(v, DTensor) else v).cpu()
             for k, v in peft.state_dict().items() if "lora_" in k}
    peft.save_pretrained(
        path, state_dict=state, is_main_process=acc.is_main_process)


def main() -> None:
    """Trains one epoch and evaluates before and after."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", default=storage.DEFAULT_SOURCE,
                        help="local folder, or a Hub dataset id")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="eval")
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=32,
                        help="global: rows per optimizer step over all GPUs")
    parser.add_argument("--lr", type=float, default=1e-4, help="peak")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1,
                        help="final learning rate / peak (linear decay)")
    parser.add_argument("--lam", type=float, default=1.0, help="anchor weight")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=16384)
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

    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from torch.distributed.tensor import DTensor
    from transformers import AutoTokenizer

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    acc = Accelerator(log_with="tensorboard", project_dir=out)
    device = acc.device
    if args.batch_size % acc.num_processes:
        raise ValueError(f"--batch-size {args.batch_size} is not divisible by "
                         f"{acc.num_processes} processes")
    set_seed(args.seed)  # the same on every process: identical LoRA init
    if acc.is_main_process:
        logger.add(out / "train.log")
        (out / "args.json").write_text(json.dumps(
            {**vars(args), "off_path": answers.OFF_PATH, "top": losses.TOP,
             "num_processes": acc.num_processes}, indent=2))
    else:
        logger.remove()
        logger.add(sys.stderr, level="WARNING")
    acc.init_trackers("tb")
    tok = AutoTokenizer.from_pretrained(args.model)
    pad = tok.pad_token_id
    train = storage.load_examples(args.data, args.train_split)
    rows = fitting_rows(acc, train, tok, args)
    logger.info("{} of {} training rows are over {} tokens, dropped",
                len(train) - len(rows), len(train), args.max_tokens)
    if len(rows) < args.batch_size:
        raise ValueError(f"{len(rows)} training examples are fewer than one "
                         f"batch of {args.batch_size}")
    evals = evaluate.subset(storage.load_examples(args.data, args.eval_split),
                            args.eval_per, args.seed)
    gen_rows = evaluate.subset(evals, args.gen_per, args.seed)

    model = modeling.load_base(args.model)
    model.config.use_cache = False
    model = modeling.add_lora(model, args.lora_rank)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    lora = [p for p in model.parameters() if p.requires_grad]
    logger.info("trainable parameters: {:,} ({}), {} processes",
                sum(p.numel() for p in lora), lora[0].dtype, acc.num_processes)
    opt = torch.optim.AdamW(lora, lr=args.lr, weight_decay=args.weight_decay)
    total = len(rows) // args.batch_size
    if args.max_steps > 0:
        total = min(total, args.max_steps)
    warm = min(args.warmup_steps, total)
    # Not passed to prepare(): Accelerate would step it once per process.
    sched = SequentialLR(opt, [
        LinearLR(opt, start_factor=1 / warm, end_factor=1.0,
                 total_iters=max(1, warm - 1)),
        LinearLR(opt, start_factor=1.0, end_factor=args.min_lr_ratio,
                 total_iters=max(1, total - warm)),
    ], milestones=[warm])
    model, opt = acc.prepare(model, opt)

    def run_eval(step: int, tag: str) -> None:
        t0 = time.time()
        # The PEFT model itself: DDP's wrapper has no generate().
        metrics, examples = evaluate.evaluate(acc.unwrap_model(model), tok,
                                              evals, gen_rows, device)
        if acc.is_main_process:
            evaluate.save(out, tag, metrics, examples)
        acc.log({f"eval/{name}/{k}": x for name, v in metrics.items()
                 for k, x in v.items()}, step=step)
        evaluate.log_summary(tag, metrics)
        logger.info("eval {} at step {} took {:.1f}m", tag, step,
                    (time.time() - t0) / 60)

    run_eval(0, "base")
    model.train()
    order = list(rows)
    random.Random(args.seed).shuffle(order)
    start, skipped = time.time(), 0
    for step in range(total):
        batch = order[step * args.batch_size : (step + 1) * args.batch_size]
        mine = batch[acc.process_index :: acc.num_processes]
        sums = torch.zeros(len(TERMS), device=device)
        for j, i in enumerate(mine):
            ex = answers.encode(
                train[i], tok, random.Random(f"{args.seed}:{i}"))
            # Gradients are reduced across processes after the last row only.
            last = j == len(mine) - 1
            with contextlib.nullcontext() if last else acc.no_sync(model):
                L = losses.compute(model, ex, device, pad)
                loss = L["tree"] + args.lam * (L["mass"] + L["out"] + L["pos"])
                acc.backward(loss / len(mine))
            sums += torch.stack([L[k] for k in TERMS]).detach()
        sums = acc.reduce(sums, "sum") / args.batch_size
        mean = dict(zip(TERMS, sums.tolist(), strict=True))
        norm = acc.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        if isinstance(norm, DTensor):
            norm = norm.full_tensor()
        norm = norm.item()
        if math.isfinite(norm):
            opt.step()
        else:
            skipped += 1
            logger.warning("step {}: skipped update, grad_norm {}", step + 1,
                           norm)
        opt.zero_grad(set_to_none=True)
        sched.step()
        lr = sched.get_last_lr()[0]
        acc.log({**{f"train/{k}": v for k, v in mean.items()},
                 "train/grad_norm": norm, "train/lr": lr}, step=step + 1)
        if (step + 1) % 5 == 0 or step == 0:
            el = time.time() - start
            logger.info(f"step {step + 1}/{total} tree {mean['tree']:.4f} "
                f"mass {mean['mass']:.5f} out {mean['out']:.4f} "
                f"pos {mean['pos']:.4f} grad_norm {norm:.3f} lr {lr:.2e}"
                f" skipped {skipped} elapsed {el / 60:.1f}m"
                f" eta {el / (step + 1) * (total - step - 1) / 60:.0f}m")
        if args.eval_every and (step + 1) % args.eval_every == 0 \
                and step + 1 < total:
            save_adapter(acc, model,
                         out / "checkpoints" / f"step-{step + 1:04d}")
            run_eval(step + 1, f"step{step + 1:04d}")
    save_adapter(acc, model, out / "adapter")
    run_eval(total, "final")
    logger.info("done")
    acc.end_training()


if __name__ == "__main__":
    main()
