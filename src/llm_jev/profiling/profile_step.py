"""Profiles training steps on one GPU: torch.profiler traces and CUDA memory snapshots.

Usage (see `make torch-profiler` and docs/profiling.md, which explains every output):
  .venv/bin/python -m llm_jev.profiling.profile_step --model Qwen/Qwen3-0.6B \
      --out runs/profile/Qwen3-0.6B

A step is the training step of llm_jev.train.train without Accelerate:
--batch-size rows, each one forward with the adapter (p_theta), one without
(p0), the loss and a backward; then gradient clipping and AdamW. Steps run in
three stages:

  1. warmup (--warmup-steps): torch.compile of the loss, cuBLAS handles and the
     caching allocator's first segments, which would distort the profile;
  2. memory (--memory-steps): the CUDA caching allocator records every
     allocation and free with its Python stack -> memory_snapshot.pickle;
     each phase's allocated / peak memory -> memory_phases.json;
  3. profile (--profile-steps): torch.profiler records CPU ops, CUDA kernels,
     shapes, memory and stacks -> trace.json.gz, op tables, memory timeline.

The memory stage runs before the warmup when --cold-memory is set, so the
snapshot also shows the first step (AdamW creating its state, compile).
"""

import argparse
import contextlib
import json
import pathlib
import pickle
import random
import time

from loguru import logger
import torch
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity
from torch.profiler import record_function

from llm_jev.dataloader import answers
from llm_jev.dataloader import storage
from llm_jev.modeling import modeling
from llm_jev.train import losses

MIB = 2**20
# record_function names, in step order: this file's phases, with the ones in
# losses.compute nested inside forward_and_loss.
PHASES = ("forward_and_loss", "p_theta_forward", "p0_forward", "loss_terms",
          "backward", "clip_grad_norm", "optimizer_step", "zero_grad")


def pick_rows(train: list, tok, args) -> list[dict]:
    """Encoded training rows of at most --max-tokens tokens, sampled with --seed."""
    rng = random.Random(args.seed)
    rows = []
    for i in rng.sample(range(len(train)), len(train)):
        ex = answers.encode(train[i], tok, random.Random(f"{args.seed}:{i}"))
        if len(ex["input_ids"]) <= args.max_tokens:
            rows.append(ex)
        if len(rows) == args.batch_size * args.steps_total:
            return rows
    raise ValueError(f"only {len(rows)} rows of at most {args.max_tokens} tokens")


class Phases:
    """Names each phase for the profiler and, when on, measures its memory.

    torch.cuda.memory_allocated is the caching allocator's host-side count of
    live tensor bytes, so it needs no synchronize; max_memory_allocated is the
    high-water mark since the last reset_peak_memory_stats.
    """

    def __init__(self):
        self.measure = False
        self.rows: list[dict] = []

    @contextlib.contextmanager
    def __call__(self, name: str, row: int | None = None):
        if not self.measure:
            with record_function(name):
                yield
            return
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        with record_function(name):
            yield
        self.rows.append({
            "phase": name, "row": row,
            "allocated_before_mib": before / MIB,
            "allocated_after_mib": torch.cuda.memory_allocated() / MIB,
            "peak_mib": torch.cuda.max_memory_allocated() / MIB,
            "peak_above_before_mib": (torch.cuda.max_memory_allocated() - before) / MIB,
            "reserved_mib": torch.cuda.memory_reserved() / MIB,
        })


def train_step(model, opt, batch: list[dict], device, pad: int, phase: Phases,
               args) -> float:
    """One optimizer step over `batch`; returns the mean loss."""
    total = 0.0
    for j, ex in enumerate(batch):
        with phase("forward_and_loss", j):
            L = losses.compute(model, ex, device, pad)
            loss = L["tree"] + args.lam * (L["mass"] + L["out"] + L["pos"])
        with phase("backward", j):
            (loss / len(batch)).backward()
        total += loss.item() / len(batch)
    with phase("clip_grad_norm"):
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
    with phase("optimizer_step"):
        opt.step()
    with phase("zero_grad"):
        opt.zero_grad(set_to_none=True)
    return total


def save_tables(prof, out: pathlib.Path, rows: int) -> None:
    """The profiler's aggregate tables, as text."""
    averages = prof.key_averages()
    tables = {
        "ops_by_cuda_time.txt": averages.table(
            sort_by="self_cuda_time_total", row_limit=rows),
        "ops_by_cpu_time.txt": averages.table(
            sort_by="self_cpu_time_total", row_limit=rows),
        "ops_by_cuda_memory.txt": averages.table(
            sort_by="self_cuda_memory_usage", row_limit=rows),
        "ops_by_input_shape.txt": prof.key_averages(group_by_input_shape=True)
            .table(sort_by="self_cuda_time_total", row_limit=rows),
        "ops_by_stack.txt": prof.key_averages(group_by_stack_n=8)
            .table(sort_by="self_cuda_time_total", row_limit=rows),
    }
    for name, table in tables.items():
        (out / name).write_text(table)
    # Each record_function appears twice: the CPU range (host time from enter
    # to exit) and a GPU range (first to last kernel launched inside it). The
    # GPU range of backward is empty: autograd runs it on its own thread.
    cpu = {e.key: e for e in averages if e.device_type == DeviceType.CPU}
    gpu = {e.key: e for e in averages if e.device_type != DeviceType.CPU}
    lines = [f"{'phase':<20}{'calls':>7}{'cpu range ms':>14}{'gpu range ms':>14}"]
    for name in ("ProfilerStep*",) + PHASES:
        if name in cpu:
            g = gpu[name].device_time_total / 1e3 if name in gpu else 0.0
            lines.append(f"{name:<20}{cpu[name].count:>7}"
                         f"{cpu[name].cpu_time_total / 1e3:>14.1f}{g:>14.1f}")
    (out / "phases.txt").write_text("\n".join(lines) + "\n")


def save_snapshot(out: pathlib.Path) -> None:
    """The allocator's recorded history as a pickle and two offline HTML views."""
    from torch.cuda import _memory_viz

    snapshot = torch.cuda.memory._snapshot()
    with open(out / "memory_snapshot.pickle", "wb") as f:
        pickle.dump(snapshot, f)
    (out / "memory_trace.html").write_text(_memory_viz.trace_plot(snapshot))
    (out / "memory_segments.html").write_text(_memory_viz.segment_plot(snapshot))


def main() -> None:
    """Runs the three stages and writes every output to --out."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--data", default=storage.DEFAULT_SOURCE,
                        help="local folder, or a Hub dataset id")
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=4,
                        help="rows per optimizer step")
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="rows over this many tokens are skipped")
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--memory-steps", type=int, default=1)
    parser.add_argument("--profile-steps", type=int, default=1)
    parser.add_argument("--cold-memory", action="store_true",
                        help="record memory from the first step, before warmup")
    parser.add_argument("--no-checkpointing", action="store_true",
                        help="keep every activation (compare peak memory)")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--table-rows", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args()
    # + 1: the profiler's discarded warmup step.
    args.steps_total = (args.warmup_steps + args.memory_steps
                        + 1 + args.profile_steps)

    from transformers import AutoTokenizer

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logger.add(out / "profile.log", mode="w")
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))
    device = torch.device("cuda")
    torch.manual_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model)
    rows = pick_rows(storage.load_examples(args.data, "train"), tok, args)
    lengths = [len(ex["input_ids"]) for ex in rows]
    logger.info("{} rows, tokens per row: min {} mean {:.0f} max {}", len(rows),
                min(lengths), sum(lengths) / len(lengths), max(lengths))

    model = modeling.add_lora(modeling.load_base(args.model), args.lora_rank)
    model.config.use_cache = False
    if not args.no_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    model.to(device).train()
    weights = torch.cuda.memory_allocated()
    lora = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(lora, lr=args.lr)
    logger.info("weights on GPU: {:.0f} MiB; LoRA parameters: {:,} ({:.1f} MiB)",
                weights / MIB, sum(p.numel() for p in lora),
                sum(p.numel() * p.element_size() for p in lora) / MIB)

    phase = Phases()
    batches = iter([rows[i : i + args.batch_size]
                    for i in range(0, len(rows), args.batch_size)])

    def run(stage: str, n: int, after_step=None) -> None:
        for k in range(n):
            t0 = time.time()
            loss = train_step(model, opt, next(batches), device,
                              tok.pad_token_id, phase, args)
            torch.cuda.synchronize()
            logger.info("{} step {}/{}: loss {:.4f}, {:.2f} s, peak {:.0f} MiB",
                        stage, k + 1, n, loss, time.time() - t0,
                        torch.cuda.max_memory_allocated() / MIB)
            if after_step:
                after_step()

    def memory_stage() -> None:
        # stacks="python": Python frames only; "all" adds C++ frames (slower to dump).
        torch.cuda.memory._record_memory_history(max_entries=500_000,
                                                 stacks="python")
        phase.measure = True
        run("memory", args.memory_steps)
        phase.measure = False
        save_snapshot(out)
        torch.cuda.memory._record_memory_history(enabled=None)
        (out / "memory_phases.json").write_text(json.dumps(
            {"weights_mib": weights / MIB, "phases": phase.rows}, indent=2))
        stats = torch.cuda.memory_stats()
        (out / "memory_stats.json").write_text(json.dumps(stats, indent=2))
        (out / "memory_summary.txt").write_text(torch.cuda.memory_summary())

    if args.cold_memory:
        memory_stage()
        run("warmup", args.warmup_steps)
    else:
        run("warmup", args.warmup_steps)
        memory_stage()

    def export(prof) -> None:
        """Called by the profiler when the active steps end."""
        logger.info("writing the profiler outputs")
        prof.export_chrome_trace(str(out / "trace.json.gz"))
        save_tables(prof, out, args.table_rows)
        prof.export_stacks(str(out / "stacks_cuda.txt"), "self_cuda_time_total")
        logger.info("writing memory_timeline.html (slow: replays every allocation)")
        prof.export_memory_timeline(str(out / "memory_timeline.html"),
                                    device="cuda:0")

    torch.cuda.reset_peak_memory_stats()
    with torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            # One traced but discarded step (the profiler's own start-up cost),
            # then --profile-steps recorded steps, once (repeat=1).
            schedule=torch.profiler.schedule(wait=0, warmup=1,
                                             active=args.profile_steps,
                                             repeat=1),
            on_trace_ready=export,
            record_shapes=True, profile_memory=True, with_stack=True,
            with_flops=True,
            # verbose: keep the Python stacks for ops_by_stack and export_stacks.
            experimental_config=torch._C._profiler._ExperimentalConfig(
                verbose=True)) as prof:
        run("profile", 1 + args.profile_steps, after_step=prof.step)
    logger.info("done: {}", out)


if __name__ == "__main__":
    main()
