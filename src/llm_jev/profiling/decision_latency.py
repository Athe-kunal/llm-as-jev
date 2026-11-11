"""Time per decision: decoders (trie scoring) against encoder runs (one pass).

Usage:
  CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m llm_jev.profiling.decision_latency \
      --decoders Athekunal/llm-jev-qwen3-0.6b Athekunal/llm-jev-qwen3-4b \
      --encoders runs/ultrabert_380m_lr1.5e-5 --out runs/latency.json

Every model scores the same rows (evaluation.evaluate.subset of the eval
split). Modes:
  - latency: one row at a time (serving a single decision). Decoders run
    answers.score (prefill + the trie's deeper nodes); encoders one forward.
  - throughput: the batched path evaluation uses: answers.score_examples for
    decoders, packed FA2 varlen (evaluate_mlm.records) for encoders on varlen.
Encoders are timed with fp32 weights under bf16 autocast (how they are
evaluated) and with bf16 weights. The first --warmup rows are not timed; the
GPU is synchronized around every timed call. Run it on an otherwise idle GPU.

Memory (GiB, torch.cuda allocator): "weights" is what is allocated once the
model is loaded; "peak" is the most allocated during a call (weights included),
taken per row in latency mode (max, and mean per length bucket) and over the
whole batched pass in throughput mode. For decoders, "prefill" repeats the
latency pass with the input read once and only the last position's logits kept
(no trie steps, which copy the prompt's KV cache once per option-number node).
"tree" times answers.score_tree: the same scores, with the whole trie read in
one extra forward on the shared prompt cache under a tree mask (no copies).
"""

import argparse
import gc
import json
import pathlib
import statistics
import time
from collections.abc import Callable

from loguru import logger
import torch

from llm_jev.dataloader import answers
from llm_jev.dataloader import mlm
from llm_jev.dataloader import storage
from llm_jev.evaluation import evaluate
from llm_jev.evaluation import evaluate_mlm
from llm_jev.modeling import encoder
from llm_jev.modeling import modeling

BUCKETS = ((0, 1024), (1024, 4096), (4096, 1 << 30))  # prompt tokens


GIB = 1 << 30
PEAKS: list[float] = []  # peak GiB of each timed() call, in order


def timed(fn: Callable[[], object]) -> float:
    """Seconds for fn() on the GPU; its peak allocated memory goes to PEAKS."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    secs = time.perf_counter() - t0
    PEAKS.append(torch.cuda.max_memory_allocated() / GIB)
    return secs


def timed_rows(fns: list[Callable[[], object]]) -> tuple[list[float], list[float]]:
    """(seconds, peak GiB) per call."""
    start = len(PEAKS)
    secs = [timed(f) for f in fns]
    return secs, PEAKS[start:]


def memory(weights: float, peaks: list[float], tokens: list[int]) -> dict:
    """Weights, max and mean peak, and mean peak per length bucket (GiB)."""
    out = {"weights_gib": weights, "peak_max_gib": max(peaks),
           "peak_mean_gib": statistics.fmean(peaks)}
    for lo, hi in BUCKETS:
        sel = [m for m, n in zip(peaks, tokens) if lo <= n < hi]
        out[f"peak_gib_tokens_{lo}-{hi if hi < 1 << 30 else 'inf'}"] = statistics.fmean(sel) if sel else None
    return out


def summary(seconds: list[float], tokens: list[int]) -> dict:
    """Mean / median / p90 ms per row, rows per second, and mean ms per length bucket."""
    ms = sorted(s * 1000 for s in seconds)
    out = {"rows": len(ms), "mean_ms": statistics.fmean(ms), "median_ms": statistics.median(ms),
           "p90_ms": ms[int(0.9 * (len(ms) - 1))], "rows_per_s": len(ms) / sum(seconds),
           "mean_tokens": statistics.fmean(tokens)}
    for lo, hi in BUCKETS:
        sel = [s * 1000 for s, n in zip(seconds, tokens) if lo <= n < hi]
        key = f"ms_tokens_{lo}-{hi if hi < 1 << 30 else 'inf'}"
        out[key] = {"rows": len(sel), "mean_ms": statistics.fmean(sel) if sel else None}
    return out


@torch.no_grad()
def time_decoder(model_id: str, rows: list, warmup: int) -> dict:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = modeling.load(model_id)
    weights = torch.cuda.memory_allocated() / GIB
    prompts = [answers.prompt_ids(r, tok) for r in rows]
    for p, r in zip(prompts[:warmup], rows[:warmup]):
        answers.score(model, p, tok, len(r.options), "cuda")
    secs, peaks = timed_rows([lambda p=p, r=r: answers.score(model, p, tok, len(r.options), "cuda")
                              for p, r in zip(prompts, rows)])
    batched = timed(lambda: answers.score_examples(model, rows, tok, "cuda"))
    tokens = [len(p) for p in prompts]

    def prefill(p):
        model(input_ids=torch.tensor([p], device="cuda"), use_cache=True, logits_to_keep=1)

    pre_secs, pre_peaks = timed_rows([lambda p=p: prefill(p) for p in prompts])
    tree_secs, tree_peaks = timed_rows([lambda p=p, r=r: answers.score_tree(model, p, tok, len(r.options), "cuda")
                                        for p, r in zip(prompts, rows)])
    out = {"latency": summary(secs, tokens), "memory": memory(weights, peaks, tokens),
           "prefill": {"latency": summary(pre_secs, tokens), "memory": memory(weights, pre_peaks, tokens)},
           "tree": {"latency": summary(tree_secs, tokens), "memory": memory(weights, tree_peaks, tokens)},
           "throughput": {"rows": len(rows), "seconds": batched, "rows_per_s": len(rows) / batched,
                          "peak_gib": PEAKS[-1]}}
    del model
    return out


@torch.no_grad()
def time_encoder(run: str, rows: list, warmup: int, dtype: torch.dtype,
                 attention: str | None = None) -> dict:
    """Latency one row at a time; on FA2 varlen also packed throughput (evaluate_mlm.records)."""
    scorer, tok, fmt = evaluate_mlm.load_run(run, dtype=dtype, attention=attention)
    weights = torch.cuda.memory_allocated() / GIB
    budget = evaluate_mlm.run_config(run).model.pack_tokens
    encoded = [mlm.encode(r, tok, fmt) for r in rows]

    def one(ex):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            encoder.score(scorer, ex, "cuda")

    for ex in encoded[:warmup]:
        one(ex)
    secs, peaks = timed_rows([lambda ex=ex: one(ex) for ex in encoded])
    throughput = None
    if evaluate_mlm.is_packed(scorer):
        evaluate_mlm.records(scorer, tok, fmt, rows[:warmup * 10], log_every=float("inf"),
                             budget=budget)  # warm-up
        packed = timed(lambda: evaluate_mlm.records(scorer, tok, fmt, rows, log_every=float("inf"),
                                                    budget=budget))
        throughput = {"rows": len(rows), "seconds": packed, "rows_per_s": len(rows) / packed,
                      "peak_gib": PEAKS[-1], "budget_tokens": budget}
    tokens = [len(ex["input_ids"]) for ex in encoded]
    del scorer
    return {"latency": summary(secs, tokens), "memory": memory(weights, peaks, tokens),
            "throughput": throughput}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--decoders", nargs="*", default=[], help="Hub ids or paths (merged weights)")
    parser.add_argument("--encoders", nargs="*", default=[], help="train_mlm run directories")
    parser.add_argument("--data", default="data/splits/reason_long_nimble")
    parser.add_argument("--split", default="eval")
    parser.add_argument("--eval-per", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--out", required=True)
    parser.add_argument("--merge", action="store_true",
                        help="keep the other models already in --out (re-time only the ones given)")
    args = parser.parse_args()

    rows = evaluate.subset(storage.load_examples(args.data, args.split), args.eval_per, evaluate.SEED)
    logger.info("{} rows on {}", len(rows), torch.cuda.get_device_name())
    results = {"gpu": torch.cuda.get_device_name(), "rows": len(rows), "models": {}}
    if args.merge and pathlib.Path(args.out).exists():
        results["models"] = json.loads(pathlib.Path(args.out).read_text())["models"]
    for m in args.decoders:
        results["models"][m] = time_decoder(m, rows, args.warmup)
        gc.collect(); torch.cuda.empty_cache()  # the model went out of scope
        logger.info("{}: {}", m, json.dumps(results["models"][m]))
    for run in args.encoders:
        for name, dtype, attention in (("fp32 weights", torch.float32, "sdpa"),
                                       ("bf16 weights", torch.bfloat16, "sdpa"),
                                       ("fp32 weights, FA2 varlen", torch.float32, encoder.VARLEN)):
            key = f"{run} ({name})"
            results["models"][key] = time_encoder(run, rows, args.warmup, dtype, attention)
            gc.collect(); torch.cuda.empty_cache()
            logger.info("{}: {}", key, json.dumps(results["models"][key]))
    pathlib.Path(args.out).write_text(json.dumps(results, indent=2))
    logger.info("wrote {}", args.out)


if __name__ == "__main__":
    main()
