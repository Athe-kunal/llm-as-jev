"""Time and memory per decision: decoders (trie scoring) against encoders.

Usage:
  python -m llm_jev.analysis.latency --config configs/analysis/latency.yaml

Every model scores the same rows (the dev sample of the eval split):
- latency: one row at a time (serving a single decision). Decoders run
  the cache-copying trie scorer; "tree" runs `decoder.score_tree` (the whole
  trie in one extra forward on the shared prompt cache) and "prefill" only
  reads the prompt. Encoders run one forward.
- throughput: the batched path evaluation uses.
Encoders are timed with fp32 weights under bf16 autocast (how they are
evaluated), with bf16 weights, and packed on FA2 varlen. The first `warmup`
rows are not timed; the GPU is synchronised around every timed call.

Memory (GiB, torch.cuda allocator): "weights" is allocated once the model is
loaded; "peak" is the most allocated during a call (weights included), per
row (max, and mean per length bucket) and over the whole batched pass.
Writes `out` (the paper reads it as runs/profile.json).
"""

from collections.abc import Callable
from collections.abc import Sequence
import gc
import json
import pathlib
import statistics
import time
from typing import Any

from loguru import logger
import torch

from llm_jev import config
from llm_jev.data import sampling
from llm_jev.data import schema
from llm_jev.data import storage
from llm_jev.models import decoder
from llm_jev.models import encoder

GIB = 1 << 30
BUCKETS = ((0, 1024), (1024, 4096), (4096, 1 << 30))  # Prompt tokens.


def _bucket_name(lo: int, hi: int) -> str:
    return f"{lo}-{hi if hi < 1 << 30 else 'inf'}"


def timed(fn: Callable[[], object]) -> tuple[float, float]:
    """(seconds, peak GiB allocated) of fn() on the GPU."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return time.perf_counter() - start, torch.cuda.max_memory_allocated() / GIB


def timed_rows(
    fns: Sequence[Callable[[], object]],
) -> tuple[list[float], list[float]]:
    """(seconds, peak GiB) per call."""
    pairs = [timed(f) for f in fns]
    return [s for s, _ in pairs], [m for _, m in pairs]


def memory(
    weights: float, peaks: Sequence[float], tokens: Sequence[int]
) -> dict[str, Any]:
    """Weights, max and mean peak, and mean peak per length bucket (GiB)."""
    out: dict[str, Any] = {
        "weights_gib": weights,
        "peak_max_gib": max(peaks),
        "peak_mean_gib": statistics.fmean(peaks),
    }
    for lo, hi in BUCKETS:
        sel = [m for m, n in zip(peaks, tokens, strict=True) if lo <= n < hi]
        out[f"peak_gib_tokens_{_bucket_name(lo, hi)}"] = (
            statistics.fmean(sel) if sel else None
        )
    return out


def summary(seconds: Sequence[float], tokens: Sequence[int]) -> dict[str, Any]:
    """Mean / median / p90 ms per row, rows per second, ms per bucket."""
    ms = sorted(s * 1000 for s in seconds)
    out: dict[str, Any] = {
        "rows": len(ms),
        "mean_ms": statistics.fmean(ms),
        "median_ms": statistics.median(ms),
        "p90_ms": ms[int(0.9 * (len(ms) - 1))],
        "rows_per_s": len(ms) / sum(seconds),
        "mean_tokens": statistics.fmean(tokens),
    }
    for lo, hi in BUCKETS:
        sel = [
            s * 1000
            for s, n in zip(seconds, tokens, strict=True)
            if lo <= n < hi
        ]
        out[f"ms_tokens_{_bucket_name(lo, hi)}"] = {
            "rows": len(sel),
            "mean_ms": statistics.fmean(sel) if sel else None,
        }
    return out


def _mode(
    fns: Sequence[Callable[[], object]], weights: float, tokens: Sequence[int]
) -> dict[str, Any]:
    secs, peaks = timed_rows(fns)
    return {
        "latency": summary(secs, tokens),
        "memory": memory(weights, peaks, tokens),
    }


@torch.no_grad()
def time_decoder(
    model_id: str, rows: Sequence[schema.Example], warmup: int
) -> dict[str, Any]:
    """Copy, prefill-only and tree latency, and batched throughput."""
    model = decoder.TrieDecoder.for_inference(
        config.DecoderModelConfig(id=model_id), "cuda"
    )
    net, tok = model.network, model.tokenizer
    weights = torch.cuda.memory_allocated() / GIB
    prompts = [decoder.prompt_ids(r, tok) for r in rows]
    ks = [len(r.options) for r in rows]
    tokens = [len(p) for p in prompts]

    def copy(p: list[int], k: int) -> object:
        return decoder.score_batch(net, [p], tok, [k], "cuda")

    def prefill(p: list[int]) -> object:
        ids = torch.tensor([p], device="cuda")
        return net(input_ids=ids, use_cache=True, logits_to_keep=1)

    def tree(p: list[int], k: int) -> object:
        return decoder.score_tree(net, p, tok, k, "cuda")

    for p, k in zip(prompts[:warmup], ks[:warmup], strict=True):
        copy(p, k)
    out = _mode(
        [lambda p=p, k=k: copy(p, k) for p, k in zip(prompts, ks, strict=True)],
        weights,
        tokens,
    )
    batched, peak = timed(lambda: model.score(rows))
    out["throughput"] = {
        "rows": len(rows),
        "seconds": batched,
        "rows_per_s": len(rows) / batched,
        "peak_gib": peak,
    }
    out["prefill"] = _mode(
        [lambda p=p: prefill(p) for p in prompts], weights, tokens
    )
    out["tree"] = _mode(
        [lambda p=p, k=k: tree(p, k) for p, k in zip(prompts, ks, strict=True)],
        weights,
        tokens,
    )
    return out


@torch.no_grad()
def time_encoder(
    run: str,
    rows: Sequence[schema.Example],
    warmup: int,
    dtype: torch.dtype,
    attention: str,
) -> dict[str, Any]:
    """Latency one row at a time; on FA2 varlen also packed throughput."""
    model = encoder.MaskedEncoder.from_run(run, "cuda", dtype, attention)
    weights = torch.cuda.memory_allocated() / GIB
    encoded = [model.encode(r) for r in rows]
    tokens = [len(e.input_ids) for e in encoded]

    def one(ex: encoder.MaskEncoding) -> object:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return model.network(
                torch.tensor([ex.input_ids], device="cuda"),
                torch.tensor(ex.masks, device="cuda"),
            )

    for ex in encoded[:warmup]:
        one(ex)
    out = _mode([lambda ex=ex: one(ex) for ex in encoded], weights, tokens)
    out["throughput"] = None
    if model.is_packed:
        model.score(rows[: warmup * 10])
        packed, peak = timed(lambda: model.score(rows))
        out["throughput"] = {
            "rows": len(rows),
            "seconds": packed,
            "rows_per_s": len(rows) / packed,
            "peak_gib": peak,
            "budget_tokens": model.pack_tokens,
        }
    return out


def main() -> None:
    """Times every model and writes the JSON."""
    cfg, _ = config.from_command_line(config.LatencyConfig, __doc__)
    rows = sampling.per_dataset(
        storage.load_examples(cfg.source, cfg.split), cfg.eval_per, config.SEED
    )
    gpu = torch.cuda.get_device_name()
    logger.info("{} rows on {}", len(rows), gpu)
    out = pathlib.Path(cfg.out)
    results: dict[str, Any] = {"gpu": gpu, "rows": len(rows), "models": {}}
    if cfg.merge and out.exists():
        results["models"] = json.loads(out.read_text())["models"]
    jobs: list[tuple[str, Callable[[], dict[str, Any]]]] = [
        (m, lambda m=m: time_decoder(m, rows, cfg.warmup)) for m in cfg.decoders
    ]
    for run in cfg.encoders:
        for name, dtype, attention in (
            ("fp32 weights", torch.float32, "sdpa"),
            ("bf16 weights", torch.bfloat16, "sdpa"),
            ("fp32 weights, FA2 varlen", torch.float32, config.VARLEN),
        ):
            jobs.append(
                (
                    f"{run} ({name})",
                    lambda r=run, d=dtype, a=attention: time_encoder(
                        r, rows, cfg.warmup, d, a
                    ),
                )
            )
    for key, job in jobs:
        results["models"][key] = job()
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("{}: {}", key, json.dumps(results["models"][key]))
    out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
