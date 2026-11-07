"""Evaluation of an encoder (MLM) decision model: the eval split, or any rows.

Usage (a run directory written by train.train_mlm):
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m llm_jev.evaluation.evaluate_mlm \
      --run runs/ultrabert_380m --data data/splits/reason_long_nimble --tag final

Metrics match evaluation.evaluate for the decoder, so the two compare
directly: per dataset accuracy and log_p_gold (log q of the gold option over
the K options), and "ALL" = the mean over datasets. Text-mode and legal-mass
metrics have no encoder counterpart. The run's own config.yaml gives the
prompt format, so a model is always scored the way it was trained.
"""

import argparse
import collections
import pathlib
import time
from typing import Any

from loguru import logger
import torch

from llm_jev.dataloader import mlm
from llm_jev.dataloader import schema
from llm_jev.dataloader import storage
from llm_jev.evaluation import evaluate as decoder_eval
from llm_jev.modeling import encoder
from llm_jev.modeling import encoder_varlen
from llm_jev.train import mlm_config


def load_run(run: str | pathlib.Path, device: str = "cuda",
             dtype: torch.dtype = torch.float32, attention: str | None = None):
    """(scorer, tokenizer, format) of a trained run: <run>/model and <run>/config.yaml.

    fp32 weights (scored under bf16 autocast) by default, as in training: bf16
    weights also round the residual stream, which over long documents flips
    borderline decisions (MuSiQue moved 8 points on 100 rows). `attention`
    overrides the run's model.attention (e.g. encoder.VARLEN to score in packs).
    """
    from transformers import AutoTokenizer

    run = pathlib.Path(run)
    cfg = mlm_config.resolve(mlm_config.load(run / "config.yaml"))
    tok = AutoTokenizer.from_pretrained(run / "model")
    model = encoder.load(str(run / "model"), attention=attention or cfg.model.attention,
                         device=device, dtype=dtype)
    return encoder.Scorer(model, tok, cfg.format).eval(), tok, cfg.format


def is_packed(scorer) -> bool:
    """Whether the scorer's encoder runs FA2 varlen, so rows can be packed."""
    return scorer.model.config._attn_implementation == encoder_varlen.NAME


@torch.no_grad()
def records(scorer, tok: Any, fmt: mlm.Format, rows: list[schema.Example],
            device: str = "cuda", log_every: float = 60.0,
            budget: int = 32768) -> list[tuple]:
    """(source, correct, log q(gold), top option) per row, options in release order.

    On FA2 varlen (is_packed) rows are scored in packs of up to `budget`
    tokens (encoder_varlen.groups); otherwise one row per forward.
    """
    encoded = [mlm.encode(r, tok, fmt) for r in rows]
    batches = (encoder_varlen.groups([len(e["input_ids"]) for e in encoded], budget)
               if is_packed(scorer) else [[i] for i in range(len(rows))])
    scores: list = [None] * len(rows)
    done, start, last = 0, time.time(), time.time()
    for batch in batches:
        with torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
            if is_packed(scorer):
                out = encoder_varlen.score_packed(
                    scorer, encoder_varlen.pack([encoded[i] for i in batch], device))
            else:
                out = [encoder.score(scorer, encoded[batch[0]], device)]
        for i, s in zip(batch, out):
            scores[i] = s
        done += len(batch)
        if time.time() - last > log_every or done == len(rows):
            last, el = time.time(), time.time() - start
            logger.info("scored {}/{} rows, {:.1f}m, eta {:.1f}m", done, len(rows),
                        el / 60, el / done * (len(rows) - done) / 60)
    out = []
    for r, ex, s in zip(rows, encoded, scores):
        logq = torch.log_softmax(s, 0)
        out.append((r.source, float(s.argmax() == ex["gold"]),
                    logq[ex["gold"]].item(), int(s.argmax())))
    return out


def summarize(recs: list[tuple]) -> dict[str, dict[str, float]]:
    """Per-dataset accuracy, log_p_gold and rows; "ALL" averages the datasets."""
    by: dict[str, list] = collections.defaultdict(list)
    for source, correct, logq, _ in recs:
        by[source].append((correct, logq))
    metrics = {k: {"accuracy": sum(c for c, _ in v) / len(v),
                   "log_p_gold": sum(l for _, l in v) / len(v), "rows": len(v)}
               for k, v in sorted(by.items())}
    metrics["ALL"] = {m: sum(v[m] for v in metrics.values()) / len(metrics)
                      for m in ("accuracy", "log_p_gold")}
    return metrics


def evaluate(scorer, tok: Any, fmt: mlm.Format, rows: list[schema.Example],
             device: str = "cuda") -> dict[str, dict[str, float]]:
    """summarize(records(...)): the metrics of `rows`."""
    return summarize(records(scorer, tok, fmt, rows, device))


def main() -> None:
    """Evaluates a trained encoder run on a split."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="train_mlm run directory")
    parser.add_argument("--data", default=storage.DEFAULT_SOURCE,
                        help="local folder / .parquet, or a Hub dataset id")
    parser.add_argument("--split", default="eval")
    parser.add_argument("--out", help="output directory (default: --run)")
    parser.add_argument("--tag", default="final")
    parser.add_argument("--eval-per", type=int, default=100)
    parser.add_argument("--seed", type=int, default=decoder_eval.SEED)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32",
                        help="weights (scoring runs under bf16 autocast either way)")
    parser.add_argument("--attention", help="override the run's model.attention, "
                        "e.g. varlen to score packed rows with FlashAttention-2")
    args = parser.parse_args()

    out = pathlib.Path(args.out or args.run)
    out.mkdir(parents=True, exist_ok=True)
    rows = decoder_eval.subset(storage.load_examples(args.data, args.split),
                               args.eval_per, args.seed)
    scorer, tok, fmt = load_run(args.run, dtype=getattr(torch, args.dtype), attention=args.attention)
    metrics = evaluate(scorer, tok, fmt, rows)
    decoder_eval.save(out, args.tag, metrics, [])
    decoder_eval.log_summary(args.tag, metrics)


if __name__ == "__main__":
    main()
