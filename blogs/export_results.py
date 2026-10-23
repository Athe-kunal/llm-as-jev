"""Exports the three trained runs' results and TensorBoard curves for the blog.

Reads each run's args.json, eval_*.json, eval_benchmarks_*.json and tb/ events
under runs/, and writes blogs/data/results.json.

Usage:
  .venv/bin/python blogs/export_results.py
"""

import argparse
import json
import pathlib

from tensorboard.backend.event_processing import event_accumulator

RUNS = pathlib.Path("runs")
MODELS = [
    ("qwen3_0.6b", "Qwen3-0.6B"),
    ("qwen3_4b", "Qwen3-4B"),
    ("qwen3.5_4b", "Qwen3.5-4B"),
]
CURVES = ["tree", "mass", "out", "pos", "grad_norm", "lr"]
METRICS = ["accuracy", "log_p_gold", "gen_correct"]


def read_json(path: pathlib.Path):
    return json.loads(path.read_text()) if path.exists() else None


def metrics(table: dict | None, keys: tuple[str, ...]) -> dict | None:
    """The chosen metrics of each row of an eval table."""
    if table is None:
        return None
    return {name: {k: row[k] for k in keys if k in row} for name, row in table.items()}


def scalars(events) -> dict:
    """Training curves and the ALL-row eval metrics, as [step, value] lists."""
    out = {"train": {}, "eval": {}}
    for name in CURVES:
        out["train"][name] = [[s.step, s.value] for s in events.Scalars(f"train/{name}")]
    for name in METRICS:
        out["eval"][name] = [[s.step, s.value] for s in events.Scalars(f"eval/ALL/{name}")]
    return out


def export(run: str, label: str) -> dict:
    folder = RUNS / run
    args = read_json(folder / "args.json")
    events = event_accumulator.EventAccumulator(str(folder / "tb"), size_guidance={"scalars": 0})
    events.Reload()
    curves = scalars(events)
    return {
        "id": run,
        "label": label,
        "model": args["model"],
        "lam": args["lam"],
        "lora_rank": args["lora_rank"],
        "batch_size": args["batch_size"],
        "lr": args["lr"],
        "steps": curves["train"]["tree"][-1][0],
        "base": metrics(read_json(folder / "eval_base.json"), tuple(METRICS)),
        "final": metrics(read_json(folder / "eval_final.json"), tuple(METRICS)),
        "bench_base": metrics(read_json(folder / "eval_benchmarks_base.json"), ("accuracy",)),
        "bench_final": metrics(read_json(folder / "eval_benchmarks_final.json"), ("accuracy",)),
        "curves": curves,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="blogs/data/results.json")
    args = parser.parse_args()
    models = [export(run, label) for run, label in MODELS]
    out = pathlib.Path(args.out)
    out.write_text(json.dumps({"models": models}))
    print(f"{len(models)} runs -> {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
