"""Exports the loss-term ablation runs and their TensorBoard curves for the blog.

Reads each run under runs/ablation/ (args.json, eval_*.json, eval_benchmarks_final.json, tb/)
and the base model's benchmark scores, and writes blogs/data/ablation.json. Runs that are
not finished yet are exported with what exists so far.

Usage:
  .venv/bin/python blogs/export_ablation.py
"""

import argparse
import json
import pathlib

from tensorboard.backend.event_processing import event_accumulator

ROOT = pathlib.Path("runs/ablation")
BASE_BENCHMARKS = pathlib.Path("runs/qwen3_0.6b/eval_benchmarks_base.json")
# (run, label, anchors); the order is the order of the blog's tables.
RUNS = [
    ("tree_only", "L_tree only", []),
    ("mass_only", "+ L_mass", ["mass"]),
    ("out_only", "+ L_out", ["out"]),
    ("pos_only", "+ L_pos", ["pos"]),
    ("no_mass", "full − L_mass", ["out", "pos"]),
    ("no_out", "full − L_out", ["mass", "pos"]),
    ("no_pos", "full − L_pos", ["mass", "out"]),
    ("full", "full loss", ["mass", "out", "pos"]),
]
CURVES = ["tree", "mass", "out", "pos", "grad_norm"]
EVAL = ["accuracy", "log_p_gold", "root_legal_mass", "gen_valid", "gen_correct"]


def read_json(path: pathlib.Path):
    return json.loads(path.read_text()) if path.exists() else None


def table(data: dict | None, keys: list[str]) -> dict | None:
    """The chosen metrics of each row of an eval table."""
    if data is None:
        return None
    return {name: {k: row[k] for k in keys if k in row} for name, row in data.items()}


def curves(folder: pathlib.Path) -> dict:
    """Training curves as [step, value] lists."""
    events = event_accumulator.EventAccumulator(str(folder / "tb"), size_guidance={"scalars": 0})
    events.Reload()
    tags = events.Tags()["scalars"]
    return {k: [[s.step, s.value] for s in events.Scalars(f"train/{k}")] for k in CURVES if f"train/{k}" in tags}


def export(run: str, label: str, anchors: list[str]) -> dict:
    folder = ROOT / run
    args = read_json(folder / "args.json")
    train = curves(folder) if (folder / "tb").exists() else {}
    final = table(read_json(folder / "eval_final.json"), EVAL)
    return {
        "id": run,
        "label": label,
        "anchors": anchors,
        "started": args is not None,
        "steps_done": train["tree"][-1][0] if train.get("tree") else 0,
        "final": final,
        "bench_final": table(read_json(folder / "eval_benchmarks_final.json"), ["accuracy"]),
        "curves": train,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="blogs/data/ablation.json")
    args = parser.parse_args()
    runs = [export(*r) for r in RUNS]
    full_args = read_json(ROOT / "full" / "args.json") or {}
    steps = max((r["steps_done"] for r in runs), default=0)
    data = {
        "model": full_args.get("model", "Qwen/Qwen3-0.6B"),
        "lam": full_args.get("lam", 1.0),
        "batch_size": full_args.get("batch_size", 32),
        "lr": full_args.get("lr", 1e-4),
        "lora_rank": full_args.get("lora_rank", 16),
        "eval_per": full_args.get("eval_per", 100),
        "total_steps": 406,
        "steps": steps,
        "base": table(read_json(ROOT / "full" / "eval_base.json"), EVAL),
        "bench_base": table(read_json(BASE_BENCHMARKS), ["accuracy"]),
        "runs": runs,
    }
    out = pathlib.Path(args.out)
    out.write_text(json.dumps(data))
    done = sum(r["final"] is not None for r in runs)
    print(f"{len(runs)} runs ({done} evaluated) -> {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
