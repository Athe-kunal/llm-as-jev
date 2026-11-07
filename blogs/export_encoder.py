"""Exports the encoder (UltraBERT) runs and the Qwen baselines they are compared with.

Reads, under runs/:
  - every encoder run matching ENCODER_GLOB (train.train_mlm): config.yaml,
    eval_base.json, eval_final.json, eval_benchmarks_final.json and tb/;
  - each baseline in BASELINES: eval_final.json, eval_benchmarks_final.json
    (the released checkpoints, scored on the same rows).
and writes blogs/data/encoder.json. ALL is recomputed here as the mean over
the datasets every model has, so it means the same thing for every model.

Usage:
  .venv/bin/python blogs/export_encoder.py
"""

import argparse
import json
import pathlib
import statistics

from omegaconf import OmegaConf
from tensorboard.backend.event_processing import event_accumulator

RUNS = pathlib.Path("runs")
ENCODER_GLOB = "ultrabert_380m*"
SKIP = {"smoke"}  # run-name fragments to leave out
BASELINES = [
    ("qwen3_0.6b_released", "Qwen3-0.6B", "Athekunal/llm-jev-qwen3-0.6b", 0.6e9),
    ("qwen3_4b_released", "Qwen3-4B", "Athekunal/llm-jev-qwen3-4b", 4.0e9),
]
CURVES = ["loss", "accuracy", "grad_norm", "lr"]
EXTERNAL = ("boolq", "mmlu", "mmlu_pro", "arc_challenge", "winogrande", "sciq")
FEATURED_EPOCHS = 1  # the decoders were trained for one epoch
DEFAULT_SEED = 20260922  # run.seed of the sweep; other seeds are replicates of a setting
LATENCY = RUNS / "latency.json"  # profiling.decision_latency output (optional)


def read_json(path: pathlib.Path):
    return json.loads(path.read_text()) if path.exists() else None


def accuracies(table: dict | None) -> dict | None:
    """Per-dataset accuracy (aggregate rows dropped; they are recomputed)."""
    if table is None:
        return None
    return {k: v["accuracy"] for k, v in table.items()
            if k not in ("ALL", "MACRO", "external_macro") and "accuracy" in v}


def with_totals(acc: dict | None, datasets: list[str], bench: bool = False) -> dict | None:
    """acc plus ALL over `datasets` (and external_macro for benchmarks)."""
    if acc is None or not datasets or not all(d in acc for d in datasets):
        return acc
    out = dict(acc, ALL=sum(acc[d] for d in datasets) / len(datasets))
    if bench and all(d in acc for d in EXTERNAL):
        out["external_macro"] = sum(acc[d] for d in EXTERNAL) / len(EXTERNAL)
    return out


def curves(folder: pathlib.Path) -> dict:
    events = event_accumulator.EventAccumulator(str(folder / "tb"),
                                                size_guidance={"scalars": 0})
    events.Reload()
    tags = set(events.Tags()["scalars"])
    return {k: [[s.step, s.value] for s in events.Scalars(f"train/{k}")]
            for k in CURVES if f"train/{k}" in tags}


def fold_seeds(encoders: list[dict]) -> list[dict]:
    """One entry per (lr, epochs): the DEFAULT_SEED run, with every seed's totals under "seeds"."""
    groups: dict[tuple, list[dict]] = {}
    for m in encoders:
        groups.setdefault((m["lr"], m["epochs"]), []).append(m)
    out = []
    for members in groups.values():
        rep = next((m for m in members if m["seed"] == DEFAULT_SEED), members[0])
        if len(members) > 1:
            rep["seeds"] = [{"seed": m["seed"], "id": m["id"], "ALL": m["final"]["ALL"],
                             "external_macro": (m["bench"] or {}).get("external_macro")}
                            for m in sorted(members, key=lambda m: m["seed"] != DEFAULT_SEED)]
            for key in ("ALL", "external_macro"):
                vals = [x[key] for x in rep["seeds"] if x[key] is not None]
                if len(vals) > 1:
                    rep[f"seed_{key}"] = {"mean": statistics.fmean(vals), "sd": statistics.stdev(vals),
                                          "n": len(vals)}
        out.append(rep)
    return out


def latency(baselines: list[dict], encoders: list[dict]) -> list[dict] | None:
    """Rows of the latency table (ms per decision), if LATENCY exists."""
    data = read_json(LATENCY)
    if data is None:
        return None
    names = {b["model"]: b["label"] for b in baselines}
    names.update({f"runs/{m['id']}": "UltraBERT-380M" for m in encoders})
    rows = []
    for key, r in data["models"].items():
        model, _, variant = key.partition(" (")
        lat = r["latency"]
        rows.append({"label": names.get(model, model), "variant": variant.rstrip(")") or "bf16 weights",
                     "mean_ms": lat["mean_ms"], "median_ms": lat["median_ms"], "p90_ms": lat["p90_ms"],
                     "buckets": {k: v["mean_ms"] for k, v in lat.items() if k.startswith("ms_tokens_")},
                     "batched_rows_per_s": (r.get("throughput") or {}).get("rows_per_s")})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="blogs/data/encoder.json")
    args = parser.parse_args()

    encoders = []
    for folder in sorted(RUNS.glob(ENCODER_GLOB)):
        if not (folder / "config.yaml").exists() or any(s in folder.name for s in SKIP):
            continue
        cfg = OmegaConf.load(folder / "config.yaml")
        final = accuracies(read_json(folder / "eval_final.json"))
        if final is None:
            continue  # still training
        encoders.append({
            "id": folder.name, "model": cfg.model.id, "lr": cfg.optim.lr, "seed": cfg.run.seed,
            "batch_size": cfg.optim.batch_size, "epochs": cfg.run.epochs,
            "format": OmegaConf.to_container(cfg.format),
            "base": accuracies(read_json(folder / "eval_base.json")),
            "final": final,
            "bench": accuracies(read_json(folder / "eval_benchmarks_final.json")),
            "curves": curves(folder),
        })
    baselines = [{"id": run, "label": label, "model": model, "params": params,
                  "final": accuracies(read_json(RUNS / run / "eval_final.json")),
                  "bench": accuracies(read_json(RUNS / run / "eval_benchmarks_final.json"))}
                 for run, label, model, params in BASELINES]

    everyone = encoders + baselines
    datasets = sorted(set.intersection(*(set(m["final"]) for m in everyone if m["final"])))
    benches = sorted(set.intersection(*(set(m["bench"]) for m in everyone if m["bench"]))
                     ) if all(m["bench"] for m in everyone) else []
    for m in everyone:
        m["final"] = with_totals(m["final"], datasets)
        m["bench"] = with_totals(m["bench"], benches, bench=True)
        if m.get("base"):
            m["base"] = with_totals(m["base"], datasets)
    encoders = fold_seeds(encoders)
    # The featured run matches the decoders' budget (one epoch); longer runs are listed, not featured.
    same_budget = [m for m in encoders if m["epochs"] == FEATURED_EPOCHS] or encoders
    best = max(same_budget, key=lambda m: m["final"]["ALL"])
    for m in encoders:
        m["label"] = f"UltraBERT-380M · lr {m['lr']:.1e}".replace("e-0", "e-") + (f" · {m['epochs']} epochs" if m["epochs"] > 1 else "")
        m["best"] = m is best
        m["steps"] = m["curves"]["loss"][-1][0] if m["curves"].get("loss") else None

    out = pathlib.Path(args.out)
    out.write_text(json.dumps({"datasets": datasets, "benchmarks": benches,
                               "encoders": encoders, "baselines": baselines,
                               "latency": latency(baselines, encoders)}))
    print(f"{len(encoders)} encoder runs (best {best['id']}, ALL {best['final']['ALL']:.3f}), "
          f"{len(baselines)} baselines, {len(datasets)} datasets -> {out}")


if __name__ == "__main__":
    main()
