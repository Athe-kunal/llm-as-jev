"""Results for the paper, read from the runs' files (the same sources as runs/tools/write_results.py).

Every number in the paper comes from here: figures, generated tables and the \\newcommand numbers.
"""
import json
import pathlib
import random
import statistics as st

ROOT = pathlib.Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
EXT = ("boolq", "mmlu", "mmlu_pro", "arc_challenge", "winogrande", "sciq")

# Each model: its runs (seeds), its family (encoder/decoder) and its parameter count.
MODELS = {
    "ub_yesno": dict(name="UltraBERT-380M, softmax, yes $-$ no", short="UltraBERT softmax yes-no", family="encoder",
                     params="380M", ours=True,
                     runs=["ultrabert_380m_lr1.5e-5_yesno", "ultrabert_380m_lr1.5e-5_yesno_seed1",
                           "ultrabert_380m_lr1.5e-5_yesno_seed2"]),
    "ub_sig": dict(name="UltraBERT-380M, sigmoid-norm, yes $-$ no", short="UltraBERT sigmoid-norm", family="encoder",
                   params="380M", ours=True,
                   runs=["ultrabert_380m_lr1.5e-5_sigmoidnorm", "ultrabert_380m_sigmoidnorm_lr1.5e-5_seed1",
                         "ultrabert_380m_sigmoidnorm_lr1.5e-5_seed2"]),
    "ub_list": dict(name="UltraBERT-380M, softmax, yes", short="UltraBERT softmax yes", family="encoder", params="380M",
                    ours=True,
                    runs=["ultrabert_380m_lr1.5e-5", "ultrabert_380m_lr1.5e-5_seed1", "ultrabert_380m_lr1.5e-5_seed2"]),
    "qw_full": dict(name="Qwen3-0.6B, full FT", short="Qwen3-0.6B full FT", family="decoder", params="0.6B",
                    ours=False,
                    runs=["qwen3_0.6b_full_lr1e-5", "qwen3_0.6b_full_lr1e-5_seed1", "qwen3_0.6b_full_lr1e-5_seed2"]),
    "qw_lora": dict(name="Qwen3-0.6B, LoRA (reproduced)", short="Qwen3-0.6B LoRA", family="decoder", params="0.6B",
                    ours=False, runs=["qwen3_0.6b_released"]),
    "qw4_lora": dict(name="Qwen3-4B, LoRA (reproduced)", short="Qwen3-4B LoRA", family="decoder", params="4B",
                     ours=False, runs=["qwen3_4b_released"]),
}
# The dev-split result file of each run (Qwen full-FT checkpoints were re-scored in bf16).
DEV_TAG = {"qwen3_0.6b_full_lr1e-5": "final_bf16", "qwen3_0.6b_full_lr1e-5_seed1": "final_bf16",
           "qwen3_0.6b_full_lr1e-5_seed2": "final_bf16"}


def _json(path):
    return json.loads(path.read_text()) if path.exists() else None


def dev_all(run):
    """Dev ALL of a run: the mean over its datasets."""
    f = _json(RUNS / run / f"eval_{DEV_TAG.get(run, 'final')}.json")
    if f is None:
        return None
    acc = [v["accuracy"] for k, v in f.items() if k not in ("ALL", "MACRO")]
    return sum(acc) / len(acc)


def external(run):
    """External macro of a run: the mean over the six external benchmarks."""
    b = _json(RUNS / run / "eval_benchmarks_final.json")
    return None if b is None else sum(b[d]["accuracy"] for d in EXT) / len(EXT)


def seed_avg(runs, tag="test"):
    """Row id -> (dataset, mean correctness over the runs), on the rows every run has."""
    per = [p for p in (_json(RUNS / r / f"eval_{tag}_rows.json") for r in runs) if p]
    if not per:
        return None
    common = set.intersection(*({x["id"] for x in p} for p in per))
    out = {}
    for p in per:
        for x in p:
            if x["id"] in common:
                src, tot = out.get(x["id"], (x["source"], 0.0))
                out[x["id"]] = (src, tot + float(x["correct"]) / len(per))
    return out


def macro(rows):
    """ALL: the mean over datasets of each dataset's accuracy."""
    by = {}
    for src, c in rows.values():
        by.setdefault(src, []).append(c)
    return sum(sum(v) / len(v) for v in by.values()) / len(by)


def boot(rows_a, rows_b=None, n=2000, seed=0):
    """Point and 95% stratified-bootstrap interval of ALL (or of the paired difference a - b)."""
    rng = random.Random(seed)
    ids = sorted(rows_a) if rows_b is None else sorted(set(rows_a) & set(rows_b))
    by = {}
    for i in ids:
        by.setdefault(rows_a[i][0], []).append(i)
    stats = []
    for _ in range(n):
        tot = 0.0
        for members in by.values():
            pick = [members[rng.randrange(len(members))] for _ in members]
            tot += sum(rows_a[i][1] - (rows_b[i][1] if rows_b is not None else 0.0) for i in pick) / len(pick)
        stats.append(tot / len(by))
    stats.sort()
    point = macro({i: rows_a[i] for i in ids}) - (macro({i: rows_b[i] for i in ids}) if rows_b is not None else 0.0)
    return point, stats[int(0.025 * n)], stats[int(0.975 * n) - 1]


def mean_sd(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    return st.mean(vals), (st.stdev(vals) if len(vals) > 1 else None)


def headline():
    """Per model: dev ALL, external macro (mean, sd over seeds) and test ALL with its 95% interval."""
    out = {}
    for key, m in MODELS.items():
        rows = seed_avg(m["runs"])
        out[key] = dict(dev=mean_sd([dev_all(r) for r in m["runs"]]),
                        ext=mean_sd([external(r) for r in m["runs"]]),
                        test=boot(rows) if rows else None, rows=rows, seeds=len(m["runs"]))
    return out


PAIRS = [("ub_sig", "qw_full"), ("ub_list", "qw_full"), ("ub_yesno", "qw_full"), ("ub_sig", "ub_list"),
         ("ub_yesno", "ub_list"), ("ub_sig", "ub_yesno"), ("qw_full", "qw_lora")]


def paired(h):
    """Paired test differences (first - second) with 95% intervals."""
    return {(a, b): boot(h[a]["rows"], h[b]["rows"]) for a, b in PAIRS}


# Learning-rate sweeps (one seed per value): loss/model -> [(lr, run)]. The Qwen runs were scored in bf16 on dev.
SWEEPS = {
    "ub_sig": [(8e-6, "ultrabert_380m_sigmoidnorm_lr8e-6"), (1.5e-5, "ultrabert_380m_lr1.5e-5_sigmoidnorm"),
               (2e-5, "ultrabert_380m_sigmoidnorm_lr2e-5"), (3e-5, "ultrabert_380m_sigmoidnorm_lr3e-5")],
    "ub_list": [(8e-6, "ultrabert_380m"), (1.5e-5, "ultrabert_380m_lr1.5e-5"),
                (3e-5, "ultrabert_380m_lr3e-5"), (5e-5, "ultrabert_380m_lr5e-5")],
    "qw_full": [(5e-6, "qwen3_0.6b_full_lr5e-6"), (1e-5, "qwen3_0.6b_full_lr1e-5"), (2e-5, "qwen3_0.6b_full_lr2e-5")],
}
DEV_TAG.update({run: "final_bf16" for k in ("qw_full",) for _, run in SWEEPS[k]})

# Per-dataset display names, in the order of the appendix table (groups: intent, short reasoning, long input).
DATASETS = [("clinc150", "CLINC150"), ("massive", "MASSIVE"), ("bitext", "Bitext"),
            ("reclor", "ReClor"), ("logiqa2", "LogiQA 2.0"), ("cosmosqa", "CosmosQA"), ("piqa", "PIQA"),
            ("anli", "ANLI"), ("proofwriter", "ProofWriter"),
            ("contractnli", "ContractNLI"), ("musique", "MuSiQue"), ("quality", "QuALITY"), ("nimble", "Nimble")]


def per_dataset(h):
    """Model -> dataset -> (mean accuracy of the seed-averaged rows, number of rows)."""
    out = {}
    for k, r in h.items():
        by = {}
        for src, c in (r["rows"] or {}).values():
            by.setdefault(src, []).append(c)
        out[k] = {s: (sum(v) / len(v), len(v)) for s, v in by.items()}
    return out
