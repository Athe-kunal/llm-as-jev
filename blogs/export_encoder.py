"""Exports the encoder section: the four UltraBERT recipes against fully fine-tuned Qwen3-0.6B.

The numbers are the paper's (paper/figures/data.py): test accuracy averaged over
seeds, dev and external accuracy, per-dataset test accuracy; plus, per recipe,
the option-sensitivity test (runs/<run>/coupling_dev.json, averaged over seeds),
the training loss of its released run, and time and memory from runs/profile.json.

Usage:
  .venv/bin/python blogs/export_encoder.py
"""

import json
import pathlib
import statistics
import sys

from tensorboard.backend.event_processing import event_accumulator

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paper" / "figures"))
import data  # noqa: E402  (paper/figures/data.py)

OUT = ROOT / "blogs" / "data" / "encoder.json"
HF = "https://huggingface.co/Athekunal/"
# key in data.MODELS -> (tab label, released run, Hugging Face repo, loss formula, one-line description)
RECIPES = {
    "ub_list": ("Softmax, yes", "ultrabert_380m_lr1.5e-5", "ultrabert-jev-380m-softmax-yes",
                r"s_k=\ell_k(\text{yes}),\qquad p_k=\frac{e^{s_k}}{\sum_j e^{s_j}}",
                "Each option is scored by the “ yes” logit at its mask, and a softmax over the options gives the probabilities."),
    "ub_yesno": ("Softmax, yes − no", "ultrabert_380m_lr1.5e-5_yesno_seed1", "ultrabert-jev-380m-softmax-yesno",
                 r"s_k=\ell_k(\text{yes})-\ell_k(\text{no}),\qquad p_k=\frac{e^{s_k}}{\sum_j e^{s_j}}",
                 "The score is the preference for “ yes” over “ no” at the mask; the softmax is unchanged."),
    "ub_sig": ("Sigmoid-norm", "ultrabert_380m_lr1.5e-5_sigmoidnorm", "ultrabert-jev-380m-sigmoid-norm",
               r"s_k=\ell_k(\text{yes})-\ell_k(\text{no}),\qquad p_k=\frac{\sigma(s_k)}{\sum_j \sigma(s_j)}",
               "Each score goes through a sigmoid and is divided by the sum over options, as in the DeepSeek-V3 expert gate."),
    "ub_sqrt": ("Sqrt-softplus", "ultrabert_380m_lr1.5e-5_sqrtsoftplus", "ultrabert-jev-380m-sqrt-softplus",
                r"s_k=\ell_k(\text{yes})-\ell_k(\text{no}),\qquad p_k=\frac{\sqrt{\operatorname{softplus}(s_k)}}{\sum_j \sqrt{\operatorname{softplus}(s_j)}}",
                "Like the sigmoid-norm, but with a function that has no upper bound."),
}
DECODER = "qw_full"
PROFILE = (("UltraBERT-380M", "runs/ultrabert_380m_lr1.5e-5 (fp32 weights)", None),
           ("Qwen3-0.6B", "Athekunal/llm-jev-qwen3-0.6b", "tree"),
           ("Qwen3-4B", "Athekunal/llm-jev-qwen3-4b", "tree"))


def coupling(runs: list[str]) -> dict:
    """Option-sensitivity metrics, averaged over the runs that have them."""
    ds = [json.loads((data.RUNS / r / "coupling_dev.json").read_text())
          for r in runs if (data.RUNS / r / "coupling_dev.json").exists()]
    mean = lambda f: statistics.fmean(f(d) for d in ds)
    return {"reorder": mean(lambda d: d["shuffle"]["decision_changes"]),
            "remove": mean(lambda d: d["distractor"]["decision_changes"]),
            "label_tv": mean(lambda d: statistics.fmean(d["balance"]["label_tv"].values())),
            "ece": mean(lambda d: d["confidence"]["ece_top"])}


def loss_curve(run: str, every: int = 2) -> list:
    ev = event_accumulator.EventAccumulator(str(data.RUNS / run / "tb"), size_guidance={"scalars": 0})
    ev.Reload()
    return [[s.step, s.value] for s in ev.Scalars("train/loss")][::every]


def summary(h: dict, pd_: dict, key: str) -> dict:
    r = h[key]
    return {"test": r["test"][0], "dev": r["dev"][0], "ext": r["ext"][0],
            "per_dataset": {k: pd_[key][k][0] for k, _ in data.DATASETS}}


def main() -> None:
    h = data.headline()
    pd_ = data.per_dataset(h)
    recipes = []
    for key, (label, run, repo, formula, text) in RECIPES.items():
        recipes.append({"key": key, "label": label, "formula": formula, "text": text, "hf": HF + repo,
                        **summary(h, pd_, key), "coupling": coupling(data.MODELS[key]["runs"]),
                        "loss": loss_curve(run)})
    prof = json.loads((data.RUNS / "profile.json").read_text())["models"]
    cost = []
    for label, key, path in PROFILE:
        rec = prof[key][path] if path else prof[key]
        cost.append({"label": label, "ms": rec["latency"]["mean_ms"], "peak_gib": rec["memory"]["peak_max_gib"]})
    OUT.write_text(json.dumps({
        "datasets": [[k, name, pd_[DECODER][k][1]] for k, name in data.DATASETS],
        "recipes": recipes, "decoder": {"label": "Qwen3-0.6B, full FT", **summary(h, pd_, DECODER)},
        "decoder4b": {"label": "Qwen3-4B, LoRA", "test": h["qw4_lora"]["test"][0]},
        "cost": cost}))
    print(f"{len(recipes)} recipes -> {OUT}")


if __name__ == "__main__":
    main()
