"""Builds (and with --upload, pushes) the Hugging Face release of the UltraBERT decision models.

Usage:
  .venv/bin/python scripts/release_encoders.py            # build runs/release/<repo>/
  .venv/bin/python scripts/release_encoders.py --upload   # build, then push each folder

One repo per recipe. Each folder holds the weights and tokenizer as trained
(model.safetensors is linked, not copied), config.json without auto_map (the
checkpoints use the UltraBERT class built into transformers; the base repo's
remote code names the weights differently and would leave them unloaded), the
training config, a README from runs/release/card_template.md, and for two
recipes the evidence head (safetensors), its config and its dev-tuned threshold.
"""

import argparse
import json
import pathlib
import shutil

import torch
from safetensors.torch import save_file

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = RUNS / "release"
USER = "Athekunal"

# repo -> (run, recipe text, score, evidence head run or None). The checkpoint is the
# seed with the best dev accuracy, except where the evidence head fixes the encoder.
RELEASES = {
    "ultrabert-jev-380m-softmax-yes": (
        "ultrabert_380m_lr1.5e-5", "softmax over each option's \" yes\" logit", "yes",
        "evidence_ultrabert_380m"),
    "ultrabert-jev-380m-softmax-yesno": (
        "ultrabert_380m_lr1.5e-5_yesno_seed1", "softmax over \" yes\" minus \" no\"", "yes-no", None),
    "ultrabert-jev-380m-sigmoid-norm": (
        "ultrabert_380m_lr1.5e-5_sigmoidnorm",
        "sigmoid of \" yes\" minus \" no\", divided by the sum over options", "yes-no",
        "evidence_ultrabert_380m_sigmoidnorm"),
    "ultrabert-jev-380m-sqrt-softplus": (
        "ultrabert_380m_lr1.5e-5_sqrtsoftplus",
        "square-root softplus of \" yes\" minus \" no\", divided by the sum over options", "yes-no", None),
}


def acc(path: pathlib.Path) -> float:
    return 100 * json.loads(path.read_text())["ALL"]["accuracy"]


def build(repo: str) -> pathlib.Path:
    run, recipe, score, head = RELEASES[repo]
    src, dst = RUNS / run / "model", OUT / repo
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    cfg = json.loads((src / "config.json").read_text())
    cfg.pop("auto_map", None)
    (dst / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    for f in src.iterdir():
        if f.name != "config.json":
            (dst / f.name).symlink_to(f.resolve())
    shutil.copy(RUNS / run / "config.yaml", dst / "training_config.yaml")
    values = {"repo": f"{USER}/{repo}", "recipe": recipe, "score": score,
              "dev": f"{acc(RUNS / run / 'eval_final.json'):.1f}",
              "test": f"{acc(RUNS / run / 'eval_test.json'):.1f}"}
    card = (OUT / "card_template.md").read_text()
    head_section = ""
    if head:
        hd = dst / "evidence_head"
        hd.mkdir()
        sd = torch.load(RUNS / head / "head.pt", map_location="cpu", weights_only=True)
        save_file({k: v.contiguous() for k, v in sd.items()}, hd / "model.safetensors")
        shutil.copy(RUNS / head / "config.yaml", hd / "training_config.yaml")
        th = json.loads((RUNS / head / "threshold_test.json").read_text())
        (hd / "threshold.json").write_text(json.dumps(
            {"threshold": th["threshold"], "tuned_on": f"{th['tuning_rows']} dev rows with gold evidence, word F1"},
            indent=2) + "\n")
        head_section = (OUT / "card_head.md").read_text()
    card = card.replace("{{HEAD_SECTION}}", head_section)
    for k, v in values.items():
        card = card.replace("{{" + k + "}}", v)
    (dst / "README.md").write_text(card)
    return dst


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--only", nargs="*", help="repo names (default: all)")
    args = parser.parse_args()
    for repo in args.only or RELEASES:
        dst = build(repo)
        print("built", dst)
        if args.upload:
            from huggingface_hub import HfApi

            api = HfApi()
            api.create_repo(f"{USER}/{repo}", exist_ok=True, private=False)
            api.upload_folder(repo_id=f"{USER}/{repo}", folder_path=str(dst),
                              commit_message="Release UltraBERT-380M decision model")
            print("uploaded", f"https://huggingface.co/{USER}/{repo}")


if __name__ == "__main__":
    main()
