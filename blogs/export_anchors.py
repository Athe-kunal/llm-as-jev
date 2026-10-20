"""Exports the KL anchors (L_mass, L_out, L_pos) of one row, as JSON.

Usage (from the repo root; needs a GPU):
  CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src .venv/bin/python blogs/export_anchors.py

The same Bitext row as export_trie.py, options in release order, plus
answers.OFF_PATH off-path nodes sampled as in answers.encode. For each
trained adapter (lambda = 1 and lambda = 0.01), every anchor term is computed
with train.losses against p0 (the same model with the adapter disabled), and
the weighted sums are checked against losses.compute. Writes
blogs/data/anchors.json (real dataset rows: git-ignored, local only).
"""

import json
import pathlib
import random
import re
import sys

import torch
import transformers
from loguru import logger

from dataloader import answers, data
from modeling import modeling
from train import losses

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from export_trie import _MODEL, pick_row  # noqa: E402

_HERE = pathlib.Path(__file__).parent
_ADAPTERS = {"lam1": ("λ = 1", "runs/qwen3_0.6b/adapter"),
             "lam001": ("λ = 0.01", "runs/qwen3_0.6b_lam0.01/adapter")}
_SHOW = 8  # tokens drawn per distribution (the KL itself uses losses.TOP)
_LOG = re.compile(r"step (\d+)/\d+ tree ([\d.]+) mass ([\d.]+) out ([\d.]+) pos ([\d.]+)")


def results(run: pathlib.Path, tag: str) -> dict[str, float]:
    """Dev accuracy and a few benchmark accuracies of a run (evaluate.py output)."""
    dev = json.loads((run / f"eval_{tag}.json").read_text())["ALL"]["accuracy"]
    bench = json.loads((run / f"eval_benchmarks_{tag}.json").read_text())
    return {"dev": dev, **{k: bench[k]["accuracy"] for k in
                           ("external_macro", "boolq", "banking77", "jevbench")}}


def curves(adapter: str) -> list[list[float]]:
    """[step, tree, mass, out, pos] from the run's train.log (every 5 steps)."""
    log = pathlib.Path(adapter).parent / "train.log"
    return [[int(m[1])] + [float(x) for x in m.groups()[1:]]
            for m in map(_LOG.search, log.read_text().splitlines()) if m]


def with_off_path(ex: dict, tok, seed: int = 0) -> dict:
    """ex plus OFF_PATH off-path nodes, sampled as answers.encode does."""
    _, nodes = answers.candidates(tok, ex["n_options"])
    gold = [g for _, _, g in ex["path"]]
    on = {tuple(gold[:t]) for t in range(len(gold))}
    off = [n for n in nodes if n not in on]
    picked = random.Random(seed).sample(off, min(answers.OFF_PATH, len(off)))
    return {**ex, "off_path": [(list(n), nodes[n]) for n in picked]}


def dist(tok, idx, ref_lp, cur_lp) -> list:
    """The first _SHOW anchored tokens: text, p0 and p_theta probabilities."""
    return [[tok.decode([i]), r, c] for i, r, c in
            zip(idx[:_SHOW].tolist(), ref_lp[:_SHOW].exp().tolist(), cur_lp[:_SHOW].exp().tolist())]


@torch.no_grad()
def anchors(model, ex: dict, tok) -> dict:
    """Every anchor term of one example, node by node and position by position."""
    pad = tok.pad_token_id
    prompt = ex["input_ids"][: ex["path"][0][0] + 1]
    seqs = [ex["input_ids"]] + [prompt + n for n, _ in ex["off_path"]]
    with model.disable_adapter():
        ref = losses._logits(model, ex, seqs, "cuda", pad)
    cur = losses._logits(model, ex, seqs, "cuda", pad)
    gold = [g for _, _, g in ex["path"]]
    paths = [gold[:t] for t in range(len(gold))] + [n for n, _ in ex["off_path"]]
    legal_ids = [l for _, l, _ in ex["path"]] + [l for _, l in ex["off_path"]]
    T, O = len(ex["path"]), len(ex["off_path"])
    legal = losses.legal_mask(legal_ids, cur["path"].shape[-1], "cuda")
    nodes = torch.cat([cur["path"], cur["last"][1:]])
    ref_nodes = torch.cat([ref["path"], ref["last"][1:]])
    st = losses.node_stats(ref_nodes, legal)
    an = losses.node_anchor(nodes, legal, st)
    s = losses._split(nodes, legal)
    weight = [0.5 / T] * T + [0.5 / max(O, 1)] * O
    out_nodes = []
    for i in range(T + O):
        cur_lp = s["logp"][i].gather(-1, st["idx"][i]) - s["log_r"][i]
        out_nodes.append({
            "kind": "gold" if i < T else "off", "path": paths[i], "legal": legal_ids[i],
            "weight": weight[i],
            "m0": st["log_m"][i].exp().item(), "m": s["log_m"][i].exp().item(),
            "r0": st["log_r"][i].exp().item(), "r": s["log_r"][i].exp().item(),
            "mass": an["mass"][i].item(), "out": an["out"][i].item(),
            "illegal": dist(tok, st["idx"][i], st["lp"][i], cur_lp),
            "illegalRest": [1 - st["lp"][i][:_SHOW].exp().sum().item(),
                            1 - cur_lp[:_SHOW].exp().sum().item()],
        })
    sites = torch.stack([cur["reply"], cur["last"][0]])
    ref_sites = torch.stack([ref["reply"], ref["last"][0]])
    ps = losses.pos_stats(ref_sites)
    pa = losses.pos_anchor(sites, ps)
    logp = torch.log_softmax(sites.float(), -1)
    positions = []
    for i, name in enumerate(["reply start", "after ]"]):
        cur_lp = logp[i].gather(-1, ps["idx"][i])
        positions.append({"name": name, "weight": 0.5, "kl": pa[i].item(),
                          "top": dist(tok, ps["idx"][i], ps["lp"][i], cur_lp),
                          "rest": [1 - ps["lp"][i][:_SHOW].exp().sum().item(),
                                   1 - cur_lp[:_SHOW].exp().sum().item()]})
    total = {"mass": sum(w * n["mass"] for w, n in zip(weight, out_nodes)),
             "out": sum(w * n["out"] for w, n in zip(weight, out_nodes)),
             "pos": sum(0.5 * p["kl"] for p in positions)}
    ref_total = losses.compute(model, ex, "cuda", pad)
    for k in total:  # same numbers as the training loss
        if abs(total[k] - ref_total[k].item()) > 1e-4 * max(1.0, abs(total[k])):
            raise ValueError(f"{k}: {total[k]} != losses.compute {ref_total[k].item()}")
    total["tree"] = ref_total["tree"].item()
    return {"nodes": out_nodes, "positions": positions, "total": total}


def main() -> None:
    """Writes data/anchors.json."""
    tok = transformers.AutoTokenizer.from_pretrained(_MODEL)
    row = pick_row(data.load_rows(split="eval"))
    ex = with_off_path(answers.encode(row, tok, None), tok)
    result = {}
    for name, (label, adapter) in _ADAPTERS.items():
        model = modeling.load_trained(_MODEL, adapter)
        result[name] = {"label": label, "curve": curves(adapter),
                        "results": results(pathlib.Path(adapter).parent, "final"),
                        **anchors(model, ex, tok)}
        t = result[name]["total"]
        logger.info("{}: tree {:.4f} mass {:.5f} out {:.4f} pos {:.4f}", label, t["tree"],
                    t["mass"], t["out"], t["pos"])
        del model
    order = list(row["labels"])
    reply = ex["input_ids"][ex["reply"] + 1: ex["reply"] + 4]
    payload = {
        "model": _MODEL, "top": losses.TOP, "offPath": answers.OFF_PATH,
        "row": {"id": row["id"], "k": len(order), "gold": ex["gold"],
                "options": [row["labels"][k] for k in order]},
        "tokText": {str(i): tok.decode([i]) for n in result["lam1"]["nodes"]
                    for i in n["path"] + n["legal"]},
        "replyText": tok.decode(reply),
        "adapters": result,
        "base": results(pathlib.Path(_ADAPTERS["lam1"][1]).parent, "base"),
    }
    out = _HERE / "data" / "anchors.json"
    out.write_text(json.dumps(payload, ensure_ascii=False))
    logger.info("wrote {} ({:.0f} KB)", out, out.stat().st_size / 1024)


if __name__ == "__main__":
    main()
