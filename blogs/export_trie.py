"""Exports one eval row's trie and the model's logits at every node, as JSON.

Usage (from the repo root; needs a GPU):
  CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src .venv/bin/python blogs/export_trie.py

The row is a Bitext eval row (27 options, release order) whose gold option
has a two-digit id, so the gold path has three nodes. Each trie node is
scored by the base Qwen3-0.6B and by the trained LoRA adapter; L_tree on the
gold path is checked against train.losses.tree_loss. Writes
blogs/data/trie_loss.json (real dataset rows: git-ignored, local only).
"""

import json
import pathlib
import random
import sys

import torch
import transformers
from loguru import logger

from dataloader import answers, data
from modeling import modeling
from train import losses

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from export_data import segments  # noqa: E402

_HERE = pathlib.Path(__file__).parent
_MODEL = "Qwen/Qwen3-0.6B"
_ADAPTER = "runs/qwen3_0.6b_lam0.01/adapter"
_SOURCE, _SEED = "bitext", 0
_TOP = 6  # most likely tokens over the whole vocabulary, per node


def pick_row(rows: list[dict]) -> dict:
    """A seeded eval row of _SOURCE whose gold id has two digits."""
    pool = [r for r in rows if r["source"] == _SOURCE
            and list(r["labels"]).index(r["label"]) >= 9]
    return random.Random(_SEED).choice(pool)


@torch.no_grad()
def node_scores(model, prompt: list[int], nodes: dict, tok) -> dict[str, dict]:
    """Per trie node: legal logits, full-vocabulary log-probs and top tokens."""
    keys = list(nodes)
    _, logits = modeling.batched_hidden(
        model, [prompt + list(n) for n in keys], "cuda", tok.pad_token_id)
    out = {}
    for n, z in zip(keys, logits.float()):
        logp = torch.log_softmax(z, -1)
        legal = nodes[n]
        top = logp.topk(_TOP)
        out[",".join(map(str, n))] = {
            "z": z[legal].tolist(),
            "logp": logp[legal].tolist(),
            "legalMass": logp[legal].exp().sum().item(),
            "top": [[tok.decode([i]), p] for i, p in
                    zip(top.indices.tolist(), top.values.exp().tolist())],
        }
    return out


def tree(scores: dict, suffixes: list, nodes: dict, gold: int) -> dict:
    """q over all candidates (product of legal-normalized steps) and the
    gold path's L_tree terms."""
    def step(node, t):
        z = torch.tensor(scores[",".join(map(str, node))]["z"])
        return torch.log_softmax(z, 0)[nodes[node].index(t)].item()

    logq = [sum(step(tuple(s[:t]), s[t]) for t in range(len(s))) for s in suffixes]
    g = suffixes[gold]
    terms = [-step(tuple(g[:t]), g[t]) for t in range(len(g))]
    return {"q": [float(torch.tensor(x).exp()) for x in logq], "terms": terms,
            "loss": sum(terms)}


def main() -> None:
    """Writes data/trie_loss.json."""
    tok = transformers.AutoTokenizer.from_pretrained(_MODEL)
    row = pick_row(data.load_rows(split="eval"))
    ex = answers.encode(row, tok, None)
    order = list(row["labels"])
    prompt = ex["input_ids"][: ex["path"][0][0] + 1]
    suffixes, nodes = answers._candidates(tok, len(order))
    segs, n_tokens = segments(row, order, tok)

    models = {}
    base = modeling.load_base(_MODEL).to("cuda").eval()
    models["base"] = node_scores(base, prompt, nodes, tok)
    del base
    models["trained"] = node_scores(
        modeling.load_trained(_MODEL, _ADAPTER), prompt, nodes, tok)

    result = {}
    for name, scores in models.items():
        result[name] = {"nodes": scores, **tree(scores, suffixes, nodes, ex["gold"])}
        # check: the same number as the training loss on the gold path
        path = [tuple(suffixes[ex["gold"]][:t]) for t in range(len(suffixes[ex["gold"]]))]
        full = torch.full((len(path), len(tok)), -1e4)
        for i, n in enumerate(path):
            full[i, nodes[n]] = torch.tensor(scores[",".join(map(str, n))]["z"])
        ref = losses.tree_loss(full, losses.legal_mask([nodes[n] for n in path], len(tok), "cpu"),
                               torch.tensor([g for _, _, g in ex["path"]])).item()
        logger.info("{}: L_tree {:.4f} (losses.tree_loss {:.4f}), q(gold) {:.3f}",
                    name, result[name]["loss"], ref, result[name]["q"][ex["gold"]])

    ids = sorted({t for s in suffixes for t in s})
    payload = {
        "model": _MODEL, "adapter": _ADAPTER,
        "row": {"id": row["id"], "source": row["source"], "segs": segs,
                "tokens": n_tokens, "k": len(order), "gold": ex["gold"],
                "options": [row["labels"][k] for k in order]},
        "vocab": len(tok), "tokText": {str(i): tok.decode([i]) for i in ids},
        "suffixes": suffixes,
        "nodes": [{"path": list(n), "legal": v} for n, v in nodes.items()],
        "models": result,
    }
    out = _HERE / "data" / "trie_loss.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False))
    logger.info("wrote {} ({:.0f} KB): {} gold [{}] {}", out, out.stat().st_size / 1024,
                row["id"], ex["gold"] + 1, row["label"])


if __name__ == "__main__":
    main()
