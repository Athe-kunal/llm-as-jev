"""How strongly an encoder's option probabilities depend on the other options.

The MoE argument for a sigmoid gate (DeepSeek-V3) is that softmax couples the
experts: raising one score moves every probability. Here the options play the
experts, and per run on the dev rows we measure:

  - confidence: raw score scale, top probability, calibration (ECE) of the top
    option, and of sigma(s_k) per option ("is this option correct on its own");
  - by K: accuracy and log p(gold) for rows with few and with many options;
  - shuffle: the options in `shuffles` more random orders; how often the
    decision (mapped back to the original option) changes, and the mean total
    variation between the probability vectors;
  - balance: predicted against gold shares of each label (datasets with one
    fixed label set) and of each position, over all orders, as total variation;
  - distractor: one random wrong option removed (rows with K >= 3, the removed
    option not the prediction); how often the decision among the remaining
    options changes, and how far s(gold), log p(gold) and sigma(s(gold)) move.

p is the run's own training normalisation: softmax(s) for loss "listwise",
sigma(s) / sum sigma(s) for "sigmoid_norm". Decisions are argmax s for both.

Usage:
  CUDA_VISIBLE_DEVICES=3 .venv/bin/python -m llm_jev.evaluation.option_coupling \
      --run runs/ultrabert_380m_lr1.5e-5 --run runs/ultrabert_380m_lr1.5e-5_sigmoidnorm
"""

import argparse
import collections
import dataclasses
import json
import pathlib
import random

from loguru import logger
import torch

from llm_jev.dataloader import mlm
from llm_jev.dataloader import storage
from llm_jev.evaluation import evaluate as decoder_eval
from llm_jev.evaluation import evaluate_mlm

K_BUCKETS = ((2, 2), (3, 4), (5, 10), (11, 10**6))  # inclusive ranges of K
ECE_BINS = 15
MAX_LABELS = 5  # a "fixed label set" dataset has one option set of at most this size


def probs(s: torch.Tensor, kind: str) -> torch.Tensor:
    """The run's training distribution over the K options."""
    if kind == "sqrt_softplus":
        from llm_jev.modeling import encoder
        return torch.softmax(encoder.log_sqrt_softplus(s), 0)
    if kind == "sigmoid_norm":
        sig = torch.sigmoid(s)
        return sig / sig.sum()
    return torch.softmax(s, 0)


def ece(conf: list[float], hit: list[float], bins: int = ECE_BINS) -> float:
    """Expected calibration error with equal-width bins."""
    total, err = len(conf), 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(conf) if lo < c <= hi or (b == 0 and c == 0)]
        if idx:
            err += len(idx) / total * abs(sum(conf[i] for i in idx) / len(idx)
                                          - sum(hit[i] for i in idx) / len(idx))
    return err


def tv(a: dict, b: dict) -> float:
    """Total variation between two share dicts."""
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in set(a) | set(b))


def shares(counter: collections.Counter) -> dict:
    n = sum(counter.values())
    return {k: v / n for k, v in counter.items()}


def mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def by_dataset_mean(per: dict[str, list[float]]) -> float:
    """Mean over datasets of each dataset's mean, as ALL elsewhere."""
    return mean(mean(v) for v in per.values() if v)


def analyse(run: pathlib.Path, rows, shuffles: int, seed: int) -> dict:
    cfg = evaluate_mlm.run_config(run)
    kind = cfg.optim.loss
    scorer, tok, fmt = evaluate_mlm.load_run(run)
    budget = cfg.model.pack_tokens

    # Jobs: (row index, variant, encoded). Variant 0 is the base order; 1..shuffles
    # are reshuffles; "drop" removes one wrong option from the base order.
    jobs = []
    for i, r in enumerate(rows):
        for v in range(shuffles + 1):
            jobs.append((i, v, mlm.encode(r, tok, fmt, random.Random(f"coupling:{seed}:{r.id}:{v}"))))
    scores = evaluate_mlm.score_encoded(scorer, [ex for _, _, ex in jobs], budget=budget)
    base = {}
    orig = collections.defaultdict(list)  # row -> [(pred original option, p over original options)]
    for (i, v, ex), s in zip(jobs, scores):
        s = s.float().cpu()
        p = probs(s, kind)
        k = len(ex["order"])
        p_orig = torch.zeros(k)
        p_orig[torch.tensor(ex["order"])] = p
        orig[i].append((ex["order"][int(s.argmax())], p_orig, ex))
        if v == 0:
            base[i] = (s, p, ex)

    # Distractor removal from the base order.
    drop_jobs = []
    for i, r in enumerate(rows):
        s, p, ex = base[i]
        k = len(ex["order"])
        pred = int(s.argmax())
        wrong = [m for m in range(k) if m != ex["gold"] and m != pred]
        if k < 3 or not wrong:
            continue
        m = random.Random(f"drop:{seed}:{r.id}").choice(wrong)
        removed = ex["order"][m]
        keep = [o for o in range(len(r.options)) if o != removed]
        sub = dataclasses.replace(r, options=tuple(r.options[o] for o in keep), gold=keep.index(r.gold))
        order = [keep.index(o) for o in ex["order"] if o != removed]  # same relative order as the base
        text, _ = mlm.build_prompt(sub, order, tok, fmt)
        ids = tok(text)["input_ids"]
        masks = [t for t, x in enumerate(ids) if x == tok.mask_token_id]
        drop_jobs.append((i, m, {"input_ids": ids, "masks": masks, "gold": order.index(sub.gold),
                                 "n_options": len(order), "order": order}))
    drop_scores = evaluate_mlm.score_encoded(scorer, [ex for _, _, ex in drop_jobs], budget=budget)

    # ---- metrics ----
    acc = collections.defaultdict(list)
    top_conf, top_hit, opt_conf, opt_hit, scale, spread = [], [], [], [], [], []
    k_acc = collections.defaultdict(list)
    k_logp = collections.defaultdict(list)
    for i, r in enumerate(rows):
        s, p, ex = base[i]
        hit = float(int(s.argmax()) == ex["gold"])
        acc[r.source].append(hit)
        top_conf.append(float(p.max()))
        top_hit.append(hit)
        sig = torch.sigmoid(s)
        for m in range(len(s)):
            opt_conf.append(float(sig[m]))
            opt_hit.append(float(m == ex["gold"]))
        scale.append(float(s.abs().mean()))
        spread.append(float(s.max() - s.min()))
        k = len(s)
        for lo, hi in K_BUCKETS:
            if lo <= k <= hi:
                k_acc[f"{lo}-{hi}" if hi < 10**6 else f">={lo}"].append(hit)
                k_logp[f"{lo}-{hi}" if hi < 10**6 else f">={lo}"].append(float(p[ex["gold"]].log()))

    flip, tvs = collections.defaultdict(list), collections.defaultdict(list)
    lab_gold, lab_pred = collections.defaultdict(collections.Counter), collections.defaultdict(collections.Counter)
    pos_gold, pos_pred = collections.defaultdict(collections.Counter), collections.defaultdict(collections.Counter)
    by_src = collections.defaultdict(list)
    for r in rows:
        by_src[r.source].append(r)
    fixed = {src for src, rs in by_src.items()
             if len({x.options for x in rs}) == 1 and len(rs[0].options) <= MAX_LABELS}
    for i, r in enumerate(rows):
        views = orig[i]
        preds = [v[0] for v in views]
        flip[r.source].append(float(len(set(preds)) > 1))
        tvs[r.source].append(mean(0.5 * float((views[0][1] - v[1]).abs().sum()) for v in views[1:]))
        for pred, _, ex in views:
            k = len(ex["order"])
            pos_gold[k][ex["gold"]] += 1
            pos_pred[k][ex["order"].index(pred)] += 1
            if r.source in fixed:
                lab_gold[r.source][r.options[r.gold]] += 1
                lab_pred[r.source][r.options[pred]] += 1

    d_flip, d_s, d_logp, d_sig = (collections.defaultdict(list) for _ in range(4))
    for (i, m, ex2), s2 in zip(drop_jobs, drop_scores):
        r = rows[i]
        s, p, ex = base[i]
        s2 = s2.float().cpu()
        p2 = probs(s2, kind)
        keep_idx = [j for j in range(len(s)) if j != m]          # base mask index -> drop mask index
        pred_base = keep_idx.index(int(s.argmax()))
        d_flip[r.source].append(float(int(s2.argmax()) != pred_base))
        g, g2 = ex["gold"], ex2["gold"]
        d_s[r.source].append(abs(float(s2[g2] - s[g])))
        d_logp[r.source].append(abs(float(p2[g2].log() - p[g].log())))
        d_sig[r.source].append(abs(float(torch.sigmoid(s2[g2]) - torch.sigmoid(s[g]))))

    return {
        "run": str(run), "loss": kind, "rows": len(rows), "shuffles": shuffles,
        "accuracy": by_dataset_mean(acc),
        "confidence": {"mean_abs_score": mean(scale), "mean_score_range": mean(spread),
                       "mean_top_p": mean(top_conf), "ece_top": ece(top_conf, top_hit),
                       "ece_option_sigmoid": ece(opt_conf, opt_hit),
                       "mean_sigmoid_gold": mean(c for c, h in zip(opt_conf, opt_hit) if h),
                       "mean_sigmoid_wrong": mean(c for c, h in zip(opt_conf, opt_hit) if not h)},
        "by_k": {b: {"rows": len(k_acc[b]), "accuracy": mean(k_acc[b]), "log_p_gold": mean(k_logp[b])}
                 for b in k_acc},
        "shuffle": {"decision_changes": by_dataset_mean(flip), "mean_tv": by_dataset_mean(tvs),
                    "by_dataset": {k: {"decision_changes": mean(v), "mean_tv": mean(tvs[k])}
                                   for k, v in sorted(flip.items())}},
        "balance": {"label_tv": {src: tv(shares(lab_gold[src]), shares(lab_pred[src])) for src in sorted(fixed)},
                    "label_shares": {src: {"gold": shares(lab_gold[src]), "predicted": shares(lab_pred[src])}
                                     for src in sorted(fixed)},
                    "position_tv": {str(k): tv(shares(pos_gold[k]), shares(pos_pred[k]))
                                    for k in sorted(pos_gold) if sum(pos_gold[k].values()) >= 100}},
        "distractor": {"rows": len(drop_jobs), "decision_changes": by_dataset_mean(d_flip),
                       "abs_d_score_gold": by_dataset_mean(d_s), "abs_d_log_p_gold": by_dataset_mean(d_logp),
                       "abs_d_sigmoid_gold": by_dataset_mean(d_sig),
                       "by_dataset": {k: {"rows": len(v), "decision_changes": mean(v),
                                          "abs_d_score_gold": mean(d_s[k]), "abs_d_log_p_gold": mean(d_logp[k])}
                                      for k, v in sorted(d_flip.items())}},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", action="append", required=True, help="train_mlm run directory (repeatable)")
    parser.add_argument("--data", default=None, help="default: the run's data.source")
    parser.add_argument("--split", default="eval")
    parser.add_argument("--part", default="dev", choices=decoder_eval.PARTS)
    parser.add_argument("--eval-per", type=int, default=100)
    parser.add_argument("--shuffles", type=int, default=3, help="extra random option orders per row")
    parser.add_argument("--seed", type=int, default=decoder_eval.SEED)
    parser.add_argument("--tag", default="coupling_dev")
    args = parser.parse_args()
    for run in map(pathlib.Path, args.run):
        data = args.data or evaluate_mlm.run_config(run).data.source
        rows = decoder_eval.select_rows(storage.load_examples(data, args.split), args.part, args.eval_per, args.seed)
        logger.info("{}: {} rows", run, len(rows))
        out = analyse(run, rows, args.shuffles, args.seed)
        (run / f"{args.tag}.json").write_text(json.dumps(out, indent=2))
        logger.info("{}: acc {:.3f} | shuffle changes {:.3f} | drop changes {:.3f} | ece {:.3f}",
                    run, out["accuracy"], out["shuffle"]["decision_changes"],
                    out["distractor"]["decision_changes"], out["confidence"]["ece_top"])


if __name__ == "__main__":
    main()
