"""Trains the citation head (modeling.citation) on AT2's ablation data, decision model frozen.

Usage:
  CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src .venv/bin/python -m train.train_citation \
      --model Qwen/Qwen3-0.6B --adapter runs/qwen3_0.6b_lam0.01/adapter \
      --data runs/at2_qwen3_0.6b/data.npz --out runs/citation_qwen3_0.6b

The rows and targets are dataloader.at2_data's (logit q of the predicted answer
under 32 ablations); each row's masks are regenerated with
context_cite.ablation_masks (same seed, so the same masks). The surrogate is
f(v) = sum_j (2 v_j - 1) * score_j and the loss losses.pearson_loss, as in
AT2; rows and the held-out split are AT2's (train.train_at2), so the two
are directly comparable. Writes <out>/head.pt and <out>/train_citation.log.
"""

import argparse
import math
import pathlib

from loguru import logger
import numpy as np
import torch

from dataloader import data
from modeling import citation, context_cite, modeling
from train import losses, train_at2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--adapter", help="the frozen decision model's LoRA adapter")
    parser.add_argument("--data", required=True, help="dataloader.at2_data output (ids, targets)")
    parser.add_argument("--rows", default=data.TRAIN_EVAL)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logger.add(out / "train_citation.log")
    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = modeling.load(args.model, args.adapter)
    model.requires_grad_(False)
    d = np.load(args.data)
    Y = torch.tensor(d["targets"], dtype=torch.float32)                         # [R, M]
    keep, tr, val, gen = train_at2.split(Y, args.val_frac, args.seed)         # AT2's rows and split
    ids, Y, tr, val = d["ids"][keep.numpy()], Y[keep], tr.tolist(), val.tolist()
    rows = {r["id"]: r for r in data.load_rows(args.rows, "train")}
    masks = [torch.tensor(context_cite.ablation_masks(Y.shape[1], len(context_cite.split_sources(rows[i]))),
                          dtype=torch.float32) * 2 - 1 for i in ids]           # [M, S] each, ±1
    head = citation.CitationHead(model.config.hidden_size).cuda()
    logger.info("{} rows: {} train, {} val; {} masks; head {:,} parameters", len(Y), len(tr), len(val),
                Y.shape[1], sum(p.numel() for p in head.parameters()))

    def predict(i: int) -> torch.Tensor:
        e = citation.encode(model, tok, rows[ids[i]])
        scores = head(e["states"].float(), e["ranges"], e["query_start"])        # [S]
        return masks[i].cuda() @ scores                                         # [M]

    def evaluate(idx: list[int]) -> float:
        head.eval()
        with torch.no_grad():
            corr = -losses.pearson_loss(torch.stack([predict(i) for i in idx]), Y[idx].cuda()).item()
        head.train()
        return corr

    steps = args.epochs * math.ceil(len(tr) / args.batch_size)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    logger.info("before training: Pearson val {:.4f}", evaluate(val))
    step = 0
    for epoch in range(args.epochs):
        order = [tr[j] for j in torch.randperm(len(tr), generator=gen).tolist()]
        run = []
        for b in range(0, len(order), args.batch_size):
            batch = order[b:b + args.batch_size]
            loss = losses.pearson_loss(torch.stack([predict(i) for i in batch]), Y[batch].cuda())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), args.max_grad_norm)
            opt.step()
            sched.step()
            run.append(loss.item())
            step += 1
        logger.info("epoch {}/{} (step {}): train loss {:.4f}, Pearson val {:.4f}",
                    epoch + 1, args.epochs, step, float(np.mean(run)), evaluate(val))
    torch.save({"state_dict": head.state_dict(), "hidden": model.config.hidden_size}, out / "head.pt")
    logger.info("wrote {}", out / "head.pt")


if __name__ == "__main__":
    main()
