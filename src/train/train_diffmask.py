"""Trains DiffMask's gate probe (and baseline vector) on the frozen decision model.

Usage (scripts/train_diffmask.sh wraps it):
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src .venv/bin/python -m train.train_diffmask \
      --adapter runs/qwen3_0.6b_lam0.01/adapter --variant fade --margin 0.05 \
      --out runs/diffmask_qwen3_0.6b/fade_m0.05

One row per step: the unmasked decision pass gives the tree scores s and
every layer's hidden states; the probe turns the document tokens' states into
gates (modeling.diffmask.HardConcrete samples); the masked pass hides the
closed tokens (--variant fade: attention weights scaled by z at every layer;
interp: input embeddings mixed with a learned baseline) and gives s~. The
probe minimises losses.diffmask_loss; the multiplier lam follows dual ascent,
lam <- max(0, lam + lam_lr * (KL - margin)). No labels are used. Saves
<out>/probe.pt (probe, baseline, args) every --save-every steps and at the end.
"""

import argparse
import json
import pathlib
import random
import time

from loguru import logger
import numpy as np
import torch

from dataloader import answers, data
from modeling import context_cite, diffmask, masked_cite, modeling
from train import losses


def prepare(tok, row: dict, device: str) -> dict:
    """The row's one-copy packed sequence (reader + trie after the document),
    its structural block mask and the positions of the gated (document) tokens."""
    sources = context_cite.split_sources(row)
    ids, owner, reader = masked_cite.token_owners(tok, row, sources)
    k = len(row["labels"])
    packed = masked_cite.pack(ids, reader, 1, list(answers.candidates(tok, k)[1]))
    mask = masked_cite.block_mask(packed, owner, np.ones((1, len(sources)), bool), device)
    gated = torch.as_tensor(np.where(owner[:reader] >= 0)[0], device=device)
    return {"packed": packed, "mask": mask, "gated": gated, "k": k}


def masked_scores(model, tok, ex: dict, z: torch.Tensor, variant: str, baseline, device: str) -> torch.Tensor:
    """s~ [K]: the tree scores with the gated tokens hidden by their gates z."""
    p, n = ex["packed"], len(ex["packed"]["input_ids"])
    if variant == "fade":
        extra = {"key_bias": diffmask.fade_bias(z, ex["gated"], n)}
    else:
        with torch.no_grad():
            e = model.get_input_embeddings()(p["input_ids"].to(device))  # [N, H]
        extra = {"inputs_embeds": diffmask.interpolate(e, z, ex["gated"], baseline)[None]}
    return masked_cite.score_copies(model, p, ex["mask"], tok, ex["k"], device, **extra)[0][0]


def main() -> None:
    """Writes <out>/probe.pt and logs running L0, KL and lam."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--adapter", default="runs/qwen3_0.6b_lam0.01/adapter")
    parser.add_argument("--data", default=data.TRAIN_EVAL)
    parser.add_argument("--variant", choices=["fade", "interp"], required=True)
    parser.add_argument("--margin", type=float, default=0.05, help="KL budget in nats")
    parser.add_argument("--steps", type=int, default=2000, help="rows, one per step")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lam-lr", type=float, default=1e-2)
    parser.add_argument("--lam-init", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=8192, help="longer prompts are skipped")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    # flex attention compiles one variant per grad mode, key requires_grad and
    # size class (~6); past torch's default limit of 8 it silently runs eager,
    # which materialises the full attention matrix (minutes per step)
    torch._dynamo.config.recompile_limit = 64
    torch._dynamo.config.accumulated_recompile_limit = 1024
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logger.add(out / "train_diffmask.log")
    logger.info("args {}", json.dumps(vars(args)))
    torch.manual_seed(args.seed)
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = modeling.load(args.model, args.adapter, device=device)
    for p in model.parameters():
        p.requires_grad_(False)
    diffmask.use_gated_attention(model)
    cfg = model.config
    probe = diffmask.GateProbe(cfg.num_hidden_layers + 1, cfg.hidden_size).to(device)
    baseline = torch.nn.Parameter(model.get_input_embeddings().weight.float().mean(0))  # interp only
    params = list(probe.parameters()) + ([baseline] if args.variant == "interp" else [])
    opt = torch.optim.Adam(params, lr=args.lr)
    hc = diffmask.HardConcrete()

    rows = [r for r in data.load_rows(args.data, "train") if len(context_cite.split_sources(r)) >= 2]
    random.Random(args.seed).shuffle(rows)
    logger.info("{} train rows with at least two sentences", len(rows))

    def save(step: int) -> None:
        torch.save({"probe": probe.state_dict(), "baseline": baseline.detach().cpu(),
                    "args": vars(args), "step": step, "lam": lam}, out / "probe.pt")

    lam, step, skipped, t0 = args.lam_init, 0, 0, time.time()
    run = {"l0": [], "kl": [], "open": []}
    for row in rows:
        if step == args.steps:
            break
        ex = prepare(tok, row, device)
        if len(ex["packed"]["input_ids"]) > args.max_tokens:
            skipped += 1
            continue
        with torch.no_grad():
            (res,), output = masked_cite.score_copies(model, ex["packed"], ex["mask"], tok, ex["k"], device,
                                                      return_output=True, output_hidden_states=True)
            states = torch.stack(output.hidden_states)[:, 0, ex["gated"]]  # [L+1, T, H] document tokens
        log_alpha = probe(states)                                          # [T]
        z = hc.sample(log_alpha)                                           # [T]
        s_masked = masked_scores(model, tok, ex, z, args.variant, baseline, device)
        parts = losses.diffmask_loss(hc.open_prob(log_alpha), res[0], s_masked, lam, args.margin)
        opt.zero_grad(set_to_none=True)
        parts["loss"].backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        kl = parts["kl"].item()
        lam = max(0.0, lam + args.lam_lr * (kl - args.margin))
        step += 1
        run["l0"].append(parts["l0"].item())
        run["kl"].append(kl)
        run["open"].append((hc.gate(log_alpha.detach()) > 0).float().mean().item())
        if step % args.log_every == 0:
            n = args.log_every
            logger.info("step {} | open prob {:.3f} | open gates {:.3f} | KL {:.3f} (margin {}) | lam {:.3f} | "
                        "{:.2f} s/step | skipped {}", step, np.mean(run["l0"][-n:]), np.mean(run["open"][-n:]),
                        np.mean(run["kl"][-n:]), args.margin, lam, (time.time() - t0) / step, skipped)
        if step % args.save_every == 0:
            save(step)
    save(step)
    logger.info("done: {} steps, {:.1f} min; wrote {}", step, (time.time() - t0) / 60, out / "probe.pt")


if __name__ == "__main__":
    main()
