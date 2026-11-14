"""AT2 for encoders: attribution with attention from the option masks.

Cohen-Wang et al., "Learning to Attribute with Attention" (2025): a
sentence's score is a learned weighted sum, over (layer, head), of the
attention from the answer's positions to the sentence's tokens. For the
encoder the answer's position is its option mask, inside the prompt, so the
attention comes from the same forward pass that scores the options.

Usage (trains theta; stage 1 is cached in <out>/data.npz):
  python -m llm_jev.analysis.at2 --config configs/analysis/at2.yaml
"""

import collections
from collections.abc import Iterator
import contextlib
import json
import pathlib
from typing import Any

from loguru import logger
import numpy as np
import torch
from torch import nn

from llm_jev import config
from llm_jev.analysis import context_cite
from llm_jev.data import sampling
from llm_jev.data import schema
from llm_jev.data import storage
from llm_jev.evidence import sentences as sentences_lib
from llm_jev.models import base
from llm_jev.models import encoder
from llm_jev.models import flash_attention

CAPTURE = "at2_capture"  # Attention implementation name while capturing.


def _capturing_attention(store: dict[int, torch.Tensor]) -> Any:
    """An attention function that records softmax(q k^T) at `at2_rows`."""

    def attention(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        scaling: float | None = None,
        sliding_window: int | None = None,
        at2_rows: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        if at2_rows is not None:
            q = query[0][:, at2_rows].float()  # [H, Q, D]
            k = key[0].float()  # [Hkv, N, D]
            k = k.repeat_interleave(q.shape[0] // k.shape[0], 0)  # [H, N, D]
            s = (q @ k.transpose(1, 2)) * scaling  # [H, Q, N]
            if sliding_window is not None:
                w = flash_attention.window(sliding_window)[0]
                j = torch.arange(s.shape[-1], device=s.device)
                far = (at2_rows[:, None] - j[None]).abs() > w
                s = s.masked_fill(far, float("-inf"))
            store[module.layer_idx] = s.softmax(-1)
        return flash_attention.attention(
            module,
            query,
            key,
            value,
            attention_mask,
            scaling=scaling,
            sliding_window=sliding_window,
            **kwargs,
        )

    return attention


@contextlib.contextmanager
def capturing(
    model: nn.Module, store: dict[int, torch.Tensor]
) -> Iterator[dict[int, torch.Tensor]]:
    """Switches `model` to the capturing attention, restoring it after.

    Requires FA2 varlen: under a custom implementation transformers builds no
    attention masks, which varlen does not need but SDPA's sliding layers
    would (they would silently see everything).

    Raises:
      ValueError: If the model does not run FA2 varlen.
    """
    from transformers import AttentionInterface

    before = model.config._attn_implementation
    if before != flash_attention.NAME:
        raise ValueError(f"AT2 capture needs attention: varlen, not {before!r}")
    AttentionInterface.register(CAPTURE, _capturing_attention(store))
    model.set_attn_implementation(CAPTURE)
    try:
        yield store
    finally:
        model.set_attn_implementation(before)


@torch.no_grad()
def mask_attention(
    model: encoder.MaskedEncoder, ex: encoder.MaskEncoding
) -> tuple[torch.Tensor, torch.Tensor]:
    """Option scores s [K] and attention [L, H, K, N] from the masks."""
    scorer = model.unwrapped
    store: dict[int, torch.Tensor] = {}
    rows = torch.tensor(ex.masks, device=model.device)
    with (
        capturing(scorer.model, store),
        torch.autocast(model.device.type, dtype=torch.bfloat16),
    ):
        s = scorer(
            torch.tensor([ex.input_ids], device=model.device),
            rows,
            at2_rows=rows,
        )
    return s.float(), torch.stack([store[i] for i in sorted(store)])


def sentence_features(
    attn: torch.Tensor, ranges: list[tuple[int, int]]
) -> torch.Tensor:
    """[S, L*H]: each sentence's attention features (AT2's phi_j).

    attn [L, H, Q, N] is averaged over its Q query rows and summed over each
    sentence's tokens [start, end); feature l*H + h is layer l, head h.
    """
    per_token = attn.mean(2).flatten(0, 1).T  # [N, L*H]
    zero = per_token.new_zeros(1, per_token.shape[1])
    cum = torch.cat([zero, per_token.cumsum(0)])  # [N+1, L*H]
    return torch.stack([cum[b] - cum[a] for a, b in ranges])


class ScoreEstimator(nn.Module):
    """The reference's linear estimator: a sentence's score is theta . phi.

    One weight per (layer, head), no bias, initialised to 1/F (the
    average-attention baseline).
    """

    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(num_features, 1, bias=False)
        nn.init.constant_(self.linear.weight, 1 / num_features)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """[..., F] features -> [...] scores."""
        return self.linear(features)[..., 0]

    def theta(self) -> torch.Tensor:
        """[F] theta, L1-normalised (the Pearson loss ignores the scale)."""
        w = self.linear.weight.detach()[0]
        return w / w.abs().sum()


def pearson_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Negative Pearson correlation per row, averaged. [B, M] x2 -> []."""
    p = pred - pred.mean(1, keepdim=True)
    t = target - target.mean(1, keepdim=True)
    norm = (p**2).mean(1).sqrt() * (t**2).mean(1).sqrt() + 1e-8
    return -((p * t).mean(1) / norm).mean()


@torch.no_grad()
def attribute(
    model: encoder.MaskedEncoder, example: schema.Example, theta: torch.Tensor
) -> context_cite.Attribution:
    """AT2 on one row, one forward pass; uniform theta = average attention."""
    ex = model.encode(example)
    sentences = sentences_lib.split(example)
    s, attn = mask_attention(model, ex)
    answer = int(s.argmax())
    token_scores = theta.to(attn) @ attn[:, :, answer].flatten(0, 1)  # [N]
    _, ranges = encoder.token_ranges(
        example, model.tokenizer, model.fmt, [x.char_range for x in sentences]
    )
    cum = torch.cat([token_scores.new_zeros(1), token_scores.cumsum(0)])
    weights = np.array([float(cum[b] - cum[a]) for a, b in ranges])
    q = torch.softmax(s, 0)[answer].item()
    return context_cite.Attribution(answer, q, sentences, weights)


def training_example(
    model: encoder.MaskedEncoder,
    example: schema.Example,
    num_masks: int = 32,
    seed: int = 0,
) -> tuple[torch.Tensor, np.ndarray, int]:
    """One AT2 training row: features [M, L*H], targets [M], the answer.

    As the reference stores them, mask m's features are (2 mask_m - 1) @ phi,
    so the surrogate's prediction for ablation m is features[m] @ theta.

    Raises:
      ValueError: If the scored prompt and the sentence ranges disagree.
    """
    ex = model.encode(example)
    sentences = sentences_lib.split(example)
    ids, ranges = encoder.token_ranges(
        example, model.tokenizer, model.fmt, [x.char_range for x in sentences]
    )
    if ids != ex.input_ids:
        raise ValueError(f"{example.id}: sentence ranges and prompt differ")
    s, attn = mask_attention(model, ex)
    answer = int(s.argmax())
    phi = sentence_features(attn[:, :, answer : answer + 1], ranges)
    masks = sentences_lib.ablation_masks(num_masks, len(sentences), 0.5, seed)
    y = context_cite.ablation_targets(model, example, sentences, masks, answer)
    signed = torch.tensor(masks, device=phi.device, dtype=phi.dtype) * 2 - 1
    return signed @ phi, y, answer


def build_data(cfg: config.AT2Config, path: pathlib.Path) -> None:
    """Stage 1: writes features, targets, answers, ids and datasets."""
    model = encoder.MaskedEncoder.from_run(
        cfg.encoder.run, attention=cfg.encoder.attention
    )
    data = cfg.data
    candidates = [
        r
        for r in storage.load_examples(data.source, data.split)
        if (data.datasets is None or r.source in data.datasets)
        and len(sentences_lib.split(r)) >= data.min_sources
    ]
    rows = sampling.balanced(candidates, data.rows, data.seed)
    logger.info(
        "{} rows: {}",
        len(rows),
        dict(collections.Counter(r.source for r in rows)),
    )
    features, targets, answers = [], [], []
    for i, row in enumerate(rows, 1):
        x, y, a = training_example(model, row, data.masks, data.seed)
        features.append(x.cpu().numpy().astype(np.float16))
        targets.append(y.astype(np.float32))
        answers.append(a)
        if i % 100 == 0 or i == len(rows):
            logger.info("{}/{} rows", i, len(rows))
    np.savez(
        path,
        features=np.stack(features),
        targets=np.stack(targets),
        answers=np.array(answers),
        ids=np.array([r.id for r in rows]),
        sources=np.array([r.source for r in rows]),
        gold=np.array([r.gold for r in rows]),
    )


def fit(
    cfg: config.AT2FitConfig,
    path: pathlib.Path,
    out: pathlib.Path,
    device: str | None = None,
) -> dict[str, Any]:
    """Stage 2: fits theta on `path`, writes theta.pt, returns metrics."""
    device = device or base.default_device()
    d = np.load(path)
    x = torch.tensor(d["features"], dtype=torch.float32)  # [R, M, F]
    y = torch.tensor(d["targets"], dtype=torch.float32)  # [R, M]
    keep = y.std(1) > 0  # Pearson needs varying targets.
    x, y, sources = x[keep], y[keep], d["sources"][keep.numpy()]
    generator = torch.Generator().manual_seed(cfg.seed)
    perm = torch.randperm(len(y), generator=generator)
    n_val = int(len(y) * cfg.val_frac)
    val, train = perm[:n_val], perm[n_val:]
    x, y = x.to(device), y.to(device)
    estimator = ScoreEstimator(x.shape[2]).to(device)

    def correlation(idx: torch.Tensor) -> float:
        with torch.no_grad():
            return -pearson_loss(estimator(x[idx]), y[idx]).item()

    def by_source(idx: torch.Tensor) -> dict[str, float]:
        names = sources[idx.numpy()]
        return {
            s: correlation(idx[torch.tensor(names == s)])
            for s in sorted(set(names))
        }

    metrics: dict[str, Any] = {
        "before": {
            "train": correlation(train),
            "val": correlation(val),
            "val_by_dataset": by_source(val),
        }
    }
    optimizer = torch.optim.Adam(estimator.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.steps
    )
    batch_size = min(cfg.batch_size, len(train))
    for step in range(cfg.steps):
        batch = train[
            torch.randperm(len(train), generator=generator)[:batch_size]
        ]
        loss = pearson_loss(estimator(x[batch]), y[batch])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            estimator.parameters(), cfg.max_grad_norm
        )
        optimizer.step()
        scheduler.step()
        if step == 0 or (step + 1) % 100 == 0:
            logger.info(
                "step {}/{}: loss {:.4f}, held-out Pearson {:.4f}",
                step + 1,
                cfg.steps,
                loss.item(),
                correlation(val),
            )
    metrics["after"] = {
        "train": correlation(train),
        "val": correlation(val),
        "val_by_dataset": by_source(val),
    }
    torch.save(estimator.theta().cpu(), out / "theta.pt")
    logger.info(
        "held-out Pearson: {:.4f} (average attention) -> {:.4f} (AT2)",
        metrics["before"]["val"],
        metrics["after"]["val"],
    )
    return metrics


def main() -> None:
    """Builds the data if missing, fits theta, writes the outputs."""
    cfg, raw = config.from_command_line(config.AT2Config, __doc__)
    out = pathlib.Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    logger.add(out / "train_at2.log")
    config.save(raw, out / "config.yaml")
    data = out / "data.npz"
    if data.exists():
        logger.info("using existing {} (delete it to rebuild)", data)
    else:
        build_data(cfg, data)
        torch.cuda.empty_cache()
    metrics = fit(cfg.train, data, out)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
