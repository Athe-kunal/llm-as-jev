"""One training loop for every decision model, on Accelerate (DDP or FSDP2).

The loop only uses the `DecisionModel` interface: each process encodes its
share of a batch row by row (`model.loss`), gradients accumulate without
synchronisation until the process's last row, then they are clipped and
AdamW steps. Evaluation runs before training, every `run.eval_every` steps
and at the end (eval_<tag>.json, TensorBoard under <run.out>/tb).
"""

from collections.abc import Callable
from collections.abc import Sequence
import contextlib
import math
import pathlib
import random
import time
from typing import Any

from loguru import logger
from omegaconf import DictConfig
import torch
from torch.optim import lr_scheduler

from llm_jev import config
from llm_jev.data import sampling
from llm_jev.data import schema
from llm_jev.data import storage
from llm_jev.evaluation import evaluate
from llm_jev.models import base


def row_rng(seed: int, epoch: int, index: int) -> random.Random:
    """The option-shuffling rng of training row `index` in an epoch."""
    return random.Random(f"{seed}:{epoch}:{index}")


def visiting_order(
    rows: Sequence[int], seed: int, epochs: int, batch_size: int
) -> list[tuple[int, int]]:
    """(epoch, row) pairs, reshuffled every epoch, whole batches only."""
    per_epoch = len(rows) // batch_size
    order: list[tuple[int, int]] = []
    for epoch in range(epochs):
        shuffled = list(rows)
        random.Random(f"{seed}:order:{epoch}").shuffle(shuffled)
        order += [(epoch, i) for i in shuffled[: per_epoch * batch_size]]
    return order


def linear_schedule(
    optimizer: torch.optim.Optimizer,
    total: int,
    warmup: int,
    min_lr_ratio: float,
) -> lr_scheduler.LRScheduler:
    """Linear warmup over `warmup` steps, then linear decay to the floor."""
    warmup = max(1, min(warmup, total))
    return lr_scheduler.SequentialLR(
        optimizer,
        [
            lr_scheduler.LinearLR(
                optimizer,
                start_factor=1 / warmup,
                end_factor=1.0,
                total_iters=max(1, warmup - 1),
            ),
            lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=min_lr_ratio,
                total_iters=max(1, total - warmup),
            ),
        ],
        milestones=[warmup],
    )


class Trainer:
    """Trains a decision model on Accelerate.

    Attributes:
      data: Which rows to train and evaluate on.
      optim: Optimizer and schedule.
      run: Output, length and evaluation schedule.
      out: The run directory.
      accelerator: The Accelerate state.
    """

    def __init__(
        self,
        data: config.DataConfig,
        optim: config.OptimConfig,
        run: config.RunConfig,
        raw_config: DictConfig,
    ) -> None:
        from accelerate import Accelerator
        from accelerate.utils import set_seed

        self.data, self.optim, self.run = data, optim, run
        self.out = pathlib.Path(run.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.accelerator = Accelerator(
            mixed_precision=run.mixed_precision,
            log_with="tensorboard",
            project_dir=self.out,
        )
        if optim.batch_size % self.accelerator.num_processes:
            raise ValueError(
                f"optim.batch_size {optim.batch_size} is not divisible by "
                f"{self.accelerator.num_processes} processes"
            )
        set_seed(run.seed)
        if self.accelerator.is_main_process:
            logger.add(self.out / "train.log")
            config.save(raw_config, self.out / "config.yaml")
        else:
            logger.remove()
        self.accelerator.init_trackers("tb")

    @property
    def device(self) -> torch.device:
        """This process's device."""
        return self.accelerator.device

    def fitting_rows(
        self, model: base.DecisionModel, rows: Sequence[schema.Example]
    ) -> list[int]:
        """Indices of the rows of at most `data.max_tokens` tokens."""
        from accelerate.utils import gather_object

        acc = self.accelerator
        mine = [
            (i, model.num_tokens(rows[i]))
            for i in range(acc.process_index, len(rows), acc.num_processes)
        ]
        lengths = dict(gather_object(mine))
        keep = [
            i for i in range(len(rows)) if lengths[i] <= self.data.max_tokens
        ]
        logger.info(
            "{} of {} training rows are over {} tokens, dropped",
            len(rows) - len(keep),
            len(rows),
            self.data.max_tokens,
        )
        return keep

    def _evaluate(
        self,
        model: base.DecisionModel,
        rows: Sequence[schema.Example],
        step: int,
        tag: str,
    ) -> None:
        start = time.time()
        metrics = evaluate.evaluate_distributed(model, rows, self.accelerator)
        if self.accelerator.is_main_process:
            evaluate.save(self.out, tag, metrics)
        self.accelerator.log(
            {
                f"eval/{name}/{key}": value
                for name, values in metrics.items()
                for key, value in values.items()
            },
            step=step,
        )
        evaluate.log_summary(tag, metrics)
        logger.info(
            "eval {} at step {} took {:.1f}m",
            tag,
            step,
            (time.time() - start) / 60,
        )

    def _grad_norm(self, model: base.DecisionModel) -> float:
        norm = self.accelerator.clip_grad_norm_(
            model.network.parameters(), self.optim.max_grad_norm
        )
        if hasattr(norm, "full_tensor"):  # FSDP2 returns a DTensor.
            norm = norm.full_tensor()
        return float(norm)

    def fit(
        self, build_model: Callable[[torch.device], base.DecisionModel]
    ) -> None:
        """Builds the model on this process's device and trains it.

        Args:
          build_model: Returns the model to train, given the device.

        Raises:
          ValueError: If there is not one full batch of training rows.
        """
        acc, optim, run = self.accelerator, self.optim, self.run
        model = build_model(self.device)
        train = storage.load_examples(self.data.source, self.data.train_split)
        rows = self.fitting_rows(model, train)
        evals = sampling.per_dataset(
            storage.load_examples(self.data.source, self.data.eval_split),
            self.data.eval_per,
            self.data.eval_seed,
        )
        params = list(model.trainable_parameters())
        logger.info(
            "trainable parameters: {:,} ({})",
            sum(p.numel() for p in params),
            params[0].dtype,
        )
        optimizer = torch.optim.AdamW(
            params,
            lr=optim.lr,
            betas=tuple(optim.betas),
            eps=optim.eps,
            weight_decay=optim.weight_decay,
        )
        order = visiting_order(rows, run.seed, run.epochs, optim.batch_size)
        total = len(order) // optim.batch_size
        if run.max_steps > 0:
            total = min(total, run.max_steps)
        if total < 1:
            raise ValueError(
                f"{len(rows)} training rows are fewer than one batch of "
                f"{optim.batch_size}"
            )
        warmup = (
            optim.warmup_steps
            if optim.warmup_steps is not None
            else round(optim.warmup_ratio * total)
        )
        scheduler = linear_schedule(
            optimizer, total, warmup, optim.min_lr_ratio
        )
        model.network, optimizer = acc.prepare(model.network, optimizer)

        if not run.skip_base_eval:
            self._evaluate(model, evals, 0, "base")
        model.train()
        start, skipped = time.time(), 0
        for step in range(total):
            batch = order[
                step * optim.batch_size : (step + 1) * optim.batch_size
            ]
            mine = batch[acc.process_index :: acc.num_processes]
            sums: dict[str, torch.Tensor] = {}
            for j, (epoch, i) in enumerate(mine):
                last = j == len(mine) - 1
                sync = (
                    contextlib.nullcontext()
                    if last
                    else acc.no_sync(model.network)
                )
                with sync:
                    out = model.loss(train[i], row_rng(run.seed, epoch, i))
                    acc.backward(out.loss / len(mine))
                for key, value in out.logs.items():
                    sums[key] = sums.get(key, 0) + value.float()
            keys = sorted(sums)
            means = acc.reduce(torch.stack([sums[k] for k in keys]), "sum")
            logs = dict(
                zip(keys, (means / optim.batch_size).tolist(), strict=True)
            )
            norm = self._grad_norm(model)
            if math.isfinite(norm):
                optimizer.step()
            else:
                skipped += 1
                logger.warning(
                    "step {}: skipped update, grad_norm {}", step + 1, norm
                )
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            lr = scheduler.get_last_lr()[0]
            acc.log(
                {
                    **{f"train/{k}": v for k, v in logs.items()},
                    "train/grad_norm": norm,
                    "train/lr": lr,
                },
                step=step + 1,
            )
            if (step + 1) % 5 == 0 or step == 0:
                elapsed = time.time() - start
                logger.info(
                    "step {}/{} {} grad_norm {:.3f} lr {:.2e} skipped {} "
                    "elapsed {:.1f}m eta {:.0f}m",
                    step + 1,
                    total,
                    " ".join(f"{k} {v:.4f}" for k, v in logs.items()),
                    norm,
                    lr,
                    skipped,
                    elapsed / 60,
                    elapsed / (step + 1) * (total - step - 1) / 60,
                )
            if (
                run.eval_every
                and (step + 1) % run.eval_every == 0
                and step + 1 < total
            ):
                checkpoint = self.out / "checkpoints" / f"step-{step + 1:04d}"
                model.save(checkpoint, acc)
                self._evaluate(model, evals, step + 1, f"step{step + 1:04d}")
        model.save(self.out / model.checkpoint_name, acc)
        self._evaluate(model, evals, total, "final")
        logger.info("done")
        acc.end_training()


def launch(
    cfg: Any,
    raw: DictConfig,
    build_model: Callable[[torch.device], base.DecisionModel],
) -> None:
    """Trains `build_model(device)` with the data, optim and run of `cfg`."""
    Trainer(cfg.data, cfg.optim, cfg.run, raw).fit(build_model)
