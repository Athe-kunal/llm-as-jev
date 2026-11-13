"""Typed configuration for encoder (MLM) training, read from YAML with OmegaConf.

A run is configs/train_mlm/<name>.yaml plus optional dotlist overrides
(`optim.lr=1e-5 run.out=runs/x`). The YAML is merged into the dataclasses
below, so an unknown key or a wrongly typed value fails at startup. Prompt
knobs reuse dataloader.mlm.Format.
"""

import dataclasses
import pathlib

from omegaconf import MISSING
from omegaconf import DictConfig
from omegaconf import OmegaConf

from llm_jev.dataloader import mlm
from llm_jev.evaluation import evaluate
from llm_jev.modeling import encoder


@dataclasses.dataclass
class ModelConfig:
    """The encoder."""

    id: str = MISSING                  # Hub id or local path
    revision: str | None = None        # Hub commit (None: latest)
    attention: str = encoder.VARLEN    # "varlen" (FA2 varlen: no masks, packed evaluation),
                                       # or a transformers attn_implementation ("sdpa", ...)
    pack_tokens: int = 32768           # tokens per packed forward (varlen evaluation)
    gradient_checkpointing: bool = True


@dataclasses.dataclass
class DataConfig:
    """Training and evaluation rows."""

    source: str = "Athekunal/llm-jev-reason-long"  # Hub dataset id or local folder
    train_split: str = "train"
    eval_split: str = "eval"
    max_tokens: int = 32768            # longer encoded rows are dropped
    eval_per: int = 100                # eval rows per source dataset
    eval_seed: int = evaluate.SEED     # picks the eval rows; separate from run.seed so
                                       # seed replicates are scored on the same rows


@dataclasses.dataclass
class OptimConfig:
    """AdamW with linear warmup, then linear decay."""

    lr: float = 8e-6                   # peak
    betas: list[float] = dataclasses.field(default_factory=lambda: [0.9, 0.98])
    eps: float = 1e-6
    weight_decay: float = 0.1
    batch_size: int = 32               # rows per optimizer step, over all GPUs
    warmup_ratio: float = 0.05         # share of the steps
    min_lr_ratio: float = 0.0          # final lr / peak
    max_grad_norm: float = 1.0
    loss: str = "listwise"             # encoder.LOSSES: "listwise" (options compete) or
                                       # "sigmoid_norm" (sigmoids normalised by their sum, as DeepSeek-V3's
                                       # gate; needs format.no_token) or "sqrt_softplus" (sqrt(softplus)
                                       # normalised by the sum; needs format.no_token)


@dataclasses.dataclass
class RunConfig:
    """Output, length and evaluation schedule."""

    out: str = MISSING                 # run directory
    seed: int = evaluate.SEED
    epochs: int = 1
    max_steps: int = -1                # > 0: stop early (smoke tests)
    eval_every: int = 0                # 0: evaluate before and after only
    skip_base_eval: bool = False
    mixed_precision: str = "bf16"      # autocast over fp32 weights


@dataclasses.dataclass
class Config:
    """Everything one encoder training run reads."""

    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    format: mlm.Format = dataclasses.field(default_factory=mlm.Format)
    data: DataConfig = dataclasses.field(default_factory=DataConfig)
    optim: OptimConfig = dataclasses.field(default_factory=OptimConfig)
    run: RunConfig = dataclasses.field(default_factory=RunConfig)


def load(path: str | pathlib.Path, overrides: list[str] = ()) -> DictConfig:
    """The YAML at `path` with dotlist overrides, validated against Config.

    Raises:
      omegaconf errors: On an unknown key, a wrong type or a missing value.
    """
    schema = OmegaConf.structured(Config)
    OmegaConf.set_readonly(schema, False)  # mlm.Format is frozen; allow the merge
    cfg = OmegaConf.merge(schema, OmegaConf.load(path),
                          OmegaConf.from_dotlist(list(overrides)))
    OmegaConf.to_object(cfg)  # fails here on a missing required value
    return cfg


def resolve(cfg: DictConfig) -> Config:
    """The typed Config (with an mlm.Format instance) from a loaded config."""
    return OmegaConf.to_object(cfg)
