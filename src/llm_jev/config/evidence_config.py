"""Typed configuration for the evidence head (train.train_evidence), read from YAML with OmegaConf."""

import dataclasses
import pathlib

from omegaconf import MISSING
from omegaconf import DictConfig
from omegaconf import OmegaConf

from llm_jev.modeling import encoder


@dataclasses.dataclass
class EncoderConfig:
    """The frozen decision model the head reads."""

    run: str = MISSING                 # train_mlm run directory
    attention: str = encoder.VARLEN


@dataclasses.dataclass
class DataConfig:
    """Evidence labels: gold spans, and optionally generated ones where gold is missing."""

    source: str = "data/splits/reason_long_nimble"
    split: str = "train"
    labels: str = "gold+generated"     # "gold" or "gold+generated" (gold wins where both exist)
    generated_repo: str = "Athekunal/llm-jev-reason-long"   # Hub dataset with spans/train.parquet
    generated_file: str = "spans/train.parquet"
    datasets: list[str] | None = None  # None: every dataset that has labels
    val_frac: float = 0.05             # training rows held out to pick the best epoch
    eval_split: str = "eval"
    eval_per: int = 50                 # eval rows with gold spans per dataset (as attribution_mlm)
    eval_seed: int = 0


@dataclasses.dataclass
class HeadConfig:
    """The MLP over [h_t ; h_answer ; h_t * h_answer]."""

    width: int = 512
    dropout: float = 0.1


@dataclasses.dataclass
class TrainConfig:
    """AdamW over the head only (the encoder is frozen)."""

    epochs: int = 3
    rows_per_step: int = 16
    lr: float = 1e-3
    weight_decay: float = 0.01
    pos_weight: float | None = None    # None: n_negative / n_positive over the training tokens
    seed: int = 0


@dataclasses.dataclass
class Config:
    """Everything one evidence-head run reads."""

    encoder: EncoderConfig = dataclasses.field(default_factory=EncoderConfig)
    data: DataConfig = dataclasses.field(default_factory=DataConfig)
    head: HeadConfig = dataclasses.field(default_factory=HeadConfig)
    train: TrainConfig = dataclasses.field(default_factory=TrainConfig)
    out: str = MISSING


def load(path: str | pathlib.Path, overrides: list[str] = ()) -> DictConfig:
    """The YAML at `path` with dotlist overrides, validated against Config."""
    cfg = OmegaConf.merge(OmegaConf.structured(Config), OmegaConf.load(path),
                          OmegaConf.from_dotlist(list(overrides)))
    OmegaConf.to_object(cfg)
    return cfg


def resolve(cfg: DictConfig) -> Config:
    """The typed Config from a loaded config."""
    return OmegaConf.to_object(cfg)
