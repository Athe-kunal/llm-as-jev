"""Typed configuration for training AT2 on an encoder run (train.train_at2).

A run is configs/at2/<name>.yaml plus dotlist overrides (data.rows=500
out=runs/x). The YAML is merged into these dataclasses, so an unknown key or a
wrongly typed value fails at startup.
"""

import dataclasses
import pathlib

from omegaconf import MISSING
from omegaconf import DictConfig
from omegaconf import OmegaConf

from llm_jev.modeling import encoder


@dataclasses.dataclass
class EncoderConfig:
    """The model whose answers AT2 attributes."""

    run: str = MISSING                 # train_mlm run directory
    attention: str = encoder.VARLEN    # capture needs FA2 varlen (encoder_at2.capturing)


@dataclasses.dataclass
class DataConfig:
    """Rows, ablations and targets (stage 1: <out>/data.npz)."""

    source: str = "data/splits/reason_long_nimble"
    split: str = "train"               # never the eval split: it is for evaluation
    datasets: list[str] | None = None  # None: every dataset (rows still need min_sources)
    rows: int = 2000                   # balanced over the datasets
    min_sources: int = 2               # ablating needs at least two sentences
    masks: int = 32                    # ablations per row
    seed: int = 0


@dataclasses.dataclass
class TrainConfig:
    """Fitting theta (stage 2), as the reference's LinearScoreEstimatorTrainer."""

    steps: int = 1000
    batch_size: int = 512
    lr: float = 1e-3
    max_grad_norm: float = 1.0
    val_frac: float = 0.1              # rows held out to report Pearson on unseen rows
    seed: int = 0


@dataclasses.dataclass
class Config:
    """Everything one AT2 run reads."""

    encoder: EncoderConfig = dataclasses.field(default_factory=EncoderConfig)
    data: DataConfig = dataclasses.field(default_factory=DataConfig)
    train: TrainConfig = dataclasses.field(default_factory=TrainConfig)
    out: str = MISSING                 # output directory


def load(path: str | pathlib.Path, overrides: list[str] = ()) -> DictConfig:
    """The YAML at `path` with dotlist overrides, validated against Config."""
    cfg = OmegaConf.merge(OmegaConf.structured(Config), OmegaConf.load(path),
                          OmegaConf.from_dotlist(list(overrides)))
    OmegaConf.to_object(cfg)  # fails here on a missing required value
    return cfg


def resolve(cfg: DictConfig) -> Config:
    """The typed Config from a loaded config."""
    return OmegaConf.to_object(cfg)
