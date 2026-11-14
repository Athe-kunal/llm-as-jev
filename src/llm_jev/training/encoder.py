"""Fine-tunes a masked-language-model encoder as a decision model.

Usage:
  accelerate launch --config_file configs/accelerate/ddp.yaml \
      -m llm_jev.training.encoder \
      --config configs/train/ultrabert_380m_sigmoid_norm.yaml \
      [optim.lr=1e-5 run.out=runs/x ...]

One mask per option (models.encoder), one of the recipes in
losses.option, full fine-tuning with fp32 weights under bf16 autocast.
"""

import torch

from llm_jev import config
from llm_jev.models import encoder
from llm_jev.training import trainer


def build(
    cfg: config.EncoderTrainConfig, device: torch.device
) -> encoder.MaskedEncoder:
    """The encoder to train, with fp32 weights (bf16 rounds away updates)."""
    model = encoder.MaskedEncoder.build(
        cfg.model, cfg.format, device, torch.float32, cfg.optim.loss
    )
    if cfg.model.gradient_checkpointing:
        model.network.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    return model


def main() -> None:
    """Trains an encoder from a YAML config and dotlist overrides."""
    cfg, raw = config.from_command_line(config.EncoderTrainConfig, __doc__)
    trainer.launch(cfg, raw, lambda device: build(cfg, device))


if __name__ == "__main__":
    main()
