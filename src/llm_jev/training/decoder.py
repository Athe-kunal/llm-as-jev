"""Fine-tunes a decoder as a decision model: L_tree + KL anchors.

Usage:
  accelerate launch --config_file configs/accelerate/fsdp2.yaml \
      -m llm_jev.training.decoder --config configs/train/qwen3_0.6b_lora.yaml \
      [optim.lr=5e-5 run.out=runs/x ...]

Follows Li & Wagle, arXiv 2610.02076v3 (Sec. 3.4); the default settings are
the paper's Table 2. LoRA (p0 = the adapter disabled) or full fine-tuning
(p0 = a frozen copy of the base model), see `config.DecoderModelConfig`.
"""

from llm_jev import config
from llm_jev.models import decoder
from llm_jev.training import trainer


def main() -> None:
    """Trains a decoder from a YAML config and dotlist overrides."""
    cfg, raw = config.from_command_line(config.DecoderTrainConfig, __doc__)
    trainer.launch(
        cfg, raw, lambda device: decoder.TrieDecoder.for_training(cfg, device)
    )


if __name__ == "__main__":
    main()
