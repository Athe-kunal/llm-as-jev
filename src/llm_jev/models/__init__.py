"""Decision models: a shared interface and its decoder and encoder families."""

import torch

from llm_jev import config
from llm_jev.models import base


def load(
    ref: config.ModelRef, device: str | None = None
) -> base.DecisionModel:
    """A decision model for inference, from a `ModelRef`.

    Args:
      ref: Which model to load.
      device: Where to put it (default: base.default_device()).

    Raises:
      ValueError: On an unknown model kind.
    """
    device = device or base.default_device()
    if ref.kind == config.DECODER:
        from llm_jev.models import decoder

        settings = config.DecoderModelConfig(id=ref.path, adapter=ref.adapter)
        return decoder.TrieDecoder.for_inference(settings, device)
    if ref.kind == config.ENCODER:
        from llm_jev.models import encoder

        return encoder.MaskedEncoder.from_run(
            ref.path, device, getattr(torch, ref.dtype), ref.attention
        )
    raise ValueError(f"model.kind is decoder or encoder, not {ref.kind!r}")
