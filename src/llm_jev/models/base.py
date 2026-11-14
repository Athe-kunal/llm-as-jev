"""The interface every decision model implements.

A decision model reads an `Example` and gives one score s_k per option; its
decision is argmax_k s_k. Training, evaluation, the benchmarks and
ContextCite only use this interface, so they work for both model families:

- `TrieDecoder` (models.decoder): a causal LM; option k is the token path
  "k]" through a candidate trie after "Best answer: [".
- `MaskedEncoder` (models.encoder): a masked LM with one mask token in front
  of every option; s_k is read from the logits at option k's mask.
"""

import abc
import dataclasses
import pathlib
import random
from collections.abc import Iterator
from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from llm_jev.data import schema


def default_device() -> str:
    """"cuda" when a GPU is visible, else "cpu"."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclasses.dataclass(frozen=True)
class LossOutput:
    """The training loss of one row.

    Attributes:
      loss: Scalar to back-propagate.
      logs: Detached scalars to log (summed over rows, then averaged).
    """

    loss: torch.Tensor
    logs: dict[str, torch.Tensor]


class DecisionModel(abc.ABC):
    """Scores the options of an `Example` with a neural network.

    Attributes:
      network: The trainable module. The trainer replaces it by the module
        Accelerate returns (DDP or FSDP), so losses must call it, not a
        submodule of it.
      tokenizer: The model's tokenizer.
      device: Where inputs are placed.
    """

    def __init__(
        self, network: nn.Module, tokenizer: Any, device: str | torch.device
    ) -> None:
        self.network = network
        self.tokenizer = tokenizer
        self.device = torch.device(device)

    @property
    def checkpoint_name(self) -> str:
        """Folder name, inside the run directory, of the saved weights."""
        return "model"

    @property
    def unwrapped(self) -> nn.Module:
        """The network without its DDP / compile wrappers."""
        from accelerate.utils import extract_model_from_parallel

        return extract_model_from_parallel(self.network)

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        """The parameters the optimizer updates."""
        return (p for p in self.network.parameters() if p.requires_grad)

    @abc.abstractmethod
    def num_tokens(self, example: schema.Example) -> int:
        """Length of the encoded row, to drop rows over a token budget."""

    @abc.abstractmethod
    def score(self, examples: Sequence[schema.Example]) -> list[torch.Tensor]:
        """Option scores s [K] (fp32, CPU or device) per example.

        Options are in release order, so argmax s is an index into
        `example.options`. Runs without gradients.
        """

    @abc.abstractmethod
    def loss(self, example: schema.Example, rng: random.Random) -> LossOutput:
        """The training loss of one row; `rng` shuffles its options."""

    @abc.abstractmethod
    def save(self, path: pathlib.Path, accelerator: Any) -> None:
        """Saves the trained weights; every process must call it."""

    def predict(self, example: schema.Example) -> int:
        """The index of the chosen option."""
        return int(self.score([example])[0].argmax())

    def train(self, mode: bool = True) -> None:
        """Sets the network's training mode."""
        self.network.train(mode)

    def eval(self) -> None:
        """Sets the network to evaluation mode."""
        self.network.eval()
