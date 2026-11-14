"""A masked-language-model encoder as a decision model: one mask per option.

The prompt (llm_jev/prompts/<format.template>.j2) puts the tokenizer's mask
token in front of every option; s_k is the MLM logit of the verbalizer
(format.yes_token), minus that of format.no_token if set, at option k's
mask. One forward pass, no trie, no chat template.

Works for any transformers MLM whose forward is
`decoder(head(model(...).last_hidden_state))`: UltraBERT and ModernBERT are.
Logits are computed only at the K mask positions.
"""

import dataclasses
import pathlib
import random
from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from llm_jev import config
from llm_jev.data import prompts
from llm_jev.data import schema
from llm_jev.losses import option
from llm_jev.models import base
from llm_jev.models import flash_attention

CharRange = tuple[int, int, int]  # (part, start, end) in that part's text.


@dataclasses.dataclass(frozen=True)
class MaskEncoding:
    """One encoded row.

    Attributes:
      input_ids: Token ids, with the tokenizer's special tokens.
      masks: masks[k] is the position of the mask of shown option k.
      gold: Index of the gold option among the shown options.
      order: order[k] is the release index of shown option k.
    """

    input_ids: list[int]
    masks: list[int]
    gold: int
    order: list[int]

    @property
    def n_options(self) -> int:
        """K."""
        return len(self.order)


def _single_token(tokenizer: Any, text: str) -> int:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) != 1:
        raise ValueError(f"verbalizer {text!r} is {len(ids)} tokens: {ids}")
    return ids[0]


def verbalizer_ids(
    tokenizer: Any, fmt: config.FormatConfig
) -> tuple[int, int | None]:
    """Token ids of fmt.yes_token and fmt.no_token (None if unset)."""
    no = None if fmt.no_token is None else _single_token(tokenizer, fmt.no_token)
    return _single_token(tokenizer, fmt.yes_token), no


def instruction_words(fmt: config.FormatConfig) -> tuple[str, str]:
    """The (positive, negative) words the instruction names, unpadded."""
    negative = fmt.no_word or fmt.no_token or " no"
    return fmt.yes_token.strip(), negative.strip()


def build_prompt(
    example: schema.Example,
    order: Sequence[int],
    tokenizer: Any,
    fmt: config.FormatConfig,
) -> tuple[str, list[tuple[int, int]]]:
    """Prompt text with one mask per option, and each part's character range.

    Raises:
      ValueError: If the tokenizer has no mask token.
    """
    if tokenizer.mask_token is None:
        raise ValueError("the tokenizer has no mask token")
    yes_word, no_word = instruction_words(fmt)
    return prompts.render_example(
        example,
        order,
        fmt.template,
        mask=tokenizer.mask_token,
        yes_word=yes_word,
        no_word=no_word,
    )


def encode_order(
    example: schema.Example,
    order: Sequence[int],
    tokenizer: Any,
    fmt: config.FormatConfig,
) -> MaskEncoding:
    """Encodes a row with its options shown in `order`.

    Raises:
      ValueError: If the prompt does not hold exactly one mask per option.
    """
    text, _ = build_prompt(example, order, tokenizer, fmt)
    ids = tokenizer(text)["input_ids"]
    masks = [i for i, t in enumerate(ids) if t == tokenizer.mask_token_id]
    if len(masks) != len(order):
        raise ValueError(
            f"{example.id}: {len(masks)} masks for {len(order)} options "
            f"(template {fmt.template!r})"
        )
    return MaskEncoding(ids, masks, list(order).index(example.gold), list(order))


def encode(
    example: schema.Example,
    tokenizer: Any,
    fmt: config.FormatConfig,
    rng: random.Random | None = None,
) -> MaskEncoding:
    """Encodes a row; with `rng` (training) its options are shuffled."""
    order = list(range(len(example.options)))
    if rng is not None:
        rng.shuffle(order)
    return encode_order(example, order, tokenizer, fmt)


def document_tokens(
    example: schema.Example, tokenizer: Any, fmt: config.FormatConfig
) -> list[tuple[int, CharRange]]:
    """(token index, (part, start, end)) of every prompt token in a part.

    Options in release order. Offsets are character offsets in that part
    (the same offsets as spans), clipped to the part; template, question,
    option and special tokens are left out.
    """
    order = range(len(example.options))
    text, where = build_prompt(example, order, tokenizer, fmt)
    offsets = tokenizer(text, return_offsets_mapping=True)["offset_mapping"]
    out = []
    for t, (a, b) in enumerate(offsets):
        if a == b:
            continue
        for p, (lo, hi) in enumerate(where):
            if a < hi and b > lo:
                out.append((t, (p, max(a, lo) - lo, min(b, hi) - lo)))
                break
    return out


def token_ranges(
    example: schema.Example,
    tokenizer: Any,
    fmt: config.FormatConfig,
    ranges: Sequence[CharRange],
) -> tuple[list[int], list[tuple[int, int]]]:
    """The prompt's token ids, and each character range's [start, end) tokens.

    Options in release order. A token that straddles a boundary belongs to
    both sides; special tokens (offset (0, 0)) belong to none.
    """
    order = range(len(example.options))
    text, where = build_prompt(example, order, tokenizer, fmt)
    enc = tokenizer(text, return_offsets_mapping=True)
    starts = torch.tensor([a for a, _ in enc["offset_mapping"]])
    ends = torch.tensor([b for _, b in enc["offset_mapping"]])
    out = []
    for part, start, end in ranges:
        lo, hi = where[part][0] + start, where[part][0] + end
        first = int(torch.searchsorted(ends, lo, right=True))
        stop = int(torch.searchsorted(starts, hi, right=False))
        out.append((first, stop))
    return enc["input_ids"], out


def _fp32(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """module(x) in fp32, with autocast off; gradients reach the weights."""
    params = {n: p.float() for n, p in module.named_parameters()}
    with torch.autocast(x.device.type, enabled=False):
        return torch.func.functional_call(module, params, (x.float(),))


def mask_logits(model: nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    """Vocabulary logits [K, V] in fp32 from final hidden states [K, d].

    The head runs in fp32: MLM logits here are ~200 in size, where bf16
    steps by 1-2, which ties options and can change the argmax.
    """
    return _fp32(model.decoder, _fp32(model.head, hidden))


class Scorer(nn.Module):
    """The encoder and its verbalizer: input ids and mask positions -> s [K].

    The module to train and to hand to Accelerate: scoring inside forward()
    keeps DDP's gradient sync and mixed-precision autocast.
    """

    def __init__(self, model: nn.Module, yes: int, no: int | None) -> None:
        super().__init__()
        self.model = model
        self.yes = yes
        self.no = no

    def hidden_states(
        self, input_ids: torch.Tensor, **model_kwargs: Any
    ) -> torch.Tensor:
        """Final hidden states [N, d] of a single (packed) sequence [1, N]."""
        out = self.model.model(input_ids=input_ids, **model_kwargs)
        return out.last_hidden_state[0]

    def scores(self, hidden: torch.Tensor) -> torch.Tensor:
        """s [K] in fp32 from the hidden states at the masks [K, d]."""
        logits = mask_logits(self.model, hidden)
        s = logits[:, self.yes]
        return s if self.no is None else s - logits[:, self.no]

    def forward(
        self, input_ids: torch.Tensor, masks: torch.Tensor, **model_kwargs: Any
    ) -> torch.Tensor:
        """input_ids [1, N], masks [K] -> s [K] in fp32.

        `model_kwargs` reach the attention function (position_ids,
        cu_seqlens, max_seqlen for packed rows, see flash_attention).
        """
        return self.scores(self.hidden_states(input_ids, **model_kwargs)[masks])


def load_mlm(
    model_id: str,
    revision: str | None = None,
    attention: str = "sdpa",
    dtype: torch.dtype = torch.bfloat16,
) -> nn.Module:
    """The encoder with its MLM head, on the CPU in eval mode.

    Raises:
      ValueError: If the model is not decoder(head(model(...))).
    """
    import transformers

    model = transformers.AutoModelForMaskedLM.from_pretrained(
        model_id,
        revision=revision,
        dtype=dtype,
        attn_implementation="sdpa" if attention == config.VARLEN else attention,
    )
    if attention == config.VARLEN:
        flash_attention.use(model)
    for name in ("model", "head", "decoder"):
        if not hasattr(model, name):
            raise ValueError(
                f"{type(model).__name__} has no .{name}; expected "
                "decoder(head(model(...)))"
            )
    return model.eval()


def run_config(run: str | pathlib.Path) -> config.EncoderTrainConfig:
    """The resolved config an encoder run was trained with."""
    raw = config.load(
        config.EncoderTrainConfig, pathlib.Path(run) / "config.yaml"
    )
    return config.resolve(raw)  # type: ignore[return-value]


class MaskedEncoder(base.DecisionModel):
    """An MLM encoder that answers by one mask per option.

    Attributes:
      fmt: The prompt format.
      loss_kind: The training recipe (llm_jev.losses.option.LOSSES).
      pack_tokens: Tokens per packed forward pass (FA2 varlen only).
    """

    network: Scorer

    def __init__(
        self,
        network: Scorer,
        tokenizer: Any,
        device: str | torch.device,
        fmt: config.FormatConfig,
        loss_kind: str = "listwise",
        pack_tokens: int = 32768,
    ) -> None:
        super().__init__(network, tokenizer, device)
        self.fmt = fmt
        self.loss_kind = loss_kind
        self.pack_tokens = pack_tokens

    @classmethod
    def build(
        cls,
        model_cfg: config.EncoderModelConfig,
        fmt: config.FormatConfig,
        device: str | torch.device,
        dtype: torch.dtype = torch.float32,
        loss_kind: str = "listwise",
        tokenizer_path: str | None = None,
    ) -> "MaskedEncoder":
        """Loads the encoder named by `model_cfg`.

        Raises:
          ValueError: If the recipe needs a "no" token the format lacks.
        """
        import transformers

        if loss_kind in option.NEEDS_NO_TOKEN and fmt.no_token is None:
            raise ValueError(
                f"loss {loss_kind} needs format.no_token (e.g. ' no'): it is "
                "applied to logit(yes) - logit(no), not to logit(yes) alone"
            )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            tokenizer_path or model_cfg.id, revision=model_cfg.revision
        )
        model = load_mlm(model_cfg.id, model_cfg.revision, model_cfg.attention, dtype)
        scorer = Scorer(model, *verbalizer_ids(tokenizer, fmt)).to(device)
        return cls(scorer, tokenizer, device, fmt, loss_kind, model_cfg.pack_tokens)

    @classmethod
    def from_run(
        cls,
        run: str | pathlib.Path,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        attention: str | None = None,
    ) -> "MaskedEncoder":
        """A trained run: <run>/model and the format in <run>/config.yaml.

        fp32 weights (scored under bf16 autocast) by default, as in training:
        bf16 weights also round the residual stream, which over long
        documents flips borderline decisions.
        """
        device = device or base.default_device()
        cfg = run_config(run)
        model_cfg = dataclasses.replace(
            cfg.model,
            id=str(pathlib.Path(run) / "model"),
            revision=None,
            attention=attention or cfg.model.attention,
        )
        encoder = cls.build(model_cfg, cfg.format, device, dtype, cfg.optim.loss)
        encoder.eval()
        return encoder

    @property
    def is_packed(self) -> bool:
        """Whether rows are scored in packs (FA2 varlen)."""
        return flash_attention.is_used(self.unwrapped.model)

    def encode(
        self, example: schema.Example, rng: random.Random | None = None
    ) -> MaskEncoding:
        """`encode` with this model's tokenizer and format."""
        return encode(example, self.tokenizer, self.fmt, rng)

    def num_tokens(self, example: schema.Example) -> int:
        """Prompt tokens."""
        return len(self.encode(example).input_ids)

    @torch.no_grad()
    def score_encoded(
        self, encoded: Sequence[MaskEncoding]
    ) -> list[torch.Tensor]:
        """s [K] per encoded row (in its shown order), packed on FA2 varlen."""
        scorer = self.unwrapped
        lengths = [len(e.input_ids) for e in encoded]
        batches = (
            flash_attention.groups(lengths, self.pack_tokens)
            if self.is_packed
            else [[i] for i in range(len(encoded))]
        )
        out: list[torch.Tensor] = [torch.empty(0)] * len(encoded)
        for batch in batches:
            with torch.autocast(self.device.type, dtype=torch.bfloat16):
                packed = flash_attention.pack(
                    [(encoded[i].input_ids, encoded[i].masks) for i in batch],
                    self.device,
                )
                kwargs = packed.model_kwargs() if self.is_packed else {}
                s = scorer(packed.input_ids, packed.masks, **kwargs)
            for i, row in zip(batch, s.split(packed.splits), strict=True):
                out[i] = row.float().cpu()
        return out

    def score(self, examples: Sequence[schema.Example]) -> list[torch.Tensor]:
        """s [K] per example, options in release order."""
        return self.score_encoded([self.encode(e) for e in examples])

    @torch.no_grad()
    def forward_row(
        self, example: schema.Example
    ) -> tuple[MaskEncoding, torch.Tensor, torch.Tensor]:
        """One forward: (encoding, hidden states [N, d] fp32, s [K])."""
        ex = self.encode(example)
        ids = torch.tensor([ex.input_ids], device=self.device)
        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            hidden = self.unwrapped.hidden_states(ids)
            s = self.unwrapped.scores(hidden[ex.masks])
        return ex, hidden.float(), s.float()

    def loss(self, example: schema.Example, rng: random.Random) -> base.LossOutput:
        """The recipe's -log p_gold; logs it and whether argmax s is gold."""
        ex = self.encode(example, rng)
        s = self.network(
            torch.tensor([ex.input_ids], device=self.device),
            torch.tensor(ex.masks, device=self.device),
        )
        value = option.loss(s, ex.gold, self.loss_kind)
        hit = (s.argmax() == ex.gold).float()
        return base.LossOutput(value, {"loss": value.detach(), "accuracy": hit})

    def save(self, path: pathlib.Path, accelerator: Any) -> None:
        """The encoder weights and tokenizer (fp32, as trained)."""
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            accelerator.unwrap_model(self.network).model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
