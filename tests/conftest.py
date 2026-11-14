"""Tiny offline models and tokenizers, so the tests run on a CPU without the Hub."""

import pathlib

import pytest
from tokenizers import Tokenizer
from tokenizers import models as tok_models
from tokenizers import pre_tokenizers
import torch
import transformers

from llm_jev import config
from llm_jev.data import schema

CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}"
    "<|im_end|>\n{% endfor %}{% if add_generation_prompt %}"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n{% endif %}"
)
SPECIAL = [
    "<pad>",
    "<unk>",
    "<mask>",
    "<cls>",
    "<sep>",
    "<|im_start|>",
    "<|im_end|>",
]


def _byte_tokenizer(**special: str) -> transformers.PreTrainedTokenizerFast:
    """A character-level tokenizer: every printable ASCII char is a token."""
    chars = [chr(i) for i in range(32, 127)] + ["\n"]
    vocab = {t: i for i, t in enumerate(SPECIAL + chars)}
    tok = Tokenizer(tok_models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
    tok.add_special_tokens(SPECIAL)
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="<unk>", **special
    )


@pytest.fixture(scope="session")
def example() -> schema.Example:
    return schema.Example(
        id="toy:1",
        source="toy",
        parts=(
            schema.Part(
                "Passage",
                "The sky is blue today. Grass grows in the field. Cats sleep a lot.",
            ),
        ),
        question="What colour is the sky?",
        options=("Blue", "Red", "Green"),
        gold=0,
        spans=(schema.Span(0, 0, 22),),
    )


@pytest.fixture(scope="session")
def decoder_tokenizer() -> transformers.PreTrainedTokenizerFast:
    tok = _byte_tokenizer(pad_token="<pad>", eos_token="<|im_end|>")
    tok.chat_template = CHAT_TEMPLATE
    return tok


@pytest.fixture(scope="session")
def decoder_dir(
    tmp_path_factory: pytest.TempPathFactory, decoder_tokenizer
) -> pathlib.Path:
    """A random two-layer Qwen3 saved with the toy tokenizer."""
    path = tmp_path_factory.mktemp("qwen3")
    cfg = transformers.Qwen3Config(
        vocab_size=len(decoder_tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        max_position_embeddings=2048,
        pad_token_id=0,
    )
    torch.manual_seed(0)
    transformers.Qwen3ForCausalLM(cfg).save_pretrained(path)
    decoder_tokenizer.save_pretrained(path)
    return path


@pytest.fixture(scope="session")
def encoder_dir(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """A random two-layer ModernBERT MLM saved with a toy tokenizer."""
    path = tmp_path_factory.mktemp("modernbert")
    tok = _byte_tokenizer(
        pad_token="<pad>",
        mask_token="<mask>",
        cls_token="<cls>",
        sep_token="<sep>",
    )
    # Single-token verbalizers: " y" / " n" are two chars, so use "Y" / "N".
    cfg = transformers.ModernBertConfig(
        vocab_size=len(tok),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=2048,
        pad_token_id=0,
        cls_token_id=3,
        sep_token_id=4,
        bos_token_id=3,
        eos_token_id=4,
    )
    torch.manual_seed(0)
    transformers.ModernBertForMaskedLM(cfg).save_pretrained(path)
    tok.save_pretrained(path)
    return path


@pytest.fixture(scope="session")
def encoder_format() -> config.FormatConfig:
    return config.FormatConfig(
        template="decision_mlm", yes_token="Y", no_token="N"
    )
