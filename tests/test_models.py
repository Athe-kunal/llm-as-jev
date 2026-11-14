import random

import peft
import pytest
import torch

from llm_jev import config
from llm_jev.losses import option
from llm_jev.losses import trie
from llm_jev.models import decoder
from llm_jev.models import encoder


@pytest.fixture(scope="module")
def trie_decoder(decoder_dir) -> decoder.TrieDecoder:
    settings = config.DecoderModelConfig(id=str(decoder_dir))
    return decoder.TrieDecoder.for_inference(settings, "cpu")


@pytest.fixture(scope="module")
def masked_encoder(encoder_dir, encoder_format) -> encoder.MaskedEncoder:
    model_cfg = config.EncoderModelConfig(id=str(encoder_dir), attention="sdpa")
    return encoder.MaskedEncoder.build(
        model_cfg, encoder_format, "cpu", torch.float32, "sigmoid_norm"
    )


# --- Decoder ---------------------------------------------------------------


def test_trie_is_prefix_free_and_rejects_prefixes() -> None:
    nodes = decoder.build_trie([[1, 9], [2, 9], [1, 8]])
    assert nodes == {(): [1, 2], (1,): [8, 9], (2,): [9]}
    with pytest.raises(ValueError, match="prefix-free"):
        decoder.build_trie([[1], [1, 2]])


def test_encoding_follows_the_gold_path(trie_decoder, example) -> None:
    tok = trie_decoder.tokenizer
    ex = decoder.encode(example, tok, random.Random(0))
    suffixes, _ = decoder.candidates(tok, 3)
    assert ex.input_ids[len(ex.prompt) :] == suffixes[ex.gold]
    for position, legal, gold in ex.path:
        assert gold in legal
        assert ex.input_ids[position + 1] == gold
    text = "".join(tok.convert_ids_to_tokens(ex.prompt))
    assert text.endswith(decoder.ANSWER_PREFIX)


def _naive_scores(model, prompt, suffixes) -> torch.Tensor:
    out = []
    for s in suffixes:
        ids = torch.tensor([prompt + s])
        logp = torch.log_softmax(model(input_ids=ids).logits[0].float(), -1)
        out.append(sum(logp[len(prompt) - 1 + t, x] for t, x in enumerate(s)))
    return torch.stack(out)


def test_trie_scorers_agree_with_full_sequences(trie_decoder, example) -> None:
    model, tok = trie_decoder.network, trie_decoder.tokenizer
    prompt = decoder.prompt_ids(example, tok)
    suffixes, _ = decoder.candidates(tok, 3)
    with torch.no_grad():
        expected = _naive_scores(model, prompt, suffixes)
        batched = decoder.score_batch(model, [prompt], tok, [3], "cpu")[0]
        tree = decoder.score_tree(model, prompt, tok, 3, "cpu")
    torch.testing.assert_close(batched, expected, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(tree, expected, atol=1e-4, rtol=1e-4)
    assert len(trie_decoder.score([example, example])) == 2


def test_trie_loss_with_lora_has_zero_anchors_at_start(
    decoder_dir, example
) -> None:
    base = decoder.load_base(str(decoder_dir), dtype=torch.float32)
    model = decoder.add_lora(base, rank=4)
    tok = decoder.TrieDecoder.for_inference(
        config.DecoderModelConfig(id=str(decoder_dir)), "cpu"
    ).tokenizer
    ex = decoder.encode(example, tok, random.Random(1))
    terms = trie.compute(model, ex, "cpu", tok.pad_token_id)
    # A fresh adapter is the identity (LoRA B = 0), so p_theta = p0.
    for key in ("mass", "out", "pos"):
        assert abs(terms[key].item()) < 1e-5
    assert terms["tree"].item() > 0
    terms["tree"].backward()
    grads = [p.grad for n, p in model.named_parameters() if "lora_" in n]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)
    assert isinstance(model, peft.PeftModel)


# --- Encoder ---------------------------------------------------------------


def test_encoder_one_mask_per_option(masked_encoder, example) -> None:
    ex = masked_encoder.encode(example, random.Random(0))
    assert len(ex.masks) == 3
    assert ex.order[ex.gold] == example.gold
    tok = masked_encoder.tokenizer
    assert all(ex.input_ids[m] == tok.mask_token_id for m in ex.masks)


def test_encoder_scores_are_yes_minus_no(masked_encoder, example) -> None:
    ex = masked_encoder.encode(example)
    scorer = masked_encoder.network
    with torch.no_grad():
        hidden = scorer.hidden_states(torch.tensor([ex.input_ids]))
        logits = encoder.mask_logits(scorer.model, hidden[ex.masks])
        s = masked_encoder.score([example])[0]
    expected = logits[:, scorer.yes] - logits[:, scorer.no]
    torch.testing.assert_close(
        s, expected, atol=1e-2, rtol=1e-2
    )  # bf16 autocast


def test_encoder_loss_backpropagates(masked_encoder, example) -> None:
    out = masked_encoder.loss(example, random.Random(0))
    out.loss.backward()
    assert set(out.logs) == {"loss", "accuracy"}
    grads = [p.grad for p in masked_encoder.network.parameters()]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)
    masked_encoder.network.zero_grad()


def test_recipe_needs_a_no_token(encoder_dir) -> None:
    model_cfg = config.EncoderModelConfig(id=str(encoder_dir), attention="sdpa")
    fmt = config.FormatConfig(yes_token="Y")
    with pytest.raises(ValueError, match="no_token"):
        encoder.MaskedEncoder.build(
            model_cfg, fmt, "cpu", loss_kind="sigmoid_norm"
        )


def test_document_tokens_cover_the_part(masked_encoder, example) -> None:
    tokens = encoder.document_tokens(
        example, masked_encoder.tokenizer, masked_encoder.fmt
    )
    text = example.parts[0].text
    assert "".join(text[a:b] for _, (_, a, b) in tokens) == text


# --- Option losses ---------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "weight"),
    [
        ("listwise", torch.exp),
        ("sigmoid_norm", torch.sigmoid),
        ("sqrt_softplus", lambda s: torch.nn.functional.softplus(s).sqrt()),
    ],
)
def test_option_losses_match_their_definition(kind, weight) -> None:
    s = torch.tensor([2.0, -1.0, 0.5, -30.0])
    g = weight(s.double())
    expected = -torch.log(g[2] / g.sum())
    torch.testing.assert_close(
        option.loss(s, 2, kind).double(), expected, rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        option.probabilities(s, kind).double(),
        g / g.sum(),
        rtol=1e-5,
        atol=1e-6,
    )


def test_unknown_option_loss() -> None:
    with pytest.raises(ValueError, match="loss is one of"):
        option.loss(torch.zeros(2), 0, "hinge")
