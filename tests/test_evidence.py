import numpy as np
import pytest
import torch

from llm_jev import config
from llm_jev.analysis import context_cite
from llm_jev.data import schema
from llm_jev.evidence import head as head_lib
from llm_jev.evidence import metrics
from llm_jev.evidence import sentences
from llm_jev.models import encoder

nltk_punkt = pytest.importorskip("nltk")


@pytest.fixture(scope="module")
def sents(example):
    try:
        return sentences.split(example)
    except LookupError:  # pragma: no cover - offline without punkt_tab
        pytest.skip("nltk punkt_tab is not available offline")


def test_split_and_ablate_round_trip(example, sents) -> None:
    assert len(sents) == 3
    assert sentences.ablate(example, sents, [True] * 3).parts == example.parts
    dropped = (
        sentences.ablate(example, sents, [True, False, True]).parts[0].text
    )
    assert "Grass" not in dropped and "Cats" in dropped


def test_has_content() -> None:
    assert sentences.has_content("The sky is blue.")
    assert not sentences.has_content("[ ] ____ :")


def test_span_metrics(example, sents) -> None:
    perfect = metrics.span_metrics(example, sents, [3.0, 1.0, 0.5])
    assert perfect["ap"] == 1.0 and perfect["top1"] == 1.0
    worst = metrics.span_metrics(example, sents, [0.0, 1.0, 2.0])
    assert worst["ap"] == pytest.approx(1 / 3) and worst["top1"] == 0.0
    no_gold = schema.Example("x", "s", example.parts, "q?", ("a", "b"), 0)
    assert metrics.span_metrics(no_gold, sents, [1, 2, 3]) is None


def test_word_scores_take_the_max_token() -> None:
    words = metrics.word_scores(
        "ab cd", [(0, 1, 0.1), (1, 2, 0.7), (3, 5, 0.2)]
    )
    assert words == [(0, 2, 0.7), (3, 5, 0.2)]


def test_ablation_masks_are_seeded() -> None:
    a = sentences.ablation_masks(4, 6, seed=3)
    np.testing.assert_array_equal(a, sentences.ablation_masks(4, 6, seed=3))
    assert a.shape == (4, 6) and a.dtype == bool


@pytest.fixture(scope="module")
def model(encoder_dir, encoder_format) -> encoder.MaskedEncoder:
    model_cfg = config.EncoderModelConfig(id=str(encoder_dir), attention="sdpa")
    return encoder.MaskedEncoder.build(model_cfg, encoder_format, "cpu")


def test_context_cite_on_an_encoder(model, example, sents) -> None:
    result = context_cite.attribute(model, example, num_ablations=16)
    assert result.answer == model.predict(example)
    assert len(result.weights) == len(sents) == len(result.sentences)


def test_logit_q_is_stable() -> None:
    s = torch.tensor([50.0, 0.0, -1.0])
    assert context_cite.logit_q(s, 0) == pytest.approx(
        50 - np.log(1 + np.exp(-1)), rel=1e-6
    )


def test_evidence_head_predicts_every_document_token(model, example) -> None:
    hidden = model.unwrapped.model.config.hidden_size
    head = head_lib.EvidenceHead(hidden, width=8).eval()
    pred = head_lib.predict(model, head, example)
    labelled = head_lib.document_tokens(model, example, example.spans)
    assert (
        len(pred.probabilities)
        == len(pred.tokens.index)
        == len(labelled.labels)
    )
    assert 0 < labelled.labels.sum() < len(labelled.labels)
    assert ((pred.probabilities >= 0) & (pred.probabilities <= 1)).all()
