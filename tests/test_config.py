import pathlib

import omegaconf
import pytest

from llm_jev import config

CONFIGS = pathlib.Path(__file__).parents[1] / "configs"
SCHEMAS = {
    "data": config.BuildDataConfig,
    "evidence/ultrabert_380m.yaml": config.EvidenceConfig,
    "evidence/spans.yaml": config.SpansConfig,
    "analysis/at2.yaml": config.AT2Config,
    "analysis/citation.yaml": config.CitationConfig,
    "analysis/coupling.yaml": config.CouplingConfig,
    "analysis/latency.yaml": config.LatencyConfig,
}
REQUIRED = ["model.kind=encoder", "model.path=runs/x", "out=runs/x"]


def _schema(path: pathlib.Path) -> type:
    rel = path.relative_to(CONFIGS).as_posix()
    if rel.startswith("train/"):
        return (
            config.EncoderTrainConfig
            if "ultrabert" in rel
            else config.DecoderTrainConfig
        )
    if rel == "evaluate/benchmarks.yaml":
        return config.BenchmarksConfig
    if rel.startswith("evaluate/"):
        return config.EvaluateConfig
    return SCHEMAS.get(rel) or SCHEMAS[rel.split("/")[0]]


@pytest.mark.parametrize(
    "path",
    [p for p in sorted(CONFIGS.rglob("*.yaml")) if "accelerate" not in p.parts],
    ids=lambda p: p.relative_to(CONFIGS).as_posix(),
)
def test_every_config_validates(path: pathlib.Path) -> None:
    schema = _schema(path)
    overrides = (
        REQUIRED
        if schema in (config.EvaluateConfig, config.BenchmarksConfig)
        else []
    )
    cfg = config.resolve(config.load(schema, path, overrides))
    assert isinstance(cfg, schema)


def test_overrides_and_interpolation() -> None:
    raw = config.load(
        config.EvaluateConfig,
        CONFIGS / "evaluate/encoder.yaml",
        ["model.path=runs/enc", "part=test"],
    )
    cfg = config.resolve(raw)
    assert cfg.out == "runs/enc"
    assert cfg.part == "test"


def test_unknown_key_fails() -> None:
    with pytest.raises(omegaconf.errors.ConfigKeyError):
        config.load(
            config.EncoderTrainConfig,
            None,
            ["optim.learning_rate=1", "model.id=x", "run.out=y"],
        )


def test_missing_value_fails() -> None:
    with pytest.raises(omegaconf.errors.MissingMandatoryValue):
        config.load(config.EncoderTrainConfig, None, [])


def test_old_encoder_run_config_still_loads(tmp_path: pathlib.Path) -> None:
    """config.yaml files saved by earlier encoder runs keep loading."""
    old = """
model: {id: dmis-lab/UltraBERT-380M, revision: null, attention: varlen, pack_tokens: 32768, gradient_checkpointing: true}
format: {template: decision_mlm, yes_token: ' yes', no_token: ' no', no_word: null}
data: {source: x, train_split: train, eval_split: eval, max_tokens: 32768, eval_per: 100, eval_seed: 20260922}
optim: {lr: 1.5e-05, betas: [0.9, 0.98], eps: 1.0e-06, weight_decay: 0.1, batch_size: 32,
        warmup_ratio: 0.05, min_lr_ratio: 0.0, max_grad_norm: 1.0, loss: sigmoid_norm}
run: {out: runs/x, seed: 1, epochs: 1, max_steps: -1, eval_every: 0, skip_base_eval: true, mixed_precision: bf16}
"""
    (tmp_path / "config.yaml").write_text(old)
    cfg = config.resolve(
        config.load(config.EncoderTrainConfig, tmp_path / "config.yaml")
    )
    assert cfg.optim.loss == "sigmoid_norm"
    assert cfg.format.no_token == " no"


def test_from_command_line() -> None:
    cfg, raw = config.from_command_line(
        config.CouplingConfig, argv=["runs=[a,b]", "shuffles=1"]
    )
    assert cfg.runs == ["a", "b"]
    assert raw.shuffles == 1
