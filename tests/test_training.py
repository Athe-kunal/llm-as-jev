"""End-to-end smoke tests: two training steps of each family on a CPU."""

import json
import pathlib
import shutil

import pytest
import torch

from llm_jev import config
from llm_jev.analysis import coupling
from llm_jev.data import schema
from llm_jev.data import storage
from llm_jev.evaluation import evaluate
from llm_jev.evidence import train as evidence_train
from llm_jev.models import decoder
from llm_jev.models import encoder
from llm_jev.training import trainer


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory, example) -> pathlib.Path:
    path = tmp_path_factory.mktemp("data")
    rows = [
        schema.Example(
            id=f"toy:{i}",
            source=f"toy{i % 2}",
            parts=example.parts,
            question=example.question,
            options=example.options,
            gold=i % 3,
            spans=example.spans,
        )
        for i in range(8)
    ]
    for split in ("train", "eval"):
        storage.save(storage.to_dataset(rows), path / f"{split}.parquet")
    return path


def _common(data_dir: pathlib.Path, out: pathlib.Path) -> dict:
    return {
        "data.source": str(data_dir),
        "data.eval_per": 2,
        "data.max_tokens": 100000,
        "optim.batch_size": 2,
        "optim.lr": 1e-3,
        "run.out": str(out),
        "run.max_steps": 2,
        "run.mixed_precision": "no",
    }


def _dotlist(values: dict) -> list[str]:
    return [f"{k}={v}" for k, v in values.items()]


def test_train_encoder(data_dir, encoder_dir, encoder_format, tmp_path) -> None:
    out = tmp_path / "enc"
    raw = config.load(
        config.EncoderTrainConfig,
        None,
        _dotlist(
            {
                **_common(data_dir, out),
                "model.id": str(encoder_dir),
                "model.attention": "sdpa",
                "model.gradient_checkpointing": False,
                "format.yes_token": encoder_format.yes_token,
                "format.no_token": encoder_format.no_token,
                "optim.loss": "sigmoid_norm",
            }
        ),
    )
    cfg = config.resolve(raw)
    trainer.launch(
        cfg,
        raw,
        lambda device: encoder.MaskedEncoder.build(
            cfg.model, cfg.format, device, torch.float32, cfg.optim.loss
        ),
    )
    metrics = json.loads((out / "eval_final.json").read_text())
    assert set(metrics) == {"toy0", "toy1", "ALL"}
    assert (out / "config.yaml").exists()

    # The run directory loads back for evaluation.
    model = encoder.MaskedEncoder.from_run(out, "cpu", attention="sdpa")
    rows = storage.load_examples(str(data_dir), "eval")
    results = evaluate.score_rows(model, rows)
    assert len(results) == len(rows)

    # Paper analysis: option coupling on the same run.
    coupling_out = coupling.analyse(out, rows, shuffles=1, seed=0)
    assert coupling_out["rows"] == len(rows)
    assert coupling_out["loss"] == "sigmoid_norm"

    # Evidence head on the frozen run (gold spans only).
    evidence_cfg = config.resolve(
        config.load(
            config.EvidenceConfig,
            None,
            [
                f"encoder.run={out}",
                f"data.source={data_dir}",
                "data.labels=gold",
                "data.val_frac=0.25",
                "train.epochs=1",
                "train.rows_per_step=2",
                f"out={out}/evidence",
            ],
        )
    )
    (out / "evidence").mkdir()
    model.unwrapped.requires_grad_(False)
    head = evidence_train.train(evidence_cfg, model, out / "evidence")
    assert (out / "evidence" / "head.pt").exists()
    assert not head.training
    shutil.rmtree(out)


@pytest.mark.parametrize("finetune", ["lora", "full"])
def test_train_decoder(data_dir, decoder_dir, tmp_path, finetune) -> None:
    out = tmp_path / f"dec_{finetune}"
    raw = config.load(
        config.DecoderTrainConfig,
        None,
        _dotlist(
            {
                **_common(data_dir, out),
                "model.id": str(decoder_dir),
                "model.finetune": finetune,
                "model.lora_rank": 4,
                "model.gradient_checkpointing": False,
                "optim.warmup_steps": 1,
            }
        ),
    )
    cfg = config.resolve(raw)
    trainer.launch(
        cfg, raw, lambda device: decoder.TrieDecoder.for_training(cfg, device)
    )
    saved = out / ("adapter" if finetune == "lora" else "model")
    assert saved.exists()
    metrics = json.loads((out / "eval_final.json").read_text())
    assert 0.0 <= metrics["ALL"]["accuracy"] <= 1.0


def test_schedule_warms_up_then_decays() -> None:
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([param], lr=1.0)
    sched = trainer.linear_schedule(opt, total=10, warmup=2, min_lr_ratio=0.1)
    lrs = []
    for _ in range(10):
        lrs.append(sched.get_last_lr()[0])
        opt.step()
        sched.step()
    assert lrs[0] == pytest.approx(0.5) and max(lrs) == pytest.approx(1.0)
    assert lrs[-1] < lrs[2]
