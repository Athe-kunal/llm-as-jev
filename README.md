# llm-jev

Decision models that pick one of several options for a question about a
document, in a single scoring pass. The code behind the technical report in
[`paper/`](paper/): fine-tuning UltraBERT-380M with one mask per option,
compared with LLM-as-Jev decoders (Li & Wagle, arXiv 2610.02076v3), plus an
evidence head that marks the sentences supporting the answer.

Two model families share one interface, `DecisionModel`
(`llm_jev/models/base.py`): `score(examples) -> s [K]` per row, `loss(row)`,
`save(...)`. Training, evaluation, the benchmarks and ContextCite are written
against that interface, so they work for both.

| family | class | how option k is scored | training loss |
|---|---|---|---|
| decoder | `TrieDecoder` (`models/decoder.py`) | log-probability of the token path "k]" after "Best answer: [" (a candidate trie) | L_tree + KL anchors to the base model (`losses/trie.py`) |
| encoder | `MaskedEncoder` (`models/encoder.py`) | MLM logit of " yes" (optionally minus " no") at the mask in front of option k | softmax, sigmoid-norm or sqrt-softplus over the options (`losses/option.py`) |

## Install

Python 3.11+, a CUDA GPU for training, and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra dev
```

FlashAttention-2 (packed encoder rows) is fetched from the Hugging Face Hub on
first use. Qwen3.5 decoders need `uv sync --extra qwen35`; span labelling
needs `--extra spans`.

## Configuration

Every command reads a typed config: an OmegaConf dataclass schema in
`llm_jev/config.py`, a YAML file under `configs/`, and optional `key=value`
overrides on the command line. Unknown keys and wrong types fail at startup,
and each run saves its resolved config next to its outputs.

```bash
llm-jev-train-encoder --config configs/train/ultrabert_380m_sigmoid_norm.yaml optim.lr=8e-6 run.out=runs/x
```

## Quickstart

```bash
make data                                                               # pinned sources -> data/splits/reason_long_nimble (CPU, ~6 min)
make train-encoder CONFIG=configs/train/ultrabert_380m_sigmoid_norm.yaml GPUS=0
make train-decoder CONFIG=configs/train/qwen3_0.6b_lora.yaml GPUS=0,1,2,3       # FSDP2
make evaluate      CONFIG=configs/evaluate/encoder.yaml ARGS="model.path=runs/ultrabert_380m_lr1.5e-5_sigmoidnorm part=test tag=test"
make benchmarks    CONFIG=configs/evaluate/benchmarks.yaml ARGS="model.kind=encoder model.path=runs/ultrabert_380m_lr1.5e-5_sigmoidnorm"
make evidence      CONFIG=configs/evidence/ultrabert_380m.yaml
```

| command | module | config |
|---|---|---|
| `llm-jev-build-data` | `llm_jev.data.build` | `configs/data/` |
| `llm-jev-train-decoder` | `llm_jev.training.decoder` | `configs/train/qwen3_*.yaml` |
| `llm-jev-train-encoder` | `llm_jev.training.encoder` | `configs/train/ultrabert_380m_*.yaml` |
| `llm-jev-evaluate` | `llm_jev.evaluation.evaluate` | `configs/evaluate/{decoder,encoder}.yaml` |
| `llm-jev-benchmarks` | `llm_jev.evaluation.benchmarks` | `configs/evaluate/benchmarks.yaml` |
| `llm-jev-train-evidence` | `llm_jev.evidence.train` | `configs/evidence/ultrabert_380m.yaml` |
| `llm-jev-spans` | `llm_jev.evidence.spans` | `configs/evidence/spans.yaml` (vLLM: `scripts/serve_labeller.sh`) |

The training commands are meant to run under `accelerate launch` (the Makefile
does this with `configs/accelerate/{ddp,fsdp2}.yaml`).

From Python:

```python
from llm_jev import config, models
from llm_jev.data import storage

model = models.load(config.ModelRef(kind="encoder", path="runs/ultrabert_380m_lr1.5e-5_sigmoidnorm"))
rows = storage.load_examples("Athekunal/llm-jev-reason-long", split="eval")[:8]
scores = model.score(rows)                  # one tensor s [K] per row
choice = [int(s.argmax()) for s in scores]  # index into row.options
```

## Paper analyses

Measurements that only feed the paper live in `llm_jev/analysis/` and are run
as modules with their configs under `configs/analysis/`; their output files
are what `paper/figures/make_figures.py` reads.

| module | what | writes |
|---|---|---|
| `analysis.latency` | ms and GPU memory per decision, decoders vs encoders | `runs/profile.json` |
| `analysis.coupling` | how much one option's probability depends on the others | `<run>/coupling_dev.json` |
| `analysis.at2` | AT2 for the encoder (attention from the option mask) | `<out>/theta.pt` |
| `analysis.citation` | AT2, average attention and ContextCite against gold evidence | `<at2>/attribution_eval_<part>.json` |
| `analysis.context_cite` | ContextCite by deletion, for any `DecisionModel` | (library) |

```bash
python -m llm_jev.analysis.latency --config configs/analysis/latency.yaml
make paper
```

## Layout

```
src/llm_jev/
  config.py        typed config schemas and loading (OmegaConf)
  data/            Example schema, dataset sources (pinned downloads), prompts, sampling, the build pipeline
  models/          DecisionModel interface; TrieDecoder; MaskedEncoder; FlashAttention-2 varlen packing
  losses/          trie loss + KL anchors (decoder); option recipes (encoder)
  training/        one Accelerate training loop (trainer.py) and the two entry points
  evaluation/      held-out split and external benchmarks, for any DecisionModel
  evidence/        sentences, metrics, the evidence head and its training, best-of-N span labels
  analysis/        paper-only measurements (latency, coupling, AT2, ContextCite, citation comparison)
  prompts/         Jinja templates
configs/           YAML for every command
tests/             CPU tests on tiny random models (pytest)
paper/             LaTeX sources and figure scripts
```

## Development

```bash
make test     # pytest, CPU only, no downloads
make lint     # ruff (Google style, Google docstrings)
make format
```

## Released

- 🤗 [Athekunal/llm-jev-reason-long](https://huggingface.co/datasets/Athekunal/llm-jev-reason-long):
  training and held-out decisions, plus generated evidence spans. ReClor is
  withheld from this release.
- [Athekunal/llm-jev-qwen3-0.6b](https://huggingface.co/Athekunal/llm-jev-qwen3-0.6b),
  [Athekunal/llm-jev-qwen3-4b](https://huggingface.co/Athekunal/llm-jev-qwen3-4b):
  LLM-as-Jev decoders (merged weights; the LoRA adapter in `adapter/`).
