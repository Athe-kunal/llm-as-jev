"""Typed configuration for every entry point, read from YAML with OmegaConf.

Each command has one top-level dataclass schema (e.g. `EncoderTrainConfig`).
A run is a YAML file under configs/ plus optional dotlist overrides on the
command line (`optim.lr=1e-5 run.out=runs/x`). The YAML is merged into the
schema, so an unknown key or a wrongly typed value fails at startup, and the
resolved config is saved next to the run's outputs.
"""

import argparse
from collections.abc import Sequence
import dataclasses
import pathlib
from typing import TypeVar

from omegaconf import MISSING
from omegaconf import DictConfig
from omegaconf import OmegaConf

SEED = 20260922  # The paper's training seed (LLM-as-Jev, Table 2).
DEFAULT_DATA = "data/splits/reason_long_nimble"  # Local build or a Hub id.
RELEASED_DATA = "Athekunal/llm-jev-reason-long"

DECODER = "decoder"
ENCODER = "encoder"
VARLEN = "varlen"  # Encoder attention: FlashAttention-2 varlen, packed rows.

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Building blocks shared by several commands.
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class DataConfig:
    """Training and evaluation rows.

    Attributes:
      source: A local folder of <split>.parquet files or a Hub dataset id.
      train_split: Split the model is trained on.
      eval_split: Held-out split evaluated during training.
      max_tokens: Encoded training rows longer than this are dropped.
      eval_per: Evaluation rows per source dataset (the dev sample).
      eval_seed: Picks the evaluation rows; separate from `run.seed` so seed
        replicates are scored on the same rows.
    """

    source: str = DEFAULT_DATA
    train_split: str = "train"
    eval_split: str = "eval"
    max_tokens: int = 16384
    eval_per: int = 100
    eval_seed: int = SEED


@dataclasses.dataclass
class OptimConfig:
    """AdamW with linear warmup, then linear decay to `min_lr_ratio` x peak.

    Attributes:
      lr: Peak learning rate.
      betas: AdamW betas.
      eps: AdamW epsilon.
      weight_decay: AdamW weight decay.
      batch_size: Rows per optimizer step, over all processes.
      warmup_steps: Warmup length in steps; if None, `warmup_ratio` is used.
      warmup_ratio: Warmup length as a share of the steps.
      min_lr_ratio: Final learning rate / peak.
      max_grad_norm: Gradient clipping norm.
    """

    lr: float = 1e-4
    betas: list[float] = dataclasses.field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-8
    weight_decay: float = 0.01
    batch_size: int = 32
    warmup_steps: int | None = None
    warmup_ratio: float = 0.05
    min_lr_ratio: float = 0.0
    max_grad_norm: float = 1.0


@dataclasses.dataclass
class RunConfig:
    """Output directory, run length and evaluation schedule.

    Attributes:
      out: Run directory (config, logs, checkpoints, metrics).
      seed: Seeds the option shuffles and the visiting order.
      epochs: Passes over the training rows.
      max_steps: If > 0, stop after this many steps (smoke tests).
      eval_every: Also evaluate (and checkpoint) every N steps; 0 = never.
      skip_base_eval: Do not evaluate before training.
      mixed_precision: Accelerate's mixed precision ("no", "bf16", "fp16").
    """

    out: str = MISSING
    seed: int = SEED
    epochs: int = 1
    max_steps: int = -1
    eval_every: int = 0
    skip_base_eval: bool = False
    mixed_precision: str = "no"

    def __post_init__(self) -> None:
        """Reads YAML's bare `no` (the boolean False) as "no"."""
        if self.mixed_precision.lower() in ("false", "none"):
            self.mixed_precision = "no"


# ---------------------------------------------------------------------------
# Models.
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class DecoderModelConfig:
    """A causal LM scored through the candidate trie.

    Attributes:
      id: Hugging Face id or local path of the base model.
      adapter: A saved LoRA adapter to load on top of `id` (inference only).
      finetune: "lora" (p0 = the adapter disabled) or "full" (fp32 weights,
        p0 = a frozen bf16 copy of the base model).
      lora_rank: LoRA rank; alpha is 2 x rank.
      gradient_checkpointing: Recompute activations in the backward pass.
      batch_tokens: Padded prompt tokens per scoring forward.
      batch_rows: Rows (and trie nodes) per scoring forward.
    """

    id: str = MISSING
    adapter: str | None = None
    finetune: str = "lora"
    lora_rank: int = 16
    gradient_checkpointing: bool = True
    batch_tokens: int = 32768
    batch_rows: int = 64


@dataclasses.dataclass
class EncoderModelConfig:
    """A masked-language-model encoder with one mask per option.

    Attributes:
      id: Hugging Face id or local path.
      revision: Hub commit (None: latest).
      attention: "varlen" (FlashAttention-2 varlen: no masks, packed
        evaluation) or a transformers attn_implementation ("sdpa", ...).
      pack_tokens: Tokens per packed forward pass in evaluation (varlen).
      gradient_checkpointing: Recompute activations in the backward pass.
    """

    id: str = MISSING
    revision: str | None = None
    attention: str = VARLEN
    pack_tokens: int = 32768
    gradient_checkpointing: bool = True


@dataclasses.dataclass
class FormatConfig:
    """How an encoder reads a decision (the prompt and its verbalizer).

    Not `yes` / `no`: YAML reads those bare words as booleans, even as keys.

    Attributes:
      template: Prompt template name (llm_jev/prompts/<template>.j2). It must
        render the `mask` variable exactly once per option, in option order.
      yes_token: Verbalizer: s_k is the MLM logit of this token at mask k.
      no_token: If set, s_k = logit(yes_token) - logit(no_token) at mask k.
      no_word: The negative word the instruction names, when it differs from
        `no_token`. Default: `no_token`, else " no".
    """

    template: str = "decision_mlm"
    yes_token: str = " yes"
    no_token: str | None = None
    no_word: str | None = None


@dataclasses.dataclass
class ModelRef:
    """A trained (or base) decision model to load for inference.

    Attributes:
      kind: "decoder" or "encoder".
      path: Decoder: Hub id or local path. Encoder: a training run directory
        (its config.yaml gives the prompt format).
      adapter: Decoder only: a saved LoRA adapter, merged for inference.
      attention: Encoder only: overrides the run's `model.attention`.
      dtype: Encoder weights: "float32" (scored under bf16 autocast, as in
        training) or "bfloat16".
    """

    kind: str = MISSING
    path: str = MISSING
    adapter: str | None = None
    attention: str | None = None
    dtype: str = "float32"


# ---------------------------------------------------------------------------
# Training.
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class TrieObjectiveConfig:
    """The decoder loss: L_tree + lam * (sum of the chosen KL anchors).

    Attributes:
      anchors: Subset of ("mass", "out", "pos") added to L_tree. All four
        terms are logged either way.
      lam: Anchor weight.
    """

    anchors: list[str] = dataclasses.field(
        default_factory=lambda: ["mass", "out", "pos"]
    )
    lam: float = 1.0


@dataclasses.dataclass
class DecoderTrainConfig:
    """Everything one decoder training run reads."""

    model: DecoderModelConfig = dataclasses.field(
        default_factory=DecoderModelConfig
    )
    objective: TrieObjectiveConfig = dataclasses.field(
        default_factory=TrieObjectiveConfig
    )
    data: DataConfig = dataclasses.field(default_factory=DataConfig)
    optim: OptimConfig = dataclasses.field(default_factory=OptimConfig)
    run: RunConfig = dataclasses.field(default_factory=RunConfig)


@dataclasses.dataclass
class EncoderOptimConfig(OptimConfig):
    """OptimConfig plus the encoder recipe's loss.

    Attributes:
      loss: One of llm_jev.losses.option.LOSSES: "listwise" (softmax over the
        options), "sigmoid_norm" (sigmoids normalised by their sum, as
        DeepSeek-V3's gate) or "sqrt_softplus". The last two need
        `format.no_token`.
    """

    loss: str = "listwise"


@dataclasses.dataclass
class EncoderTrainConfig:
    """Everything one encoder training run reads."""

    model: EncoderModelConfig = dataclasses.field(
        default_factory=EncoderModelConfig
    )
    format: FormatConfig = dataclasses.field(default_factory=FormatConfig)
    data: DataConfig = dataclasses.field(default_factory=DataConfig)
    optim: EncoderOptimConfig = dataclasses.field(
        default_factory=EncoderOptimConfig
    )
    run: RunConfig = dataclasses.field(default_factory=RunConfig)


# ---------------------------------------------------------------------------
# Data, evaluation and evidence.
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class BuildDataConfig:
    """Builds the training mix (Reason+Long-12k + Nimble).

    Attributes:
      out: Output folder for train.parquet, eval.parquet and manifest.json.
      download_only: Fetch the pinned sources into the cache and stop.
      eval_rows: Held-out rows per dataset.
      max_tokens: Longer prompts are dropped, never truncated.
      num_proc: Worker processes of the feature map.
      datasets: Subset of the mixture (None: all of it).
      per_dataset: Training rows drawn per dataset.
      length_tokenizer: Tokenizer that measures prompt lengths.
    """

    out: str = DEFAULT_DATA
    download_only: bool = False
    eval_rows: int = 500
    max_tokens: int = 10240
    num_proc: int = 8
    datasets: list[str] | None = None
    per_dataset: int = 1000
    length_tokenizer: str = "Qwen/Qwen3.5-4B"


@dataclasses.dataclass
class EvaluateConfig:
    """Evaluates a decision model on a held-out split.

    Attributes:
      model: The model to load.
      source: Data folder or Hub id.
      split: The split to read.
      part: "dev" (the per-dataset sample settings were chosen on), "test"
        (every other row) or "all".
      eval_per: Rows per dataset of the dev sample.
      seed: Picks the dev sample.
      rows_file: A JSON list of row ids to keep (e.g. rows that fit 8K).
      save_rows: Also write eval_<tag>_rows.json (per-row correctness).
      out: Output directory.
      tag: Output files are eval_<tag>*.json.
    """

    model: ModelRef = dataclasses.field(default_factory=ModelRef)
    source: str = DEFAULT_DATA
    split: str = "eval"
    part: str = "dev"
    eval_per: int = 100
    seed: int = SEED
    rows_file: str | None = None
    save_rows: bool = False
    out: str = MISSING
    tag: str = "final"


@dataclasses.dataclass
class BenchmarksConfig:
    """Evaluates a decision model on the external benchmarks.

    Attributes:
      model: The model to load.
      benchmarks: Names in llm_jev.evaluation.benchmarks.LOADERS (None: all).
      out: Output directory.
      tag: Output files are eval_benchmarks_<tag>.json.
    """

    model: ModelRef = dataclasses.field(default_factory=ModelRef)
    benchmarks: list[str] | None = None
    out: str = MISSING
    tag: str = "final"


@dataclasses.dataclass
class FrozenEncoderConfig:
    """A trained encoder run that a downstream method reads, frozen.

    Attributes:
      run: Encoder training run directory.
      attention: Attention backend; AT2's capture needs "varlen".
    """

    run: str = MISSING
    attention: str = VARLEN


@dataclasses.dataclass
class EvidenceDataConfig:
    """Evidence labels: gold spans, and generated spans where gold is missing.

    Attributes:
      source: Data folder or Hub id.
      split: Training split.
      labels: "gold" or "gold+generated" (gold wins where both exist).
      generated_repo: Hub dataset holding the generated spans.
      generated_file: Parquet file of generated spans in that dataset.
      datasets: Only these datasets (None: every dataset with labels).
      val_frac: Training rows held out to pick the best epoch.
      eval_split: Split the head is evaluated on.
      eval_per: Evaluation rows with gold spans per dataset (0: all).
      eval_seed: Picks the evaluation rows.
      eval_part: Report on this part ("dev" / "test") of the held-out split.
      tune_part: Tune the word threshold on this part (None: on held-out
        rows with gold spans that are not report rows).
    """

    source: str = DEFAULT_DATA
    split: str = "train"
    labels: str = "gold+generated"
    generated_repo: str = RELEASED_DATA
    generated_file: str = "spans/train.parquet"
    datasets: list[str] | None = None
    val_frac: float = 0.05
    eval_split: str = "eval"
    eval_per: int = 50
    eval_seed: int = 0
    eval_part: str | None = None
    tune_part: str | None = None


@dataclasses.dataclass
class EvidenceHeadConfig:
    """The MLP over [h_t ; h_answer ; h_t * h_answer]."""

    width: int = 512
    dropout: float = 0.1


@dataclasses.dataclass
class EvidenceTrainConfig:
    """AdamW over the head only (the encoder is frozen).

    Attributes:
      pos_weight: BCE positive weight (None: n_negative / n_positive over
        the training tokens).
    """

    epochs: int = 3
    rows_per_step: int = 16
    lr: float = 1e-3
    weight_decay: float = 0.01
    pos_weight: float | None = None
    seed: int = 0


@dataclasses.dataclass
class EvidenceConfig:
    """Everything one evidence-head run reads."""

    encoder: FrozenEncoderConfig = dataclasses.field(
        default_factory=FrozenEncoderConfig
    )
    data: EvidenceDataConfig = dataclasses.field(
        default_factory=EvidenceDataConfig
    )
    head: EvidenceHeadConfig = dataclasses.field(
        default_factory=EvidenceHeadConfig
    )
    train: EvidenceTrainConfig = dataclasses.field(
        default_factory=EvidenceTrainConfig
    )
    out: str = MISSING


@dataclasses.dataclass
class SpansConfig:
    """Best-of-N evidence labels from an OpenAI-compatible (vLLM) server.

    Attributes:
      model: Hub id of the labelling model, for its tokenizer.
      served_name: The name the server gives the model.
      base_url: One or more servers (rows are spread round-robin).
      source: Data folder or Hub id.
      split: Split to label.
      n: Citation sets sampled per row.
      temperature: Sampling temperature.
      top_p: Nucleus sampling.
      max_cited_tokens: Larger multi-sentence sets are discarded.
      max_tokens: Per completion, thinking included.
      max_model_len: The server's context length.
      concurrency: Rows in flight.
      timeout: Seconds per request.
      datasets: Only these datasets (None: all).
      with_spans: Only rows with human spans (to validate the labels).
      limit: Rows per dataset.
      out: Output JSONL; re-running with the same file resumes.
    """

    model: str = "Qwen/Qwen3.6-27B"
    served_name: str = "labeller"
    base_url: list[str] = dataclasses.field(
        default_factory=lambda: ["http://127.0.0.1:8011/v1"]
    )
    source: str = DEFAULT_DATA
    split: str = "train"
    n: int = 10
    temperature: float = 1.2
    top_p: float = 0.9
    max_cited_tokens: int = 384
    max_tokens: int = 16384
    max_model_len: int = 32768
    concurrency: int = 16
    timeout: float = 3600.0
    datasets: list[str] | None = None
    with_spans: bool = False
    limit: int = 20
    out: str = MISSING


# ---------------------------------------------------------------------------
# Paper analyses (llm_jev.analysis).
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class AT2DataConfig:
    """Rows and ablations for AT2's training data (stage 1).

    Attributes:
      source: Data folder or Hub id.
      split: Never the eval split: it is for evaluation.
      datasets: Only these datasets (None: all).
      rows: Rows, balanced over the datasets.
      min_sources: Ablating needs at least two sentences.
      masks: Ablations per row.
      seed: Picks rows and ablations.
    """

    source: str = DEFAULT_DATA
    split: str = "train"
    datasets: list[str] | None = None
    rows: int = 2000
    min_sources: int = 2
    masks: int = 32
    seed: int = 0


@dataclasses.dataclass
class AT2FitConfig:
    """Fitting theta (stage 2), as the reference's estimator trainer."""

    steps: int = 1000
    batch_size: int = 512
    lr: float = 1e-3
    max_grad_norm: float = 1.0
    val_frac: float = 0.1
    seed: int = 0


@dataclasses.dataclass
class AT2Config:
    """Everything one AT2 run reads."""

    encoder: FrozenEncoderConfig = dataclasses.field(
        default_factory=FrozenEncoderConfig
    )
    data: AT2DataConfig = dataclasses.field(default_factory=AT2DataConfig)
    train: AT2FitConfig = dataclasses.field(default_factory=AT2FitConfig)
    out: str = MISSING


@dataclasses.dataclass
class CitationConfig:
    """Compares AT2, average attention and ContextCite on gold evidence.

    Attributes:
      at2: AT2 run directory (its config names the encoder).
      source: Data folder or Hub id.
      split: Split with gold spans.
      per: Rows with gold evidence per dataset (0: all).
      part: Restrict to this part ("dev" / "test") of the held-out split.
      seed: Picks the rows.
    """

    at2: str = MISSING
    source: str = DEFAULT_DATA
    split: str = "eval"
    per: int = 50
    part: str | None = None
    seed: int = 0


@dataclasses.dataclass
class CouplingConfig:
    """How strongly encoder option probabilities depend on the other options.

    Attributes:
      runs: Encoder training run directories.
      source: Data folder or Hub id (None: each run's data.source).
      split: The split to read.
      part: Part of the split ("dev", "test", "all").
      eval_per: Rows per dataset of the dev sample.
      shuffles: Extra random option orders per row.
      seed: Seeds the sample and the shuffles.
      tag: Writes <run>/<tag>.json.
    """

    runs: list[str] = MISSING
    source: str | None = None
    split: str = "eval"
    part: str = "dev"
    eval_per: int = 100
    shuffles: int = 3
    seed: int = SEED
    tag: str = "coupling_dev"


@dataclasses.dataclass
class LatencyConfig:
    """Time and memory per decision: decoders against encoders.

    Attributes:
      decoders: Hub ids or paths (merged weights).
      encoders: Encoder training run directories.
      source: Data folder or Hub id.
      split: The split to read.
      eval_per: Rows per dataset.
      warmup: Untimed warm-up rows.
      out: Output JSON.
      merge: Keep the other models already in `out`.
    """

    decoders: list[str] = dataclasses.field(default_factory=list)
    encoders: list[str] = dataclasses.field(default_factory=list)
    source: str = DEFAULT_DATA
    split: str = "eval"
    eval_per: int = 100
    warmup: int = 20
    out: str = MISSING
    merge: bool = False


# ---------------------------------------------------------------------------
# Loading.
# ---------------------------------------------------------------------------


def load(
    schema: type[T],
    path: str | pathlib.Path | None = None,
    overrides: Sequence[str] = (),
) -> DictConfig:
    """Merges the schema's defaults, a YAML file and dotlist overrides.

    Args:
      schema: The dataclass the config must match.
      path: A YAML file, or None to start from the schema's defaults.
      overrides: Dotlist entries, e.g. ["optim.lr=1e-5"].

    Returns:
      The merged config, validated against `schema`.

    Raises:
      omegaconf.errors.OmegaConfBaseException: On an unknown key, a wrongly
        typed value or a missing required value.
    """
    layers = [OmegaConf.structured(schema)]
    if path is not None:
        layers.append(OmegaConf.load(path))
    layers.append(OmegaConf.from_dotlist(list(overrides)))
    cfg = OmegaConf.merge(*layers)
    OmegaConf.to_object(cfg)  # Fails here on a missing required value.
    return cfg


def resolve(cfg: DictConfig) -> object:
    """The typed dataclass instance of a loaded config."""
    return OmegaConf.to_object(cfg)


def save(cfg: DictConfig, path: str | pathlib.Path) -> None:
    """Writes a loaded config as YAML."""
    OmegaConf.save(cfg, path)


def from_command_line(
    schema: type[T], description: str = "", argv: Sequence[str] | None = None
) -> tuple[T, DictConfig]:
    """Parses `[--config FILE] [key=value ...]` into a typed config.

    Args:
      schema: The dataclass the config must match.
      description: Shown by --help.
      argv: Arguments (default: sys.argv[1:]).

    Returns:
      (the typed config, the merged DictConfig to save with the outputs).
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", help="YAML file (default: schema defaults)")
    parser.add_argument(
        "overrides", nargs="*", help="dotlist overrides, e.g. optim.lr=1e-5"
    )
    args = parser.parse_args(argv)
    raw = load(schema, args.config, args.overrides)
    return resolve(raw), raw  # type: ignore[return-value]
