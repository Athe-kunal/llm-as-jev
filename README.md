# llm-jev

Train LLMs as Jev decision models and attribute their decisions with fast ContextCite.

Two parts, one package (`llm_jev`):

- **LLM-as-Jev training.** A decision model answers a question about a
  document by choosing one option. The option is scored through a candidate
  trie, and training uses a tree-factorized listwise loss (L_tree) with KL
  anchors to the base model (Li & Wagle, "LLM-as-Jev", arXiv 2610.02076v3).
  The model is evaluated on a held-out split and on external benchmarks.
- **ContextCite for the decision.** Which context sentences the predicted
  answer depends on, by ContextCite (Cohen-Wang et al., NeurIPS 2024): ablate
  sentences, re-score the answer, fit a sparse linear surrogate. Three
  implementations of the same ablations:
  - deletion (`context_cite.attribute`): the reference method, one forward per ablation;
  - reader-only masking (`masked_cite.attribute(..., removal="reader")`): the
    document runs once and all ablated readers are packed into one sequence
    under a FlexAttention block mask;
  - full masking (`masked_cite.attribute(..., removal="full")`): dropped
    sentences are hidden from every later token; the kept tokens of all
    ablations are packed with FlashAttention-2 varlen, and shared prefixes
    are computed once (a prefix tree over the ablation copies).

  Both masked paths use `torch.compile`, and the ablation count follows the
  row's sentence count (`masked_cite.ablation_count`). Reader-only masking
  is the fastest but leaks (later sentences can carry a dropped sentence's
  content), so it is not equivalent to deletion; full masking is.

## Install

Requires Python 3.11+, a CUDA GPU for training and attribution, and
[uv](https://docs.astral.sh/uv/).

```bash
uv sync                                  # the package, editable, with torch (CUDA 12.8 wheels)
uv sync --extra dev                      # pytest
```

`causal-conv1d` is built from source against the installed torch (it needs
`CUDA_HOME`, see `pyproject.toml`). Hugging Face models are loaded by id
(latest version); the FlashAttention-2 kernel is fetched from the Hub on
first use (`llm_jev.modeling.fa2`).

## Quickstart

```bash
make data                                   # Reason+Long-12k -> data/splits/reason_long_12k (CPU, ~10 min)
scripts/train.sh Qwen/Qwen3-0.6B runs/qwen3_0.6b 2       # one GPU; logs to runs/qwen3_0.6b.log
scripts/evaluate.sh Qwen/Qwen3-0.6B runs/qwen3_0.6b 2    # eval split + external benchmarks
```

The console scripts run the same modules as `python -m llm_jev.<module>`:

| command | module | what |
|---|---|---|
| `llm-jev-train` | `llm_jev.train.train` | one epoch of L_tree + KL anchors, evaluated before and after |
| `llm-jev-evaluate` | `llm_jev.evaluation.evaluate` | eval split metrics |
| `llm-jev-benchmarks` | `llm_jev.evaluation.benchmarks` | external benchmarks (BoolQ, MMLU, ARC-C, ... and JevBench) |
| `llm-jev-merge` | `llm_jev.serving.merge` | merges a LoRA adapter into its base model for SGLang |
| `llm-jev-sglang-cite` | `llm_jev.serving.sglang_client` | decisions and ContextCite through an SGLang server |
| `llm-jev-build-data` | `llm_jev.dataloader.build_data` | builds Reason+Long-12k (`--download` fetches the sources) |

Attribution with the packed paths, from Python:

```python
from transformers import AutoTokenizer
from llm_jev.dataloader import data
from llm_jev.modeling import fa2, masked_cite, modeling

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = modeling.load("Qwen/Qwen3-0.6B", "runs/qwen3_0.6b/adapter")
fa2.use(model)                                   # flex attention and FA2 dispatch
row = data.load_rows(split="eval")[0]
res = masked_cite.attribute(model, tok, row, removal="full")   # or removal="reader"
top = sorted(res["sources"], key=lambda s: -s["w"])[:3]         # per-sentence weights w
```

The launchers in `scripts/` take the GPU index as an argument and set
`CUDA_VISIBLE_DEVICES` from it; the Python entry points use
`CUDA_VISIBLE_DEVICES` directly. Runs and data live in `runs/` and `data/`,
which are git-ignored.

## Serving with SGLang

SGLang runs in its own environment (a uv tool, `make sglang-install`), since
its torch and transformers pins differ from training's. Its score API cannot
select a LoRA adapter, so the adapter is merged into the weights first.

```bash
make sglang-install                                                  # once
make sglang-serve BASE=Qwen/Qwen3-4B ADAPTER=runs/qwen3_4b/adapter GPU=0 PORT=30000
llm-jev-sglang-cite --url http://127.0.0.1:30000 --tokenizer runs/qwen3_4b/merged --source contractnli --per 5
make sglang-stop PORT=30000
```

`make sglang-serve` merges `ADAPTER` into `runs/qwen3_4b/merged` when it is
missing, then starts `scripts/sglang_serve.sh` and waits for the server. The
client (`llm_jev.serving.sglang_client`) sends our own prompt as token ids to
`/v1/score`; it does not use `/v1/decisions`, which renders its own A-Z
prompt. ContextCite there deletes the dropped sentences (the reference
method): the radix cache reuses each ablated prompt's prefix, and the trie's
deeper nodes are sent after the prompts so they hit the cache.

```python
from transformers import AutoTokenizer
from llm_jev.serving import sglang_client

tok = AutoTokenizer.from_pretrained("runs/qwen3_4b/merged")
answer, s = sglang_client.predict("http://127.0.0.1:30000", tok, row)
res = sglang_client.attribute("http://127.0.0.1:30000", tok, row)   # same output as masked_cite.attribute
```

## Module map

```
src/llm_jev/
  dataloader/    data.py (rows from a folder, .jsonl or the Hub), answers.py (prompt, candidate trie,
                 scoring), prompts.py (Jinja loader), build_data.py (Reason+Long-12k)
  modeling/      modeling.py (base and LoRA models, batched forward), context_cite.py (ContextCite by
                 deletion), masked_cite.py (one-pass ContextCite: reader and full masking), fa2.py
                 (FlashAttention-2 varlen next to flex attention)
  train/         losses.py (L_tree, KL anchors), train.py
  serving/       merge.py (LoRA merged into the weights), sglang_client.py (decisions and ContextCite
                 over SGLang's /v1/score)
  evaluation/    evaluate.py (eval split metrics), benchmarks.py (external benchmarks)
  prompts/       decision.j2 (the decision prompt; package data)
scripts/         train.sh, evaluate.sh, sglang_serve.sh (launchers, one GPU each)
```

