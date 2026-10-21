# Uniform `Example` contract for the dataloader

Date: 2026-10-21. Status: design approved in conversation; written spec awaiting review.

## Goal

Every dataset (the 12 Reason+Long-12k sources, the external eval benchmarks, and Nimble) is read by a
*source* and converted to one typed dataclass, `Example`. Everything downstream (prompt building,
training, evaluation, span generation, context-cite) consumes only `Example`. Adding a dataset
becomes one small registered source. The code base is meant to be usable by other people, so the
contract is strict and there is **no backward compatibility with the old dict/JSONL format**.

Constraints from the user:
- Reuse and refactor existing code; do not rewrite what works (parsers in `build_data.py`, the
  trie/scoring code in `answers.py`, the `OverlapIndex`, `take_eval`, `sample_train` logic all move,
  they are not re-invented).
- Nimble uses the **same answer sequence** (`Best answer: [k]`), so one trie per option count
  serves every dataset. No new answer scheme.
- Keep dataclasses (stdlib), no pydantic.

Out of scope: renaming `answers.py`, any change to the model, loss or training recipe, new
answer-code schemes.

## Schema (`dataloader/schema.py`)

```python
@dataclass(frozen=True)
class Part:    name: str; text: str
@dataclass(frozen=True)
class Span:    part: int; start: int; end: int          # part = index into Example.parts

@dataclass(frozen=True)
class Example:
    id: str
    source: str
    parts: tuple[Part, ...]                  # may be empty (MMLU, ARC)
    question: str
    options: tuple[str, ...]                 # >= 2; position + 1 is the number shown in the prompt
    gold: int                                # 0-based index into options
    spans: tuple[Span, ...] = ()
    group: str = ""                          # leakage unit (document / story / theory / family)
    subset: str = ""                         # stratification key (label name, hypothesis, ...)
    meta: Mapping[str, Any] = field(default_factory=dict)       # provenance, annotation output
    overlap_text: tuple[str, ...] | None = field(default=None, repr=False, compare=False)
```

- `__post_init__` validates and raises `ValueError` naming the example id: at least two options,
  `0 <= gold < len(options)`, every span has `0 <= start < end <= len(parts[span.part].text)`.
- Properties: `has_spans`, `answer` (`options[gold]`), `span_text(span)`.
- `to_dict()` / `from_dict()` are the single serialization (nested plain JSON; no `labels` mapping,
  so no Hub repair step). `overlap_text` is not serialized; it replaces the private `_units` key
  and the `_clean` helper. Default decontamination text is parts + question + options.
- Dropped from the old row: `labels`/`label` (now `options`/`gold`), `needs_spans` (now
  `has_spans`), `Span.text` (derived).
- Soft span targets (vote fractions) stay in `meta` until training needs a typed field.
- The semantic label name (`entailed`, `yes`, ...) is kept as `subset` where it was used for
  balancing and the manifest.

## Storage (`dataloader/storage.py`, replaces `data.py`)

The pipeline's container is a `datasets.Dataset`, one row per example, in the fixed schema
`storage.FEATURES` (the dict form of `Example.to_row`: `meta` is a JSON string so every column has
one Arrow type). It is saved as Parquet (`train.parquet`, `eval.parquet`), which is also what a Hub
dataset holds, so a built dataset can be pushed as it is. `load_dataset(source, split)` reads a
`.parquet` file, a folder holding `<split>.parquet`, or a Hugging Face dataset id;
`load_examples` is the same as validated `Example`s. There is no JSONL format and no reader for
the old `labels`/`label` rows.

`overlap_text` is not in the saved schema. During the build it is an extra working column
(`to_dataset(..., with_overlap_text=True)`), because every step must get what it needs from the row
alone.

## Sources (`dataloader/sources/`)

```python
class DatasetSource(Protocol):
    name: str
    def load(self) -> dict[str, list[Example]]: ...      # {"train": [...], "validation": [...]}
```

- Registry: `@register` + `get_source(name)` / `SOURCES`, replacing the `DATASETS` dict.
- The twelve parsers in `build_data.py` keep their bodies and move into `intent.py` (CLINC150,
  MASSIVE, Bitext), `multiple_choice.py` (ReClor, LogiQA 2.0, CosmosQA, PIQA, alpha-NLI, QuALITY)
  and `reasoning.py` (ProofWriter, ContractNLI, MuSiQue). `_row`/`_mc_row`/`_intent_rows` build an
  `Example` instead of a dict; the loaders' logic is unchanged.
- Per-source metadata (split names: today `SOURCE_SPLITS`) sits on the source. Per-source sampling
  quirks (CosmosQA 16% "none", ProofWriter label balance) are declarative data on the source: a pure
  `stratum(example) -> str` function and `shares` (a share per stratum, the last takes the
  remainder, or `"equal"`). There are no `if name == ...` branches and no sampler functions.
- `benchmarks.py` (external eval sets) also emits `Example` through the same protocol.
- Downloading (`download()`, pinned revisions, sha256 into `SOURCES.json`) moves to
  `sources/download.py` unchanged.

### Nimble (`sources/nimble.py`)

Source: `github.com/bespokelabsai/nimble`, `data/train.jsonl` and `data/eval.jsonl`, pinned at
commit `4edb0703a2e75d0170c3a45645a1961f4d1a2867` (the upstream `eval.jsonl` is byte-identical to
the local `data/external/nimble/eval.jsonl`). Upstream names its test split `eval`; the source
exposes it as `validation`, the same as every other source.

Resolution order for each split: (1) `<data home>/nimble/<split>.jsonl` if present (data home = `$LLM_JEV_DATA`, default
`$HF_HOME/llm_jev`); (2) otherwise download the pinned raw GitHub file there (checked against its
pinned sha256 and recorded in `SOURCES.json`, same as the other sources); (3) if the download fails, **raise a `FileNotFoundError`** that names the local
path and the URL. No silent empty split.

Mapping (restored from the earlier `spanjudge/synth.py:nimble()` in git history, commit 24c193a,
and the same rules as `benchmarks.jevbench`):

| raw field | `Example` |
|---|---|
| `id` | `id = "nimble:<id>"`, `source = "nimble"` |
| `input.state` | `parts` via `_nimble_parts`: `str` -> `Document`; list of turns -> one `Statements` part, lines `"Speaker: text"`; dict -> one part per key, name = `key.replace("_", " ").title()`, `questions` skipped, nested `state` flattened, empty parts dropped |
| `input.questions.decision.instructions` | `question` |
| `decision.type == "choice"` | `options` = criteria values in order, `gold` = index of `reference.target` |
| `decision.type == "noul"` | `options` = criteria values in key order (`false`, `true`), `gold` = index of `str(target).lower()` |
| `decision.type == "score"` | `options` = criteria list in order (levels 0..), `gold = int(target)` |
| `family` | `group`; the base and counterfactual variants share it so a pair never straddles train and eval |
| `decision.type` | `subset` |
| `domain`, `variant`, `source_family`, `method` | `meta` |
| `evidence_certificate` | not carried (large; not needed to train); spans come from annotation |

Because options are positional, Nimble's `true`/`false`/`0..5`/choice keys all render as
`Best answer: [1] .. [k]` through the existing `answers.candidates(tokenizer, k)`. No trie change.

The annotated `data/synth/nimble-*.jsonl` is not a source; it is annotation output in the old
format and is not read by the new code (rebuild it from the Nimble source).

## Processing (`dataloader/processing.py`, `contamination.py`, `build.py`)

Everything maps onto `datasets.Dataset` operations, with no dataset-specific code in the pipeline:

1. **One parallel map.** `compute_features` is a stateless batched function for
   `Dataset.map(..., batched=True, num_proc=N)`. It adds `input_key` (normalized input hash),
   `n_tokens` (prompt length), `ngram_mask` / `sentence_mask` (overlap with the 17 benchmarks) and
   `stratum` to each example independently. Its read-only context (tokenizer, `OverlapIndex`) is set
   once in the parent and inherited by the forked workers, so the large index is neither pickled
   nor rebuilt.
2. **Column selections.** Duplicates (`dedupe_indices`), the overlap with the validation split,
   benchmark contamination (`contamination_stats` + mask filter) and length are `select`/`filter`
   on those small columns. `take_eval_indices` and `sample_indices` draw the seeded subsets from
   them (group-aware eval, stratum shares), with the same RNG call order as the original builder.
3. `build.py` composes these per dataset, concatenates, applies the seeded permutation, drops the
   feature columns and saves Parquet, plus `manifest.json`.

Feature columns are computed before the filters (so a few more rows are tokenized than before); the
selected rows and the manifest statistics are unchanged.

**Duplicate rule (`--dedupe-conflict`, recorded in the manifest).** The paper only requires the
benchmark-overlap filter and purging inputs that appear in the validation split; collapsing
duplicates inside the training pool is our choice. `gold` (default) is the original builder's rule:
copies must agree on the answer's position, so copies listing the same options in another order
count as conflicting and the whole question is dropped. Audit of the 12 sources: 1,430 distinct
questions erased, 1,421 of them agreeing on the answer text. `answer` compares the answer text and
erases only the 13 true conflicts. Switching changes 3,702 of the 12,000 training rows (the pools of
ReClor, LogiQA, CosmosQA, PIQA and alpha-NLI shift, so their seeded 1,000-row draws change) but only
18 recovered questions land in the new sample; the eval split changes by 3 CosmosQA rows. The
datasets with human spans (ContractNLI, MuSiQue, ProofWriter) are identical under both rules.

## Consumers

`answers.user_message` / `build_prompt` / `encode`, `train`, `evaluation/evaluate.py`,
`evaluation/benchmarks.py`, `span_generation/best_of_n.py` and `modeling/context_cite.py` switch from
dict keys to attributes (`ex.options[ex.gold]`, `ex.parts[s.part].text`). This also resolves a latent
inconsistency: on disk `spans[].part` is a part *name*, but `best_of_n.py:45` and `context_cite.py:64`
index `parts[s["part"]]` as an int. The trie, scoring and batching functions in `answers.py` are
untouched; only the functions that take a row change.

## Errors

Schema violations raise with the example id. Sources fail loudly on malformed records. Nothing is
dropped silently: every drop is counted in the manifest. A missing Nimble file that cannot be
downloaded raises (see above).

## Testing

- Schema: round trip (`from_dict(to_dict(x)) == x`) and each validation error.
- One tiny checked-in raw fixture per source, asserting the produced `Example`s.
- Nimble: fixtures for each state shape (str, list of turns, list of str, dict) and each question
  type (choice, noul, score); download-failure raises; **golden test** that the new loader's parts,
  question, options and gold match every row of the existing `data/synth/nimble-train.jsonl`.
- Processing: one test per step.
- Prompt golden test: one `Example` renders the same prompt text as the current dict version.

## Rollout and operational constraints

1. `schema.py`, `storage.py`, tests (additive; nothing existing changes).
2. Sources, including Nimble.
3. `processing.py` and `build.py`.
4. Consumer migration.
5. Regenerate local data once with the new builder.

Step 4 (done): `answers.py` (`user_message`, `build_prompt`, `encode`), `context_cite.py`,
`masked_cite.py`, `best_of_n.py`, `serving/sglang_client.py`, `train.py`, `evaluate.py` and
`benchmarks.py` take `Example`s directly; `data.py` and the interim adapter are deleted, and nothing
reads the old dict rows. `answers.build_prompt` returns the part ranges as a list indexed like
`Span.part`; `context_cite.ablate` returns an `Example`; `to_spans`/`select` return `schema.Span`s;
`best_of_n` writes `spans` as `{"part": index, "start", "end"}`. Checked against the previous commit's
code on 600 real examples (encodings, prompts, sources, ablated prompts, token owners, span
selection and metrics: no differences), on real ContextCite/`masked_cite` attributions (identical
weights), on `best_of_n` against a live vLLM server (same prompts and answers; the server's own
nondeterminism is larger than any difference), and end to end (train, evaluate, SGLang serving).

Step 3 (done) replaced `build_data.py` (deleted) with `build.py` and the `map`-based `processing.py`,
and `make data` now builds into `data/splits/reason_long_12k_v2` (the old-format
`data/splits/reason_long_12k` stays untouched for the running job).

A span-generation job (`llm_jev.span_generation.best_of_n`, pid 2595774/2618098) is running now and
reads `data/splits/reason_long_12k`. Therefore: steps 1-3 are additive and safe; **step 4 edits
`best_of_n.py` and `context_cite.py` and does not start until that job finishes**; step 5 writes to a
new directory (not over `data/splits/reason_long_12k`) and no template (`prompts/*.j2`,
`span_generation/cite.j2`) is moved. The old files stop loading after step 4, by design.

Acceptance: the migrated pipeline reproduces the existing prompt text, and the LLM-as-Jev
reproduction numbers do not change.
