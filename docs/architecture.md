# Architecture: the LLM-as-Jev answer interface and loss

Our open-source version of the fine-tuning recipe of Li & Wagle, "LLM-as-Jev"
(arXiv 2610.02076v3, §3.2–3.4 and App. C, I), trained and evaluated on the
Reason+Long-12k data of `docs/data.md`. Models: Qwen3-0.6B and Qwen3.5-4B,
as in the paper.

## 0. Code map (`src/`)

| file | role |
|---|---|
| `dataloader/data.py` | `load_rows(source, split)`: rows from a local folder / `.jsonl`, or a Hugging Face dataset id |
| `modeling/modeling.py` | `load_base`, `add_lora`, `load_trained`, `batched_hidden` |
| `dataloader/answers.py` | prompt (`prompts/decision.j2`), trie, `encode`, `score` (Eq. 1 + in-trie decoding) |
| `train/losses.py` | ℒ_tree and the KL anchors, batched over trie nodes (TensorDict, shapes per line); `compute` for one example |
| `train/train.py` | training loop (CLI) |
| `evaluation/evaluate.py` | metrics on the eval split (CLI) |
| `evaluation/benchmarks.py` | the paper's external benchmarks and JevBench as rows (CLI) |
| `modeling/span_heads.py` | evidence-span heads kept from the earlier joint trainer (not yet wired in) |
| `dataloader/build_data.py` | builds Reason+Long-12k (`docs/data.md`) |
| `dataloader/prompts.py`, `prompts/decision.j2` | Jinja templates (`prompts/` at `src/prompts/`) |
| `blogs/model_viewer.py` | trained-vs-base viewer (`scripts/view.sh`); dataset examples in `blogs/01_prompt_and_data/` |

Launchers: `scripts/train.sh`, `scripts/evaluate.sh`, `scripts/view.sh`;
data: `make data`. Models are loaded from the Hugging Face Hub (latest
version). All logging goes through loguru.

## 1. Answer interface and candidate trie (§3.2)

The options are listed as `[1] …`, `[2] …`, and the assistant turn is
prefilled with `Best answer: [` (after the empty think block). Candidate k is
the suffix `k]`.

Suffix tokens after `Best answer: [`, with the real chat template:

| option k | tokens after `Best answer: [` | ids (identical in Qwen3-0.6B and Qwen3.5-4B) |
|---|---|---|
| 1 | `1` `]` | 16, 60 |
| 10 | `1` `0` `]` | 16, 15, 60 |
| 12 | `1` `2` `]` | 16, 17, 60 |
| 151 | `1` `5` `1` `]` | 16, 20, 16, 60 |

Each digit is its own token and `]` (id 60) always closes the suffix. For both
tokenizers the prompt's tokens are an exact prefix of prompt + suffix, so all
K candidates share one prompt pass (the check the paper requires, App. C).

**Why the closing `]` matters.** Without it, option `1` would be a prefix of
options `10`–`19`, and "the model wrote 1" would be ambiguous. With it, the
suffixes are **prefix-free**: no complete suffix starts another one, so every
candidate is a distinct leaf of a tree.

The trie for K = 12:

```
root ─┬─ "1" ─┬─ "]"  → option 1
      │       ├─ "0" ── "]" → option 10
      │       ├─ "1" ── "]" → option 11
      │       └─ "2" ── "]" → option 12
      ├─ "2" ── "]" → option 2
      ...
      └─ "9" ── "]" → option 9
```

Each node v has a set of **legal tokens** ℒ(v), the tokens that stay inside
the trie:

- At the root, ℒ = {1, …, 9}. `0` is not legal, and neither is `A` or `\n`.
- At "1", ℒ = {`]`, 0, 1, 2}. This node is the real decision between option 1
  and options 10–12.
- At "2", and at "10", "11" and "12", ℒ = {`]`}: there is only one legal
  token.

**Depth.** A suffix has ⌊log₁₀ K⌋ + 2 tokens:

| datasets | options | tokens per suffix |
|---|---|---|
| PIQA, αNLI | 2 | 2 |
| ProofWriter, ContractNLI | 3 | 2 |
| ReClor, LogiQA, CosmosQA, QuALITY | 4 | 2 |
| Bitext | 27 | 3 |
| MASSIVE | 60 | 3 |
| CLINC150 | 151 | 4 |

## 2. Prompt (§3.2, App. A)

`answers.build_prompt` renders `src/prompts/decision.j2` as the user turn,
applies the model's chat template (`add_generation_prompt=True`,
`enable_thinking=False`, no system prompt) and appends `Best answer: [`:

```
State: <part 1 text>

<part 2 text>

<question or instruction>
Options:
[1] <option 1>
[2] <option 2>
Select one option. Answer only with its bracketed numeric identifier.
```

The paper fixes only the frame (`State:`, `Options:`, the `[k]` lines, the
last line). Our choices for what goes in the state:

- Part texts without part names, separated by a blank line (the paper never
  labels parts).
- The row's question or instruction on every row, just before `Options:`,
  so the task sits next to the options. The paper's intent data also has
  fixed instructions (App. F: "four fixed instruction templates"); its only
  described layout (MMBench, App. B) puts the instruction first.
- Yes/no rows use the same `[1]` / `[2]` options, with descriptive option
  texts (§3.1: options "describing the acceptance criteria for 'yes' and
  'no'").

## 3. Decision loss ℒ_tree (Eq. 2)

`answers.encode` feeds the prompt followed by the whole gold suffix (teacher
forcing) and records the gold path: for each suffix token, the position
whose logits predict it, the legal tokens of the node it leaves, and the
gold token. `losses.tree_loss` sums one term per path step:

```python
for t, (_, legal, gold) in enumerate(path):
    z = logits[t].float()
    loss = loss + torch.logsumexp(z[legal], 0) - z[gold]
```

Each step is −log(p(gold) / m), with m the node's legal mass. Writing
p = softmax(z) over the vocabulary, the full-vocabulary normalizer cancels:

$$-\log\frac{p_g}{m} = \text{lse}_{\text{legal}}(z) - z_g$$

so each step is the cross-entropy over that node's legal tokens. Steps where
only `]` is legal add exactly 0. The cast to `.float()` keeps the
log-sum-exp in fp32 even when the model runs in bf16. Only the last n+1
positions' logits are needed (`logits_to_keep`), never the whole sequence's.

**Check: frozen Qwen3-0.6B, before training, on one Bitext row** ("i bought
somegoddamn item how do i cancel order {{Order Number}}", gold
`[1] cancel_order`, K = 27). Each candidate is scored as if it were the gold
one:

1. **The q_k sum to 1.** Σ q_k = 1.000001 over the 27 candidates: the
   product of local softmaxes is an exact probability distribution over the
   K candidates (the paper's "exact listwise cross-entropy in a single
   forward pass"). Training computes only the gold path, but that term is
   the log of a properly normalized K-way probability.
2. **Top candidates:**

   | option | q (Eq. 2, training) | P (Eq. 1, inference) |
   |---|---|---|
   | [1] cancel_order | 0.8631 | 0.8631 |
   | [16] get_invoice | 0.0373 | 0.0373 |
   | [6] check_payment_methods | 0.0369 | 0.0369 |

   ℒ_tree at the gold answer is −log 0.8631 = 0.147.
3. **Why q and P are identical here.** The base model puts 0.9998 of its
   probability on the legal tokens at the root, and the unnormalized
   Σ exp(s_k) = 0.9986. When almost no probability leaks out of the trie,
   renormalizing at each node changes nothing.

**Why this matters for the anchors.** ℒ_tree trains q, which ignores leaked
mass: the model could move probability out of the trie (to `\n`, `A`, …)
at no cost. P at inference and text-mode generation would then drift from q.
ℒ_mass is the anchor that keeps the legal mass (0.9998 here) where the base
model has it.

## 4. KL anchors (Eq. 3, App. I)

ℒ_anchor = ℒ_mass + ℒ_out + ℒ_pos, each against the frozen base model p₀:

| anchor | where | what it compares |
|---|---|---|
| ℒ_mass | trie nodes | binary KL_B(m₀ ‖ m_θ): legal vs illegal mass |
| ℒ_out | trie nodes | KL(p̄₀ ‖ p̄_θ) over the illegal tokens, each side renormalized within its illegal mass |
| ℒ_pos | reply start and the position after `]` | KL over the full next-token distribution |

A node's distribution factors into legal vs illegal (ℒ_mass), the choice
among legal tokens (ℒ_tree, trained) and the choice among illegal tokens
(ℒ_out); only the middle factor is free to move. Weights: the gold-path
nodes share 0.5 and the 3 sampled off-path nodes share 0.5; the two context
positions get 0.5 each.

**Truncation.** Each reference distribution keeps its top 64 tokens plus one
remainder bucket; the model's distribution is bucketed over the same 64
tokens, and the KL is taken between the two 65-way distributions
(`losses._kl_top`). ℒ_mass is the binary KL computed from the legal and illegal
masses, each summed directly (`losses._split`): computing the illegal mass as
1 − m rounds to 0 in fp32 when m is within ~10⁻⁸ of 1, as at most trie
nodes, and made ℒ_out NaN.

**Batched layout.** `compute` stacks all N = T + O trie nodes (T gold-path,
O off-path) into one `[N, V]` logits tensor, with legal tokens as an
`[N, V]` boolean mask and the 0.5-shared weights as an `[N]` vector, so
every helper works on all nodes at once. Per-node results (`_split`,
`node_stats`, `node_anchor`, `pos_stats`) are `TensorDict`s with
`batch_size=[N]`, and `compute` returns a `TensorDict` of the four scalar
losses; each line of `losses.py` is annotated with its shape. Against the
earlier per-node loop this gives identical losses and gradients (checked
with perturbed LoRA weights, so the anchors are nonzero).

**Check (Qwen3-0.6B, the Bitext row of Section 3).** The model against
itself gives exactly 0 at every site. Top-64 covers 77.2% of the illegal
mass at the root (the illegal mass there is only 0.0002, a flat tail; the
paper's average is 94.5–98.9%) and 100.0% at both context positions.
Perturbed logits, top-64 + bucket vs exact KL:

| perturbation | ℒ_mass (root) | ℒ_out at root, top-64 / exact | ℒ_pos, reply start, top-64 / exact | ℒ_pos, after `]`, top-64 / exact |
|---|---|---|---|---|
| temperature 1.5 | 0.032 | 0.789 / 0.862 | 0.0193 / 0.0193 | 0.0053 / 0.0053 |
| Gaussian noise, sd 1 | 0.000 | 0.311 / 0.738 | 0.0026 / 0.0048 | 0.000 / 0.000 |

Temperature lets mass leak out of the trie (ℒ_mass catches it). Noise
reshuffles inside the tail, which the bucket cannot see, so the truncated
ℒ_out under-reports; the gap is at most tail mass × the KL within the tail
(the paper measures ≤ 7.3 × 10⁻⁴ nats per site in trained models). ℒ_out is
renormalized, so it is as large when the illegal mass is 0.0002 as when it
is 0.3: the model must keep the ranking of its non-answer tokens.

**Our implementation choices.**

- **Reference model: on the fly, not cached.** The paper caches p₀'s
  top-64 statistics in a pre-pass. We compute them during training from
  the same LoRA model with the adapter disabled (`model.disable_adapter()`),
  one extra no-grad forward per sequence: no cache files, and exact for
  the option shuffle each row gets. This needs LoRA (full fine-tuning has
  no frozen copy in memory).
- **Off-path nodes: one extra sequence each, as in the paper** ("each
  anchored node requires an additional forward evaluation over the prompt
  prefix"): prompt + the node's tokens, batched with the gold sequence.
  3 nodes are sampled uniformly from the trie nodes off the gold path (all
  of them when K ≤ 4).

## 5. Training and evaluation (`train.py`, `evaluate.py`)

```bash
scripts/train.sh Qwen/Qwen3-0.6B runs/qwen3_0.6b 2    # background; GPU 2
scripts/train.sh Qwen/Qwen3.5-4B runs/qwen3.5_4b 3
scripts/evaluate.sh Qwen/Qwen3.5-4B runs/qwen3.5_4b 1        # trained
scripts/evaluate.sh Qwen/Qwen3.5-4B runs/qwen3.5_4b 1 base   # base model
```

`--data` (train and eval scripts) takes a local folder or a Hugging Face
dataset id with `train` and `eval` splits.

Settings from the paper (Table 2): LoRA rank 16, α 32, dropout 0 on the
decoder linear layers; AdamW, weight decay 0.01, gradient clipping 1.0;
peak learning rate 1e-4, linear warmup over 100 steps, linear decay to 10%
of the peak; global batch 32; one epoch of Reason+Long-12k (375 steps);
λ = 1; 3 off-path nodes; top-64; seed 20260922; BF16 weights, FP32 LoRA
parameters and optimizer states; gradient checkpointing; max length 10,240
(no training row exceeds it with either tokenizer).

Ours: one GPU per run with gradient accumulation (the paper uses FSDP);
LoRA on the linear layers under `.layers.` (no vision tower,
`modeling.add_lora`); the update is skipped if the gradient norm is not
finite. (The first runs used Qwen3-0.6B `c1899de2` and Qwen3.5-4B
`851bf6e8`, the Hub versions of 2026-10-17.)

Evaluation, before training and after it, on 100 rows per dataset of our
in-domain `eval.jsonl` (the paper evaluates on external benchmarks only):

- accuracy of argmax s_k (Eq. 1), mean log P(gold), mean legal mass at the
  root;
- text mode on 20 rows per dataset: greedy generation of up to 32 tokens
  after `Best answer: [`; the share that starts with a valid `k]`
  (`gen_valid`), that is correct, and that ends right after the `]`
  (`gen_ends_after_bracket`). Base Qwen3-0.6B usually continues after the
  bracket (e.g. `44] balance`, `1]\n\n**Explanation:** …`), so the last
  number is compared with the base model's, not with 1.

Cost (Qwen3.5-4B, one GPU): 47.2M training tokens per epoch including the
off-path sequences; the longest row (8,851 tokens, 4 sequences) takes 16.5 s
for the forward and backward passes and peaks at 26 GiB.

**Numerical fixes found while testing** (both in `losses.py`): the illegal
mass is summed directly, never computed as 1 − m (fp32 rounds it to 0 and
ℒ_out became NaN); `_log1mexp` clamps inside the log, because clamping
log 0 = −inf afterwards gives a NaN gradient (0 · ∞) when the top-64 tokens
hold all the mass, which made every update's gradient NaN in BF16.

## 6. Inference: scoring and decoding (`answers.score_batch`)

One left-padded prefill of several prompts (positions skip the padding);
then, per depth of the trie, every node of that depth of every prompt runs
as one batch on top of a copy of its prompt's cache (1–3 short steps,
App. C), in chunks so that nodes × longest prompt stays within a token
budget. `answers._select_cache` builds each copy with the library's
`reorder_cache` (an `index_select` along the batch), which covers attention
keys/values and Qwen3.5's linear-attention conv and recurrent states.
`answers.score` is the one-prompt case (the viewer). It returns s_k (Eq. 1) for every candidate, the root's
legal mass, and the answer **decoded inside the trie**: the most likely
legal token at each node from the root down to the closing `]`, so the
output is always exactly one valid `k]` and nothing is generated after it.

Unconstrained `generate()` is not the answer path: the model emits `]\n\n`
as one merged token, so stopping on the `]` token never fires, and Qwen3-0.6B
(base and trained alike) then restates the option or writes an explanation.
Training never sees text after `]` (the input ends at the gold `k]`); only
ℒ_pos at the position after `]` keeps that distribution at the base model's.

Checked against recomputing the prompt once per node: same argmax on every
row tested, |ΔP| ≤ 0.03 (bf16); the decoded answer equals the argmax of
s_k. Time per row on one GPU after warm-up:

| model | CLINC150 (151 options) | MASSIVE (60) | QuALITY (4, 3.4K tokens) |
|---|---|---|---|
| Qwen3-0.6B, per-node recompute → cached | 2.74 → 0.38 s | 0.42 → 0.12 s | 0.38 → 0.09 s |
| Qwen3.5-4B, per-node recompute → cached | 9.3 → 0.37 s | 1.64 → 0.19 s | 0.97 → 0.24 s |

**Batching across rows** (`evaluate.evaluate`, `--batch-tokens`, default
32,768 padded prompt tokens per batch): rows are sorted by prompt length and
packed, then scored with `score_batch`; text mode stays one row at a time.
`benchmarks.py` also merges the LoRA adapter into the weights
(`merge_and_unload`), so trained runs cost the same as base runs. Batched
and one-at-a-time scores agree to 2e-4 (Qwen3-0.6B) and 0.02 (Qwen3.5-4B,
whose linear-attention kernels are not bit-exact across batch shapes) in
fp32. In bf16 they drift by up to ~0.3–1.4 nats on 77-option rows, the same
size as between a batch of one and a batch of two identical prompts (no
padding), i.e. bf16 rounding; on 120 benchmark rows the argmax agreed on
115/120 (0.6B) and 119/120 (4B) with equal accuracy.
