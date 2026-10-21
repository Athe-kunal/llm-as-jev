# Learnings v2: LLM-as-Jev reproduction and evidence attribution

Findings from the LLM-as-Jev phase (from 2026-10-17): the decision model
answers through the trie (`docs/architecture.md` §1–6) and the evidence is
attributed afterwards (§7 ContextCite, §8 AT2). The earlier joint span-judge
work is in `learnings.md`.

## 1. AT2 is much weaker than ContextCite on our decision model

**Setup.** AT2 (Cohen-Wang et al., 2025): a sentence's score is θ·φ_j, a
learned weighted sum over all 448 (layer, head) pairs of Qwen3-0.6B of the
attention from the answer's positions to the sentence's tokens. θ is fitted
so that the summed scores of the kept sentences correlate (Pearson) with the
answer's logit q under ablations. 2,000 training rows (250 from each of 8
multi-sentence datasets), 32 ablations each, the 0.6B λ = 0.01 adapter,
the paper's optimiser (Adam 1e-3, cosine, 1,000 × 512). Sentences are
`context_cite.split_sources` with content merging (§7).

**Fit.** Held-out Pearson 0.21 (uniform θ, i.e. average attention) → 0.27
after training (train rows 0.38). Per dataset on held-out rows: αNLI 0.44,
ReClor 0.42, CosmosQA 0.37, LogiQA 0.32, ProofWriter 0.27, ContractNLI
0.18, MuSiQue 0.07, QuALITY 0.04.

**Against human evidence** (average precision of the sentence ranking, 20
eval rows per span dataset, `evaluation/attribution.py`):

| AP | random | average attention | AT2 | ContextCite (64 ablations) |
|---|---|---|---|---|
| ContractNLI | 0.04 | 0.12 | 0.32 | 0.51 |
| MuSiQue | 0.15 | 0.21 | 0.31 | 0.25 |
| ProofWriter | 0.22 | 0.41 | 0.48 | 0.68 |
| seconds per row | | | 0.11–0.18 | 0.4–2.5 |

On a second, harder sample (10 rows per dataset, seed 7) AT2's ContractNLI
AP was 0.11 (ContextCite 0.46). With 20 rows per dataset differences under
~0.05 are noise.

**Manual review** (30 rows, a Sonnet reviewer reading every citation):
ContextCite better in 20, AT2 in 4, both good 2, both poor 4; ContextCite
wins on every dataset. AT2 cites contract boilerplate (titles, party
names, checkboxes, signature lines) and same-predicate look-alikes in
ProofWriter ("Erin is red" for a question about Dave); ContextCite's top-1
is usually the sentence that states the answer, its 2nd–3rd picks often
filler. In 9 of 30 rows the reviewer disagreed with the automatic metrics
(e.g. a non-gold ContextCite sentence that is real evidence).

## 2. Why AT2 is poor here

1. **Position bias.** On ContractNLI 47% of AT2's top-3 sentences lie in the
   first 10% of the document (11% of sentences, 6% of gold evidence);
   MuSiQue 33%; ProofWriter 10% (no bias). ContextCite: 17%, 17%, 0%. Token
   view: AT2's largest token scores sit on sentence-final periods and the
   document's first tokens. Length is not the cause: AT2's ContractNLI picks
   are shorter than gold sentences (median 21 vs 62 tokens), so length
   normalisation would make it worse.
2. **Attention from the answer misses the evidence routing.** The prompt is
   document first, question and options after, causal attention: the
   question and option tokens read the document, the 2–3 answer tokens read
   the question. Untrained average attention from every position after the
   document instead of the answer tokens alone raises AP on ContractNLI
   0.12 → 0.24 and MuSiQue 0.21 → 0.27 (ProofWriter 0.41 → 0.42). θ has not
   yet been retrained on those features (θ trained on answer-token features
   applied to them gives 0.35 / 0.25 / 0.39, a mismatched combination).
3. **Small model, short target.** The paper attributes long generated
   responses of 7–8B models; ours is a 2-token decision of a 0.6B model, a
   thinner signal; 448 weights on 1,800 rows overfit (0.38 vs 0.27).
4. **Different ablation.** We delete sentences and re-render the prompt
   (positions shift); the paper masks attention, closer to the
   linear-in-attention surrogate AT2 assumes.

## 3. Things that did not explain it, or were fixed on the way

- Content-free sentences (blank form fields "Agent:____", checkboxes,
  nltk fragments like "(C.A.R."): 15.7% of ContractNLI sentences, never
  gold. Merging them into neighbours (§7) helped ContextCite (ContractNLI
  AP 0.44 → 0.51, top-1 gold 0.50 → 0.60) but not AT2 (0.37 → 0.32, within
  noise); the boilerplate AT2 cites is real text ("Title: Title:", party
  names), which no content filter removes.
- Token-level AT2 (the paper attributes tokens): a sentence's score is
  exactly the sum of its tokens' θ·φᵢ (checked to 1e-7), so token
  highlighting needs no retraining; it shows the same bias token by token.
  θ was fitted on sentence ablations, not token ablations as in the paper.
- A trained citation head on the frozen 0.6B (SpanHead on hidden states,
  same rows, targets and loss as AT2) reached held-out Pearson 0.23: no
  better than AT2's linear θ.

## 4. Open options

- Retrain θ on question+answer attention features (targets are stored, only
  the attention pass is recomputed: minutes); a variant, not the paper's.
- Keep ContextCite as the reference for faithful citations; speed it up
  with 32 ablations and a single-request scorer for ≤ 9 options.
- For evidence finding rather than attribution: a separate natively
  bidirectional encoder over [question ; options ; document] trained on the
  Qwen3.6-27B best-of-N span labels (`src/span_generation/`). Making the
  causal Qwen decision model bidirectional is ruled out (`learnings.md`
  §5.12, §5.15: Qwen3-1.7B collapsed, accuracy 0.756 → 0.531).

## 5. One-pass ContextCite (reader-only masking)

**Method** (`modeling/masked_cite.py`, 2026-10-20). The prompt is split
into the document (the row's parts) and the reader (question, options,
"Best answer: [" and the answer trie). The document runs once; each
ablation is a copy of the reader that may not attend to the dropped
sentences' tokens. All copies, plus a copy that keeps everything (the
decision itself), are packed into one sequence under one FlexAttention
block mask, the trie included (tree attention: each node sees the prompt
and its own prefix). Equivalent to prefilling the document into a KV cache
once and running every reader copy on top of it with the dropped positions
masked. Reader-only masking is our shortcut, not the ContextCite paper's
method (the paper deletes text); it is an attention knockout (Geva et al.
2023) at every layer for every reader token.

**Exact.** In float32, copy 0 equals the decision scorer
(`answers.score`) to ≤ 7e-5 and every ablated copy equals a dense-mask SDPA
reference to ≤ 1e-4 (12 datasets). In bf16 the scores differ by 0.1–0.37
from the SDPA scorer (kernel order; typical |s| of the answer 0.5), and by
up to 0.36 between compiled and eager flex: method comparisons must run in
float32.

**Fast.** 0.6B, bf16, 64 ablations + the decision, compiled with
`torch.compile(dynamic=True)`: ContractNLI 5.53 → 0.49 s per row (11×),
QuALITY 3.05 → 0.31 s (10×), MuSiQue 2.22 → 0.28 s (8×). The forward pass is
79% of the time. 4B in the live tester: 0.6 s (64), 1.1 s (128), 3.0 s
(256 ablations) on ContractNLI. The ablation count is now about two per
sentence, a power of two in [32, 256] (`ablation_count`). Pitfalls: torch's
default recompile limit (8) silently falls back to eager flex, which
materialises the attention matrix (minutes per step); compiling without
`dynamic=True` recompiles for every row length.

## 6. Reader-only masking is not deletion: it leaks

Float32 0.6B, 20 span rows per dataset, 64 ablations, the same masks:

| | AP deletion | AP reader-only | top-1 deletion | top-1 reader-only | target corr. (bf16 floor) |
|---|---|---|---|---|---|
| ContractNLI | 0.50 | 0.49 | 0.60 | 0.55 | 0.54 (0.99) |
| MuSiQue | 0.26 | 0.28 | 0.50 | 0.45 | 0.45 (0.97) |
| ProofWriter | 0.67 | 0.60 | 0.80 | 0.50 | 0.75 (0.99) |

The targets correlate only 0.45–0.75 with deletion's while bf16 rounding
alone keeps the deletion scorer at 0.97–0.99 against float32: the gap is
the method. Cause: document tokens are computed once with every sentence
visible, so a dropped sentence's content reaches the reader through later
sentences that absorbed it ("Fiona is cold" → "All cold people are kind").
This breaks ContextCite's independence assumption (dropping sentence j
removes j's effect): the effect of a sentence then depends on which
carriers are kept, credit drifts to the carriers (a ContractNLI title
outranked the gold clause), and ProofWriter, whose rules restate facts,
loses most (top-1 0.80 → 0.50).

## 7. The 0.6B barely reads the document

Hiding the whole document from every token (300 train rows) changes the
tree distribution by a median KL of 0.17 nats; 20% of rows move less than
0.05 nats, 5% more than 1 nat, and the answer is unchanged on 36–67% of
rows per dataset (average q of the answer 0.5–0.8). Closing the gold
sentences moved logit q by −0.5 to +0.1 on most rows, often no more than
closing as many random document tokens. Every attribution method is
bounded by this. Consequence for DiffMask (`modeling/diffmask.py`, stopped
for now): with a fixed KL margin of 0.05 the probe closes every gate (on
most rows that nearly satisfies the constraint), and once closed the Hard
Concrete samples clip to 0 and get no gradient; a larger dual step, a
larger initial multiplier and a bounded log alpha did not give selective
gates. Options left: a per-row relative margin, training only on rows that
depend on their document, or the 4B.

## 8. Full masking

**Method.** A dropped sentence is invisible to every later token, the
document included (`masked_cite.attribute(..., removal="full")`): each copy
carries the whole document and the block mask hides the copy's own dropped
tokens. It is deletion without re-tokenising and without shifting
positions. Equal to a dense-mask SDPA reference to ≤ 8e-5 (float32, 8 rows
× 4 masks). Since nothing reads a hidden token, a fully masked ablation is
just the kept tokens at their original positions under plain causal
attention.

**Comparison** (float32 0.6B, 20 rows per dataset, 64 ablations, the same
masks; stopped after ContractNLI, float32 runs dropped):

| ContractNLI | AP | top-1 | F1 | weight corr. with deletion | same top-1 as deletion |
|---|---|---|---|---|---|
| deletion | 0.50 | 0.60 | 0.30 | 1.00 | 1.00 |
| reader-only | 0.49 | 0.55 | 0.32 | 0.65 | 0.65 |
| full masking | 0.48 | 0.55 | 0.30 | 0.73 | 0.80 |

Full masking is closer to deletion than reader-only (weights 0.73 vs 0.65,
same top-1 0.80 vs 0.65) without a better AP on ContractNLI. On the 4B
(bf16, one row each, anecdotal) full masking moved a ContractNLI row from
AP 0.33 (title sentence first) to 1.00; MuSiQue, ProofWriter and Nimble
rows were unchanged.

**Fast full masking: kept tokens only, FA2, a prefix tree.** Nothing reads
a hidden token, so an ablation is just its kept tokens at their original
positions under plain causal attention (`masked_cite.full_batch`,
`modeling/fa2.py`). The first version (flex, whole document per copy)
computed the hidden sentences too, about half its work at keep ½.
FlashAttention-2 varlen comes from the Hugging Face Hub build
`kernels-community/flash-attn` (torch 2.9 / CUDA 12.8; flash-attn has no
wheel for torch 2.9). Copies sorted by their keep rows form a prefix tree
(AReaL-DTA, arXiv 2602.00482): each copy computes only its tokens after the
prefix it shares with the previous copy, and every segment attends to keys
gathered from its root-to-node path in one varlen call per layer (causal,
aligned to the end; the answer trie the same way). Exact: bf16 error
against a float32 unpacked reference 0.05–0.16, the same as plain SDPA's
bf16 error (0.04–0.20).

4B on an A100 80GB (the machine's GPUs; an earlier estimate here assumed an
H100), 64 ablations + the decision, keep ½: a 6k-token ContractNLI row
14.0 s (flex, whole document per copy) → 11.6 s (kept tokens, FA2, tree;
5.3 s on the row's first measurement with a busy GPU was not repeatable);
MuSiQue (2.7k tokens) 4.6 s; ProofWriter 0.6–0.7 s; Nimble 0.45–0.8 s.
FA2 runs at about 156 TFLOP/s, half the A100's 312 TFLOP/s bf16 peak
(no FP8 on the A100); 17% of the time is unfused elementwise kernels.

Tokens computed, independent masks ÷ prefix tree (measured, 64 ablations):

| | keep ½ | keep 0.8 |
|---|---|---|
| ContractNLI | 1.05–1.08× | 1.08–1.18× |
| MuSiQue | 1.05–1.09× | 1.12–1.14× |
| ProofWriter | 1.10–1.11× | 1.21× |
| Nimble | 1.57–1.83× | 3.29–3.88× |

ContextCite's independent masks diverge within the first ~log2(n)
sentences, so the tree pays only on short documents; a higher keep
probability shares more but keeps more tokens per ablation (the 6k row:
11.6 → 19.0 s at keep 0.8).

torch.compile(dynamic=True) on the FA2 path: 1.25–1.32× on every row after
a one-time ~40 s compile, no recompiles across row lengths (the 6k row
11.6 → 9.1 s; MuSiQue 3.5–4.5 s; ProofWriter and Nimble 0.4–1.0 s).

Ways to go further on long documents, none tried yet: 32 ablations (~2×, if AP holds), copies split across GPUs; and the
ones that cut the work by an order of magnitude: (A) a decision model with
block attention over the document (Block-Attention, Sun et al. 2024),
under which reader-only masking is exact full masking; (B) reader-only
screening of ~10 candidate sentences, then full masking that varies only
those (the prefix tree then shares everything before the first
candidate); (C) recomputing only the most-affected tokens from the
full-context KV (CacheBlend, Yao et al. 2025); (D) gradients of the
answer's logit with respect to per-sentence attention gates (attribution
patching; one forward and one backward pass).

## 9. Live tester (blogs, section 7)

ContextCite only (AT2 removed), the Qwen3-4B decision model
(`runs/qwen3_4b/adapter`), one compiled flex-attention copy for the answer
and the attribution; switches for the ablation count (auto, 64, 256) and
the removal (reader-only, full masking). Nimble's 324 eval rows are
included with Nimble's verified focus and policy evidence as gold
(`dataloader/nimble.py`); about 4.7 of 10 sentences per Nimble row are
gold because the policy quotes are often whole rubric paragraphs.
