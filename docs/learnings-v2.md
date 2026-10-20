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
