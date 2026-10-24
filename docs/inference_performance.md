# Making ContextCite inference faster with SGLang and vLLM

This document explains the serving changes in this repository, why each one
helps, and which part of the two references it comes from:

- **[P]** K. Du et al., *PrefillOnly: An Inference Engine for Prefill-only
  Workloads in Large Language Model Applications*, SOSP '25
  (doi:10.1145/3731569.3764834).
- **[B]** LMSYS blog, *Scaling JEV-like Decision Models with SGLang*,
  2026-09-25 (lmsys.org/blog/2026-09-25-sglang-decision-models).

Read it top to bottom. Sections 1–3 build the mental model, section 4 is the
list of changes, section 5 covers what was not changed and why, and section 6
explains how to measure the result yourself.

---

## 1. The workload: what one ContextCite row costs

### 1.1 One decision

The decision prompt (`src/llm_jev/prompts/decision.j2`, wrapped in the Qwen
chat template by `answers.build_prompt`) ends in the assistant turn at
`Best answer: [`. Option *k* is the token suffix `k]`. The score of option *k*
is the sum of its suffix tokens' log-probabilities (Eq. 1 of LLM-as-Jev):

```
s_k = log p(t_1 | prompt) + log p(t_2 | prompt, t_1) + ...
```

The suffixes form a **trie** (`answers.trie`). With a Qwen tokenizer and 4
options:

```
()      -> legal next tokens {1, 2, 3, 4}       root: needs logits after the prompt
(1,)    -> legal next tokens {]}                depth 1: needs logits after prompt + "1"
(2,)    -> {]}   (3,) -> {]}   (4,) -> {]}
```

With 12 options the trie has a third level, because `1]` and `10]`, `11]`,
`12]` share the `1` node.

So one decision is **1 + (number of non-root nodes)** next-token queries:
one over the full prompt, and the rest over the prompt plus one or two
extra tokens. Each query needs the log-probabilities of a handful of *label
tokens* (the legal next tokens), never a generated token.

### 1.2 One ContextCite row

ContextCite (`context_cite.py`) splits the row into S sentences, draws n
random keep-masks (each sentence kept with p = 0.5;
`masked_cite.ablation_count` gives n = 2·S rounded up to a power of two,
clipped to [32, 256]), rebuilds the prompt for every mask, and scores the
originally predicted answer on each. Through SGLang
(`serving/sglang_client.attribute`) that is

```
(n + 1) prompts  x  (1 + #trie nodes) label queries
```

For S = 20 sentences and 4 options: 65 prompts × 5 queries = 325 queries.
The 65 prompts are each a full prefill of a few thousand tokens. The 260
depth-1 queries each add one new token.

### 1.3 It is a *prefill-only* workload

[P §2.3] defines a **prefill-only request** as one where the model produces
the probability distribution of a single output token. The engine runs the
prefill phase and stops. That describes every query above. Two
consequences follow from the paper:

1. **The work is compute-bound, not memory-bound** [P §2.3, §6.1]. A decode
   step reads all weights to produce one token per sequence, so it is
   memory-bandwidth-bound, and engines batch many sequences to amortise the
   weight reads. A prefill of a few thousand tokens already gives the GPU
   large matrix multiplies to run. Batching more prefills together does not
   raise throughput. PrefillOnly runs requests one at a time for this
   reason [P §6.1].
2. **The cost of a request is known before it runs** [P §1, §6]. With output
   length fixed at one, the job completion time depends only on the
   number of tokens that must be prefilled. That lets the scheduler use
   *shortest job first*, which is impossible for chat requests of unknown
   output length.

[B] arrives at the same place from the API side. A decision model needs "a
small, structured signal rather than generated prose", and the server
should return the requested label scores directly (`/v1/score` with
`label_token_ids`), not top-k logprobs of a generation.

---

## 2. Background you need for the changes

### 2.1 Prefill, KV cache and prefix (radix) caching

During prefill every layer computes keys and values for every prompt token
(the **KV cache**). A later request whose token sequence starts with the
same tokens can reuse those keys and values and prefill only the rest. SGLang
keeps finished requests' KV in a **radix tree** keyed by token ids (the
*radix cache*). vLLM does the same with hashed fixed-size blocks
(`--enable-prefix-caching`). Both evict least-recently-used entries when
memory runs out.

Prefix reuse only works for an identical **prefix**. One changed token in
the middle invalidates everything after it, because every later token
attended to it.

### 2.2 How much of a ContextCite row can prefix caching save?

An ablated prompt is identical to another one only up to the **first
sentence where their masks differ**. The question, options and answer
prefix come *after* the document, so they are never shared across
different masks. The best any cache can do is to compute each distinct
token-prefix once, which is the size of the prefix tree over all the row's
prompts. Computed for 28-token sentences, an 8-token template head and a
120-token tail, with the repository's own `ablation_masks` and
`ablation_count`:

| sentences S | prompts n+1 | tokens, no cache | tokens, ideal prefix cache | saved | of which: template head only |
|---:|---:|---:|---:|---:|---:|
| 5  | 33  |   6,660 |   3,292 | 50.6 % | 3.8 % |
| 10 | 33  |   8,844 |   6,424 | 27.4 % | 2.9 % |
| 20 | 65  |  26,072 |  21,388 | 18.0 % | 2.0 % |
| 40 | 129 |  87,744 |  76,192 | 13.2 % | 1.2 % |
| 80 | 257 | 320,848 | 294,076 |  8.3 % | 0.6 % |

Three things to learn from this table:

- **The savings are real but bounded, and shrink with S.** With n random
  masks, roughly the first log2(n) sentences are shared by several masks.
  After that almost every mask is unique.
- **For short rows they are large.** With S = 5 there are only 2^5 = 32
  distinct masks for 32 ablations, so many prompts repeat *exactly* and are
  served almost entirely from cache.
- **They are only reached if the shared prefix is still cached when the
  request that needs it runs.** That depends on the **order** in which the
  server runs the prompts, which is the scheduling question in section 3.

The depth-1 trie queries are the opposite case: they share the *entire*
prompt with their root query, so they cost one token each, *if* the
prompt's KV is still in the cache when they arrive.

### 2.3 Chunked prefill and CUDA graphs

- **Chunked prefill** splits a long prompt (or a batch) into chunks of
  `--chunked-prefill-size` tokens, prefilled in successive steps. It exists
  to interleave decode steps of other requests with long prefills (smoother
  inter-token latency) and to cap activation memory. For a long prompt it
  replaces one attention call over all tokens with several calls that
  re-read the earlier chunks' KV. [P §2.6] measured **14 % lower
  end-to-end throughput** for a 20,000-token input chunked at 512 tokens.
- **CUDA graphs** record a fixed sequence of GPU kernels once and replay it
  to avoid CPU launch overhead. SGLang captures them for decode batch sizes
  at startup, which takes time and reserves GPU memory. A score request
  never decodes, so decode graphs are never replayed.

---

## 3. Scheduling prefill-only requests with a prefix cache [P §6]

This is the core idea of the paper, and it applies directly to ContextCite.

### 3.1 The three policies

Suppose the GPU runs one request per step and the waiting queue holds
several. Which one should run next?

- **FIFO** (first come, first served): SGLang's default `fcfs`. It ignores
  both cost and cache.
- **Shortest prefill first (SPF), estimated at arrival**: classic shortest
  job first. It minimises mean latency when costs are fixed. With a prefix
  cache, costs are *not* fixed. A request's prefill time drops when a
  related request has just written its prefix to the cache, and rises again
  when that prefix is evicted.
- **SPF with continuous prefill-time estimation** (PrefillOnly, Algorithm 1):
  at **every** scheduling step, re-estimate every waiting request's cost
  against the cache *as it is now*, and pick the smallest:

  ```
  for each waiting request r:
      cost(r) = T_prefill(n_input(r), n_cached(r)) - λ · T_queue(r)
  run argmin cost
  ```

  The paper's proxy is `T_prefill ≈ n_input − n_cached`, the number of
  **uncached tokens**. On an A100 with Qwen-32B-FP8 its Pearson correlation
  with the measured prefill time is **0.987**, and it needs no profiling.
  The `−λ·T_queue` term ages waiting requests so long ones are not starved
  (a larger λ trades mean latency for tail latency).

### 3.2 The paper's example, worked through

Requests A and B arrive first, then C and D while A runs. Lengths satisfy
A < C < B < D. A and D share a prefix, B and C share another, and the cache
holds one prefix at a time. (Section 11 of the blog, *Faster inference*,
replays this example.)

| policy | order | cache hits |
|---|---|---|
| FIFO | A, B, C, D | 1 (C hits B; D misses because B evicted A's prefix) |
| SPF at arrival | A, C, B, D | 1 (C and D were costed as misses on arrival; B hits C) |
| SPF, re-estimated | A, D, C, B | **2** (after A runs, D's cost drops, so D runs while A's prefix is cached; then C, then B hits C) |

The re-estimated scheduler wins because it notices *immediately* when a
request has become cheap and runs it before the cache entry that made it
cheap is evicted.

### 3.3 Why this maps onto ContextCite

A ContextCite row puts dozens to hundreds of related prompts into the queue
at the same moment. As soon as one prompt has run, every waiting prompt that
shares its leading sentences, and every depth-1 trie query of that prompt,
gets cheaper. Re-estimating each step lets the scheduler pick those up
first, while their prefix is the most recently used entry in the cache.

### 3.4 What SGLang 0.5.22 implements

SGLang 0.5.22 (the version `make sglang-install` pins) ships this policy as
`--schedule-policy shortest-prefill-first`. Reading
`sglang/srt/managers/schedule_policy.py`:

- `calc_priority` runs before every prefill scheduling step. For
  cache-aware policies it re-matches each waiting request against the radix
  tree (`_compute_prefix_matches`). That is the paper's continuous
  estimation.
- The cost is `_shortest_prefill_work(r) = len(input) − num_matched_prefix_tokens`,
  exactly the paper's uncached-token proxy.
- `_sort_by_shortest_prefill` sorts by
  `(temporarily_deprioritized, uncached_work, wait_queue_entry_time)`.
  Waiting time is only a tie-break here, not the paper's `−λ·T_queue`
  aging, so very long prompts can wait behind a stream of short ones. That
  is acceptable for a batch job such as attributing an evaluation split.
- **In-batch prefix deduplication.** A request whose cached prefix is short
  (≤ `IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD`, default 32 tokens) but which
  shares ≥ 32 tokens with *another waiting request* is deprioritised. Two
  requests with the same uncached prefix then do not both compute it in
  the same batch: one runs first and the other hits the cache on the next
  step.
- The policy **requires the radix cache** (it raises an error otherwise).
- Unlike `lpm` (longest prefix match), it does **not** fall back to FCFS when
  the queue exceeds `WAITING_QUEUE_PREFIX_MATCH_MAX = 128` requests. The
  client keeps up to 8 chunks per row in flight (64 prompts or 256 trie
  items), and several rows at once (`--rows-in-flight`), so the queue
  regularly passes 128. At that size `lpm` would silently degrade to FIFO.

---

## 4. The changes

Each entry gives what changed, the mechanism, and the source it comes from.

### 4.1 SGLang server (`scripts/sglang_serve.sh`)

Before:

```bash
--chunked-prefill-size 32768 --max-prefill-tokens 65536
# default --schedule-policy fcfs
```

After:

```bash
--schedule-policy shortest-prefill-first \
--chunked-prefill-size -1 --max-prefill-tokens "$CONTEXT" \
--disable-decode-cuda-graph
```

Every value can still be overridden: `POLICY=fcfs`, `PREFILL=65536`, or
extra flags appended after the positional arguments.

**S1. `--schedule-policy shortest-prefill-first`** [P §6, Algorithm 1]. This
is section 3. FIFO runs prompts in arrival order and pays for every shared
prefix once per batch it lands in. SPF with per-step re-estimation runs
whatever has become cheapest, which is how the savings in table 2.2 get
realised. It also helps when several rows are in flight
(`--rows-in-flight`, below): short rows and cache-hitting queries run first,
which lowers mean latency, the classic property of shortest job first.

**S2. `--chunked-prefill-size -1`** (chunked prefill off) [P §2.6; B
appendix]. Our prompts are at most `CONTEXT` = 32k tokens and fit in one
prefill. Chunking would only split attention into several calls (the
paper's 14 % throughput loss) and add scheduler steps. Both server
configurations [B] benchmarks for scoring (MIS, and Setwise/Fused-Choice)
run with `--chunked-prefill-size -1`. Chunked prefill matters when decode
requests share the GPU, and here they never do.

The cost of turning it off is activation memory, the paper's §4 point. A
32k-token prefill materialises the MLP's gate/up output for all tokens at
once. For Qwen3-4B that is 2 × 9,728 values per token, about 1.3 GB in
bf16 at 32k tokens, roughly 10× one layer's KV. `--mem-fraction-static
0.85` leaves 15 % of the GPU for such temporaries. If a smaller GPU runs out
of memory on the longest prompts, lower `MEM` or `CONTEXT` rather than
re-enabling chunking. PrefillOnly's *hybrid prefilling* removes this peak by
chunking only the MLP while keeping attention whole (section 5).

**S3. `--max-prefill-tokens $CONTEXT`** (32k instead of 64k) [P §6.1]. This
bounds the tokens prefilled in one scheduling step. Because prefill is
compute-bound, a 32k-token step already saturates the GPU, and halving the
step size costs little throughput. It doubles the number of scheduling
decisions per row, and each decision re-estimates costs against a fresher
cache (S1). Prompts that share a prefix are also less likely to be prefilled
side by side in one step, where neither can reuse the other. SGLang raises
this bound to the context length if set lower, so `CONTEXT` is the
smallest value that has any effect.

**S4. `--disable-decode-cuda-graph`** (prefill-only property [P §2.3]).
`/v1/score` requests finish after prefill, so decode graphs are never
replayed. Not capturing them shortens server startup and returns their
memory pool to the KV cache, which means more room for prefixes (S1 only
helps while they stay cached). Prefill-phase graphs are left at SGLang's
default.

**S5. Kept: the radix cache and the triton attention backend.** The radix
cache is what S1 schedules around, and the depth-1 trie queries depend on
it. The triton backend stays because FlashInfer JIT-compiles with the system
`nvcc` (see the script's comment). It is also why MIS is not used
(section 5).

### 4.2 SGLang client (`src/llm_jev/serving/sglang_client.py`)

**C1. The full prompt is scored together with the ablations.** Before,
`attribute` first called `predict` (two serial round trips: root, then
depth 1) and only then sent the n ablations. The full prompt is just the
all-ones mask, and its scores are needed only after everything returns:
`answer = argmax s` is applied to the ablation scores at the end. It now
goes in as prompt 0 of the same batch. This removes one full prefill and two
round trips from every row's critical path, and the full prompt shares
prefixes with about half the ablations, so the scheduler can use it as a
cache seed.

**C2. Trie depths are pipelined per chunk instead of per row.** Before:

```
send all n root prompts (chunks of 32, 8 threads)  ->  wait for ALL
send all depth-1 items                              ->  wait for ALL
```

Now each chunk of 8 prompts is one unit of work on the thread pool:

```
chunk i: send its 8 roots -> wait -> send its depth-1 items -> wait -> (depth 2 ...)
```

Two effects:

- **Overlap.** Chunk *i*'s one-token depth-1 queries are queued while later
  chunks are still prefilling, instead of waiting for the slowest root of
  the whole row. The GPU is never idle between the two phases.
- **Cache hits before eviction** [P §6.2–6.3]. A depth-1 query arrives
  right after its prompt was written to the radix cache, while that entry is
  the most recently used. Under SPF its cost is 1 uncached token, so it is
  scheduled at the very next step. Before, depth-1 queries waited for
  the whole row. With several rows in flight (C3) and a full cache, the
  prompt could be evicted by then, and the query would silently re-prefill
  the whole prompt.

The barrier *within* a chunk (depth 1 only after its roots return) is
deliberate. Sent at the same time, a root and its depth-1 items have the
same uncached prefix and could be prefilled side by side in one batch,
which costs (1 + K) prompts' worth of compute instead of one.

Results are unchanged: the same (sequence, label) pairs are requested, and
section 6.1 checks this.

**C3. Rows in flight and a latency summary** [B, "measure serving
performance at the intended load"]. `llm-jev-sglang-cite` attributed rows
one after another. Between rows, and during each row's last chunks, the GPU
was under-used. `--rows-in-flight` (default 4) keeps several rows queued so
the scheduler always has work, and with S1 the cheapest work runs first. The
run now ends with

```
N rows in W s (R rows/s, 4 in flight): row seconds p50 …, p95 …
```

Following [B], report **throughput (rows/s) together with tail latency
(p95)**, not one number. With `--rows-in-flight 1` the per-row seconds are
unloaded latency. With more rows in flight they include queueing behind
other rows, which is exactly the trade-off [B]'s load sweeps show.

### 4.3 vLLM labelling server (`src/llm_jev/span_generation/`)

The best-of-N labeller uses vLLM for two jobs: generating citation sets
(decode-heavy, unchanged) and scoring ablations (`best_of_n.option_scores`,
prefill-only).

**V1. Ablation scoring: `echo=True` prompt logprobs → `logprob_token_ids` per trie node.**

Before, every option *k* of every ablated prompt was one completion request
for `prompt + suffix_k` with `echo=True, logprobs=0`. The server returned
the log-probability of every prompt token, and the client summed the
suffix's. In vLLM 0.32 this has two costs:

1. **It bypasses the prefix cache.** `SamplingParams.__post_init__` sets
   `skip_reading_prefix_cache = True` whenever `prompt_logprobs` is
   requested, which `echo` with logprobs does. The reason is that cached
   tokens are never recomputed, so their logprobs would be missing. So the K
   requests for one ablated prompt each re-prefilled the full prompt,
   despite `--enable-prefix-caching`.
2. **It computes logits at every prompt position.** Prompt logprobs need the
   LM head (hidden × vocab, about 150k for Qwen) and a log-softmax at
   *every* position, not only the last. Only the suffix's two values were
   used.

After, `option_scores` walks the trie like the SGLang client: one request per
node, `prompt + node` with `max_tokens=1` and
`logprob_token_ids=<legal next tokens of that node>` (with
`return_tokens_as_token_ids` so keys are ids, not decoded strings). The root
goes first. Deeper nodes follow once it has returned, so they find the
prompt in the prefix cache. The sampler gathers exactly those ids from the
full-vocabulary log-softmax of the last position (`logprobs_mode`
`raw_logprobs`, the default), so the values are the same normalised
log-probabilities as before.

This is [B]'s first point in vLLM form: **make the output contract
explicit**. Ask for the label tokens you need instead of reading them out
of an echo or a top-k list. Per ablated prompt with K options and length L:

| | prefill tokens | positions through the LM head | prefix cache |
|---|---|---|---|
| before | K · L | K · L | not read |
| after | ≈ L + K | 1 + K | read |

With K = 4 that is about 4× fewer prefill tokens and over 1000× fewer
LM-head rows for the scoring phase. Requires vLLM ≥ 0.32 (`logprob_token_ids`
on `/v1/completions`). `serve.sh` already allows 20 logprobs per request,
more than a node's ≤ 11 legal tokens.

---

## 5. What was considered and not done, and why

**Multi-item scoring (MIS)** [B]. With `--enable-mis`, a `/v1/score` request's
`query` and all its `items` are packed into one sequence,
`query <d> item1 <d> item2 <d> …`. A FlashInfer custom mask makes each item
attend only to the query and to itself, so the query is computed once per
request. In [B]'s pointwise benchmark this keeps p95 latency nearly flat
from 2 to 16 candidates (Qwen3-8B, 16 options: 20.6 ms MIS vs 53.1 ms SIS).
For us:

- *Ablations don't fit it.* MIS shares a **query prefix** across items. Our
  ablations differ in the middle of the document, so there is no common
  query beyond the template head (table 2.2).
- *The trie does fit it*: `query = prompt[:-1]`, `items = [prompt[-1:] + node
  for node in trie]` would score a whole decision in one sequence. The score
  is read at the position before each delimiter, so an empty item cannot
  stand for the root. Hence `prompt[-1:]` is moved into each item.
- *But MIS forces the radix cache off* (`handle_multi_item_scoring`
  disables it, along with chunked prefill and CUDA graphs). That gives up
  the cross-ablation sharing that S1 and C2 exploit, and SPF cannot run
  without the cache.
- *It needs FlashInfer*, which this repo avoids (S5).
- *Exactness must be checked first.* In 0.5.22, item tokens get positions
  `len(query) + j` (correct), but the packed sequence contains a placeholder
  delimiter token (id 9999) at each item boundary. Before trusting MIS
  scores as ContextCite targets, compare them against SIS on real rows.

MIS is the right tool for **serving decisions** (one prompt, many
candidates, no ablations) and worth an A/B there. It is not the default for
attribution.

**Hybrid prefilling and suffix-KV discarding** [P §4–5]. These raise the
*maximum input length* a GPU can prefill. Hybrid prefilling processes the
MLP chunk by chunk while keeping attention whole. Suffix-KV discarding drops
the KV of tokens beyond what the cache can hold. With a 4B model, 32k-token
prompts and an 80 GB GPU, input length is not our bottleneck: one prompt's KV
is about 4.7 GB (Qwen3-4B: 36 layers × 8 KV heads × 128 dims × 2 × 2 bytes ≈
144 KiB per token). Neither SGLang nor vLLM exposes these as flags; the
paper's implementation is a vLLM fork. Both become relevant if this moves to
a much larger model, much longer documents or small GPUs (the paper's L4
setting).

**Setwise / Fused-Choice scoring** [B]. These change *what the model is
asked*: all options at one answer boundary, or a marker per candidate.
Our decision prompt already shows every option and reads one answer
boundary (the trie root), which is the Fused-Choice shape. Switching to
Setwise would change the trained output contract, not just the serving
speed.

**Prefill/decode disaggregation** [P §2.4]. Relevant for the vLLM labeller
only (generation and scoring in one server). Splitting them needs two pools
of GPUs. Fixing V1 removes most of the scoring load instead.

---

## 6. Verification and measurement

### 6.1 What was checked here (no GPU in the development container)

`equiv_test.py` (in the session scratchpad, not committed) runs the **old
and new clients** against a fake HTTP server implementing `/v1/score` and
`/v1/completions`. Its log-probability for (sequence, token) is a fixed
hash of both, so two clients that request the same pairs must return
identical numbers. It also counts the tokens a perfect prefix cache would
have to prefill.

On a 14-sentence row (character-level tokens, so ~640-token prompts):

| check | 4 options | 12 options (trie depth 3) |
|---|---|---|
| SGLang `attribute`: answer, q, bias, every weight w | identical | identical |
| SGLang score items (old / new) | 165 / 165 | 429 / 429 |
| vLLM `rewards`: q(gold), necessity, sufficiency, reward | identical | identical |
| vLLM tokens prefilled, old (echo, cache bypassed) | 17,956 | 63,045 |
| vLLM tokens prefilled, new (ideal prefix cache) | 2,061 (8.7× fewer) | 2,127 (29.6× fewer) |

The SGLang client sends the same items as before; what changed is *when*
they are sent (C1, C2), which only the GPU timing in 6.2 can show. The vLLM
change is visible in token counts alone, and it grows with the number of
options because the old path re-prefilled the prompt once per option.

The blog section's FIFO / SPF / re-estimated simulation reproduces the
paper's Fig. 5 counts (1, 1 and 2 cache hits).

**Not measured:** GPU wall-clock time. The flags were checked against the
0.5.22 and vLLM 0.32 sources, not timed. Section 6.2 shows how to time them.

### 6.2 How to measure the speed-up yourself

Use the same rows, the same model, one GPU and one server at a time. Vary
one thing per run:

```bash
make sglang-serve BASE=Qwen/Qwen3-4B ADAPTER=runs/qwen3_4b/adapter GPU=0 PORT=30000
llm-jev-sglang-cite --url http://127.0.0.1:30000 --tokenizer runs/qwen3_4b/merged \
    --per 10 --rows-in-flight 1 --out runs/cite_new_unloaded.jsonl
llm-jev-sglang-cite ... --rows-in-flight 4 --out runs/cite_new_loaded.jsonl
make sglang-stop PORT=30000

# the old scheduler and batching, everything else equal
POLICY=fcfs PREFILL=65536 scripts/sglang_serve.sh runs/qwen3_4b/merged 0 30000 --chunked-prefill-size 32768
```

To time the old *client* as well, check out the previous
`sglang_client.py` from git and run the same commands.

Read, for each run:

- the closing line: **rows/s** and **p50 / p95 row seconds**;
- the server log (`logs/sglang_30000.log`): `#cached-token` against
  `#new-token` on the prefill lines is the realised prefix-cache hit
  rate, the quantity S1–S3 and C2 are meant to raise;
- that the attributions match: `answer`, `q` and the weights `w` per row in
  the two JSONL files should agree up to GPU numerics.

As [B]'s appendix warns: record versions and server arguments
(`curl :30000/get_server_info`), repeat near saturation, and make sure the
client is not the bottleneck. Here that means `--rows-in-flight` and the
8 request threads per row.

---

## 7. Summary

| # | change | source | mechanism |
|---|---|---|---|
| S1 | `shortest-prefill-first` | P §6 | re-estimate uncached work each step; run cache hits before eviction |
| S2 | chunked prefill off | P §2.6, B | one attention call per prompt; no decode to interleave |
| S3 | prefill batch 64k → 32k | P §6.1 | prefill is compute-bound; more, fresher scheduling decisions |
| S4 | no decode CUDA graphs | P §2.3 | score requests never decode; memory goes to the prefix cache |
| C1 | full prompt in the ablation batch | P §6 | one prefill and two round trips off each row's critical path |
| C2 | per-chunk trie pipelining | P §6.3 | depth-1 queries hit a just-written prefix; no idle gap |
| C3 | rows in flight, p50/p95/rows-s | B | keep the queue full; measure throughput and tail latency |
| V1 | vLLM `logprob_token_ids` per trie node | B | prefix cache usable again; LM head on 1 + K rows, not K · L |
