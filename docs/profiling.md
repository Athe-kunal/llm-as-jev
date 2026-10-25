# Profiling a training step: torch.profiler and the CUDA memory allocator

`make torch-profiler` runs a few real training steps on one GPU and saves what
PyTorch's two profiling tools see. The steps use the same model, LoRA, loss and
data as `scripts/train.sh`. Training itself never profiles: the only change to
the training code is the `record_function` labels in `losses.compute`, and
those do nothing when no profiler is running.

```bash
make torch-profiler                                   # Qwen3-0.6B on GPU 0 -> runs/profile/Qwen3-0.6B
make torch-profiler GPU=2 PROFILE_MODEL=Qwen/Qwen3-4B # another model or GPU
make torch-profiler PROFILE_ARGS="--no-checkpointing --max-tokens 16384" \
    PROFILE_OUT=runs/profile/0.6b_nockpt              # an experiment, in its own folder
```

With the defaults (4 rows per step, rows of at most 4,096 tokens), Qwen3-0.6B
takes about 6 minutes, and most of that is writing the outputs.
`.venv/bin/python -m llm_jev.profiling.profile_step --help` lists every flag.
The memory timeline needs matplotlib: `uv sync --inexact --extra profile`.

## 1. What the script does

`src/llm_jev/profiling/profile_step.py` runs three stages:

| stage | steps | why |
|---|---|---|
| warmup | `--warmup-steps` (3) | The first steps are slow for one-time reasons: `torch.compile` compiles `loss_terms`, cuBLAS creates its handles, and the caching allocator `cudaMalloc`s its first segments. Profile them and you measure start-up, not training. |
| memory | `--memory-steps` (1) | `torch.cuda.memory._record_memory_history` makes the allocator record every alloc and free with its Python stack. Each phase's allocated and peak bytes are also written down. |
| profile | 1 + `--profile-steps` (1) | `torch.profiler` records CPU ops, CUDA kernels, input shapes, memory and stacks. The first step is traced and then discarded, because it carries the profiler's own start-up cost. |

The memory and profile stages run separately because each one slows the other
down. Recording a stack on every allocation would inflate the timings.

One step does the same work as a training step: for each row, the forward with
the adapter (`p_theta_forward`), the forward without it (`p0_forward`),
`loss_terms` and `backward`, then `clip_grad_norm`, `optimizer_step` and
`zero_grad`. These names are `record_function` ranges, so they show up in
every view.

## 2. The outputs, and how to open them

The files are on the GPU server. Open them through your editor's remote
connection, or copy the folder to your laptop:
`scp -r server:~/astarag/jev-llm-as-judge/runs/profile/Qwen3-0.6B .`

| file | open with | what it answers |
|---|---|---|
| `profile.log` | any editor | Rows and token lengths, weight memory, and the time and peak memory of each step. |
| `phases.txt` | any editor | How long each phase takes on the CPU and on the GPU. **Start here.** |
| `ops_by_cuda_time.txt` | any editor | Which ops and kernels use the GPU's time. |
| `ops_by_cpu_time.txt` | any editor | Which ops use the host's time (Python, dispatch, kernel launches). |
| `ops_by_cuda_memory.txt` | any editor | Which ops allocate the most GPU memory in total (bytes churned, not live). |
| `ops_by_input_shape.txt` | any editor | The same tables split by input shape, for example the 3072-wide MLP versus the 1024-wide attention. |
| `ops_by_stack.txt` | any editor | Time per op and per Python call site (8 frames). |
| `stacks_cuda.txt` | `flamegraph.pl stacks_cuda.txt > cuda.svg` | A flame graph: GPU time by Python call stack. |
| `trace.json.gz` | <https://ui.perfetto.dev> (Open trace file) | **The timeline**: every op on the CPU threads and every kernel on the CUDA stream, over time. |
| `memory_timeline.html` | a browser | Live GPU memory over the profiled step, by category: parameters, gradients, activations, optimizer state, temporaries. |
| `memory_snapshot.pickle` | <https://pytorch.org/memory_viz> (drag the file in) | **The allocator's history**: every allocation as a band over time; click one to see the Python stack that made it. |
| `memory_trace.html`, `memory_segments.html` | a browser | The same snapshot, already rendered offline: the trace view, and the segment view (how the reserved blocks are carved up). |
| `memory_phases.json` | any editor | Allocated memory before and after each phase, and the peak inside it. |
| `memory_summary.txt`, `memory_stats.json` | any editor | The allocator's counters: allocated, reserved, non-releasable (fragmentation), number of `cudaMalloc` calls, OOM retries. |

## 3. Concepts you need to read them

**The CPU queues work and the GPU runs it later.** A Python line such as `x @ w`
makes the host dispatch an op and *launch* a kernel onto the CUDA stream, then
return at once. The GPU runs queued kernels in order. That gives two clocks:

- *CPU time* is how long the host spends in an op: Python, dispatch, and the
  launch call.
- *CUDA time* is how long the kernel runs on the GPU.

If the GPU finishes kernels faster than the host can launch them, the GPU
waits, and the trace shows gaps between kernels. This is called
**launch-bound** or **CPU-bound**. If the queue stays full, the work is
**GPU-bound**, and only faster kernels or fewer FLOPs help. Calls such as
`.item()`, `.cpu()`, `print(tensor)` or `torch.cuda.synchronize()` make the
host wait for the GPU to drain. In the trace they look like a long op on the
CPU with nothing launched during it.

**Self time and total time.** An op's *total* time includes its children, and
its *self* time does not. To find hot spots, sort by self time (the tables do).
Total time shows where a whole subtree goes, for example `aten::linear` →
`aten::mm`.

**A record_function range has two rows.** In `phases.txt`, the *cpu range* is
the host time from entering the range to leaving it. The *gpu range* runs from
the first to the last kernel that was launched inside it. Backward's GPU range
is empty because autograd runs backward on its own thread, outside the
range. Look at the autograd thread in the trace instead.

**The caching allocator.** `cudaMalloc` and `cudaFree` are slow and force a
synchronize, so PyTorch requests large *segments* from the driver and splits
them into *blocks* itself. Freed blocks go back to PyTorch's cache, not to the
driver. Two numbers matter:

- `memory_allocated`: bytes in live tensors.
- `memory_reserved`: bytes held from the driver (live tensors plus the cache).
  This is what `nvidia-smi` shows, plus the CUDA context of about 500 MiB.

Reserved minus allocated is cache. Some of it can be unusable
**fragmentation**: free blocks that are too small or split up for the next
request. You can get an OOM while `allocated` is well under the GPU's size.
`memory_segments.html` and "Non-releasable memory" in `memory_summary.txt`
show it. The usual fix is `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

**Where training memory goes.** Live memory during a step is

    weights + LoRA gradients + AdamW state (2 x LoRA, created on the first step)
    + activations saved for backward + temporaries

In this recipe the base weights are frozen, so gradients and optimizer state
exist only for the LoRA parameters. **Gradient checkpointing** keeps only each
decoder layer's input. Backward runs the layer's forward again to rebuild the
rest. That trades about one extra forward of compute for a large cut in
activation memory. `--no-checkpointing` shows what it saves.

## 4. A reading workflow

1. **`profile.log` → `phases.txt`.** Compare the step's CPU range to its GPU
   range. If the GPU range is much shorter, the GPU is idle part of the time.
   Then find which phase is largest.
2. **`ops_by_cuda_time.txt`.** Are GEMMs (`aten::mm`, `*gemm*` kernels) and
   attention (`fmha*`, `flash*`) at the top? Then you are compute-bound, which
   is where you want to be. If elementwise kernels (`aten::mul`, `add`,
   `copy_`, `*elementwise_kernel*`) take a large share, you are paying for
   memory traffic and kernel launches. `torch.compile` fuses these.
3. **`ops_by_input_shape.txt` / `ops_by_stack.txt`.** Find out *which* call
   sites launch the hot ops.
4. **`trace.json.gz` in Perfetto.** Use W/S to zoom and A/D to pan. Click a
   kernel to see the CPU op that launched it (the flow arrow). Look for gaps on
   the GPU stream (launch-bound), long single kernels, and host waits on
   `.item()` and similar calls.
5. **Memory: `memory_phases.json` first, then the snapshot.** Find the phase
   with the largest `peak_above_before_mib`. In memory_viz, find the highest
   point and click the bands under it to get their stacks. Ask whether each
   tensor is needed and whether it is needed *then*.

## 5. What the first run shows (Qwen3-0.6B, 4 rows, mean 598 tokens, checkpointing on)

These numbers come from `runs/profile/Qwen3-0.6B` (2026-10-25) and are
examples of what to look for. Rows differ in length, so each step's numbers
differ too.

- **Weights 1,175 MiB.** That is 0.6B parameters x 2 bytes (bf16) plus the
  fp32 LoRA (10.1M parameters, 38.5 MiB). The LoRA gradients (38.5 MiB) and
  the AdamW state (2 x 38.5 MiB) are small next to the weights.
- **Activations are small.** Per row, the peak above the starting point is
  30-220 MiB (`memory_phases.json`): checkpointing plus short rows. Backward
  peaks slightly higher than forward because each layer is recomputed and its
  gradients exist at the same time. The longest row sets the step's peak.
  Profile with `--max-tokens 16384` to see where a long row puts it.
- **Reserved 2,278 MiB, but only about 1,310 MiB allocated.** The allocator
  kept the high-water mark of the warmup (2,065 MiB) as cache. That is not a
  leak. It is why `nvidia-smi` overstates what the job needs.
- **The GPU is idle about a third of the time.** The profiled step took
  3,680 ms on the CPU but only 2,465 ms on the GPU (`phases.txt`). In
  `p_theta_forward` the CPU and GPU ranges are almost equal (1,177 vs
  1,175 ms), which is the sign of launch-bound work: the GPU finishes each
  kernel and waits for the next launch. With 600-token rows a 0.6B model's
  kernels are tiny. The profiler's own per-op cost makes the CPU side look
  worse than in a normal run.
- **Elementwise work and LoRA overhead.** After `aten::mm` (196 ms) come
  `aten::mul` (87 ms) and `aten::copy_` (85 ms). `ops_by_input_shape.txt`
  shows `copy_` called thousands of times on activations shaped
  `[4, 521, 1024]` and `[4, 521, 3072]`. PEFT casts each LoRA layer's bf16
  input to the adapter's fp32 dtype, and the fp32 LoRA matmuls show up as
  `ampere_sgemm_*` kernels (fp32, no tensor cores) next to the bf16
  `ampere_bf16_s16816gemm_*` kernels. `p0_forward` (adapter disabled, no
  autograd) runs in 325 ms of GPU time, against 1,175 ms for
  `p_theta_forward`. The adapter and its bookkeeping cost more than the base
  model.
- **Recompiles.** `TORCH_LOGS=recompiles make torch-profiler` shows
  `loss_terms` recompiling for new shapes, for example when a row has a
  single gold-path node. That is why the warmup is 3 steps. In real training,
  each new shape class costs a few seconds once.

## 6. Experiments to try

Each one changes one thing. Give each its own `PROFILE_OUT` so you can compare.

| experiment | `PROFILE_ARGS` / variables | what to compare |
|---|---|---|
| checkpointing off | `--no-checkpointing` | Peak in `memory_phases.json` (activations grow with layers x tokens), and step time (no recompute in backward). |
| long rows | `--max-tokens 16384 --batch-size 2` | Attention's share of `ops_by_cuda_time` and the peak. Does the step become GPU-bound? |
| bigger model | `PROFILE_MODEL=Qwen/Qwen3-4B` | Kernels get larger. Does the CPU/GPU gap in `phases.txt` close? |
| allocator | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True make torch-profiler` | Reserved versus allocated, and "Non-releasable memory". |
| first step | `--cold-memory` | The snapshot now includes the first step: AdamW creating `exp_avg` / `exp_avg_sq`, and compilation. |
