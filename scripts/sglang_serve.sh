#!/usr/bin/env bash
# Serves a decision model with SGLang on 127.0.0.1, in the background (one GPU).
# llm_jev.serving.sglang_client scores decisions and ContextCite ablations
# through its /v1/score API.
#
#   scripts/sglang_serve.sh <model dir or HF id> [gpu] [port] [extra launch_server args ...]
#   scripts/sglang_serve.sh runs/qwen3_4b/merged 0 30000      # merged by llm_jev.serving.merge
#
# SGLang is a uv tool install (`uv tool install --prerelease=allow --python 3.11
# sglang==0.5.22`), outside this project's venv: its torch and transformers
# pins differ from training's. The radix cache stays on: ablated prompts share
# their prefix up to the first dropped sentence, trie items their whole prompt.
# Every request is prefill-only (docs/inference_performance.md): shortest
# uncached prefill first, re-estimated at each step; no chunked prefill; no
# decode CUDA graphs.
# Output: logs/sglang_<port>.log; logs/sglang_<port>.pid holds the server's
# process id (its process group: make sglang-stop). Prints the process id.
set -euo pipefail
cd "$(dirname "$0")/.."

MEM=${MEM:-0.85}                     # GPU memory share for weights + KV cache
CONTEXT=${CONTEXT:-32768}            # longest prompt, tokens
# triton: FlashInfer (the default) JIT-compiles its kernels with the system
# nvcc, which can fail when that CUDA version differs from the one the tool's
# torch was built for; hence the triton backend default.
ATTENTION=${ATTENTION:-triton}
SAMPLING=${SAMPLING:-pytorch}
POLICY=${POLICY:-shortest-prefill-first}
# tokens per prefill batch; SGLang raises it to CONTEXT if lower
PREFILL=${PREFILL:-$CONTEXT}

model=${1:?model dir or HF id}; gpu=${2:-0}; port=${3:-30000}; shift $(( $# < 3 ? $# : 3 ))
python=$(uv tool dir)/sglang/bin/python
mkdir -p logs
CUDA_VISIBLE_DEVICES=$gpu setsid nohup "$python" -m sglang.launch_server \
    --model-path "$model" --host 127.0.0.1 --port "$port" \
    --mem-fraction-static "$MEM" --context-length "$CONTEXT" \
    --attention-backend "$ATTENTION" --sampling-backend "$SAMPLING" \
    --schedule-policy "$POLICY" --chunked-prefill-size -1 --max-prefill-tokens "$PREFILL" \
    --disable-decode-cuda-graph "$@" \
    > "logs/sglang_$port.log" 2>&1 &
echo $! > "logs/sglang_$port.pid"
echo "sglang $model on GPU $gpu -> http://127.0.0.1:$port (pid $!, log logs/sglang_$port.log)"
