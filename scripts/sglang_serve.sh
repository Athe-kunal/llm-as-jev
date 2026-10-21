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
# Output: logs/sglang_<port>.log; logs/sglang_<port>.pid holds the server's
# process id (its process group: make sglang-stop). Prints the process id.
set -euo pipefail
cd "$(dirname "$0")/.."

MEM=${MEM:-0.85}                     # GPU memory share for weights + KV cache
CONTEXT=${CONTEXT:-32768}            # longest prompt, tokens
# triton: FlashInfer (the default) JIT-compiles its kernels with the system
# nvcc, here CUDA 12.4, against the tool's CUDA 13 torch, and fails
ATTENTION=${ATTENTION:-triton}
SAMPLING=${SAMPLING:-pytorch}
# tokens per prefill batch: a ContextCite row is 64+ prompts of up to ~6k
# tokens (SGLang's default, 8192, fits one or two)
PREFILL=${PREFILL:-32768}

model=${1:?model dir or HF id}; gpu=${2:-0}; port=${3:-30000}; shift $(( $# < 3 ? $# : 3 ))
python=$(uv tool dir)/sglang/bin/python
mkdir -p logs
CUDA_VISIBLE_DEVICES=$gpu setsid nohup "$python" -m sglang.launch_server \
    --model-path "$model" --host 127.0.0.1 --port "$port" \
    --mem-fraction-static "$MEM" --context-length "$CONTEXT" \
    --attention-backend "$ATTENTION" --sampling-backend "$SAMPLING" \
    --chunked-prefill-size "$PREFILL" --max-prefill-tokens $((2 * PREFILL)) "$@" \
    > "logs/sglang_$port.log" 2>&1 &
echo $! > "logs/sglang_$port.pid"
echo "sglang $model on GPU $gpu -> http://127.0.0.1:$port (pid $!, log logs/sglang_$port.log)"
