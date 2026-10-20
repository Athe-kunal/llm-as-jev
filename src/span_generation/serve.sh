#!/usr/bin/env bash
# Deploys the labelling LLM with vLLM: one OpenAI-compatible server per GPU,
# on 127.0.0.1. The same server proposes citations (chat) and scores the
# ablations (completions with echoed prompt log-probabilities).
#
#   src/span_generation/serve.sh                       Qwen/Qwen3.6-27B, GPUs 2 3, ports 8011 8012
#   MODEL=Qwen/Qwen3-0.6B GPUS="2" PORTS="8013" src/span_generation/serve.sh
#
# vLLM is the uv tool install (`uv tool install vllm==0.29.0 --with
# third_party/vllm-cuda-no-nvml`), outside this project's venv; one server
# per GPU, no tensor parallelism, because the NVIDIA driver mismatch breaks
# NCCL (as the earlier scripts/vllm_serve.sh). Prefix caching makes the K
# option suffixes of one ablated prompt reuse its prefix.
set -euo pipefail
cd "$(dirname "$0")/../.."
model=${MODEL:-Qwen/Qwen3.6-27B}
read -ra gpus <<< "${GPUS:-2 3}"
read -ra ports <<< "${PORTS:-8011 8012}"
mkdir -p logs
export HF_HOME="${QWEN_HF_HOME:-$HOME/.cache/huggingface}" HF_HUB_OFFLINE=1
unset TRANSFORMERS_CACHE HF_HUB_CACHE
export VLLM_USE_FLASHINFER_SAMPLER=0
for i in "${!gpus[@]}"; do
  CUDA_VISIBLE_DEVICES="${gpus[$i]}" setsid nohup vllm serve "$model" \
    --served-model-name labeller --port "${ports[$i]}" --host 127.0.0.1 \
    --max-model-len 32768 --gpu-memory-utilization "${GPU_UTIL:-0.88}" \
    --max-num-seqs "${MAX_SEQS:-256}" --max-logprobs 20 --enable-prefix-caching \
    > "logs/vllm_labeller_gpu${gpus[$i]}.log" 2>&1 &
  echo "GPU ${gpus[$i]} -> http://127.0.0.1:${ports[$i]}/v1 (logs/vllm_labeller_gpu${gpus[$i]}.log)"
done
