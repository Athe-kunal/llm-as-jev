#!/usr/bin/env bash
# Deploys the labelling LLM with vLLM, OpenAI-compatible, on 127.0.0.1. The
# same server proposes citations (chat) and scores the ablations
# (completions with echoed prompt log-probabilities).
#
#   TP=2 GPUS="2 3" PORTS="8011" src/llm_jev/span_generation/serve.sh   one server, tensor parallel over 2 GPUs
#   GPUS="2 3" PORTS="8011 8012" src/llm_jev/span_generation/serve.sh   one server per GPU (data parallel)
#   MODEL=Qwen/Qwen3-0.6B GPUS="2" PORTS="8013" src/llm_jev/span_generation/serve.sh
#
# vLLM is the uv tool install (`uv tool install vllm==0.29.0 --with
# third_party/vllm-cuda-no-nvml`), outside this project's venv. TP > 1 needs
# working NCCL (the earlier driver mismatch broke it; one server per GPU is
# the fallback). Prefix caching makes the K
# option suffixes of one ablated prompt reuse its prefix. Image and video
# inputs are turned off: the prompts are text, and profiling Qwen3.6's
# multimodal path crashes this vLLM build ('NoneType' object has no
# attribute 'size' in determine_available_memory).
set -euo pipefail
cd "$(dirname "$0")/../../.."
model=${MODEL:-Qwen/Qwen3.6-27B}
read -ra gpus <<< "${GPUS:-2 3}"
read -ra ports <<< "${PORTS:-8011 8012}"
tp=${TP:-1}
if (( tp > 1 )); then  # one server over the GPUs, in groups of tp
  groups=(); for ((i = 0; i < ${#gpus[@]}; i += tp)); do groups+=("$(IFS=,; echo "${gpus[*]:i:tp}")"); done
  gpus=("${groups[@]}")
fi
mkdir -p logs
export HF_HOME="${QWEN_HF_HOME:-$HOME/.cache/huggingface}" HF_HUB_OFFLINE=1
unset TRANSFORMERS_CACHE HF_HUB_CACHE
export VLLM_USE_FLASHINFER_SAMPLER=0
for i in "${!gpus[@]}"; do
  CUDA_VISIBLE_DEVICES="${gpus[$i]}" setsid nohup vllm serve "$model" \
    --served-model-name labeller --port "${ports[$i]}" --host 127.0.0.1 \
    --max-model-len 32768 --gpu-memory-utilization "${GPU_UTIL:-0.88}" \
    --max-num-seqs "${MAX_SEQS:-256}" --max-logprobs 20 --enable-prefix-caching \
    --limit-mm-per-prompt '{"image":0,"video":0}' --tensor-parallel-size "$tp" \
    > "logs/vllm_labeller_gpu${gpus[$i]//,/_}.log" 2>&1 &
  echo "GPU ${gpus[$i]} -> http://127.0.0.1:${ports[$i]}/v1 (logs/vllm_labeller_gpu${gpus[$i]//,/_}.log)"
done
