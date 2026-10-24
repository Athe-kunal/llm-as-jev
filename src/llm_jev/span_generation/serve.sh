#!/usr/bin/env bash
# Deploys the labelling LLM with vLLM, OpenAI-compatible, on 127.0.0.1. The
# same server proposes citations (chat) and scores the ablations
# (one-token completions with logprob_token_ids: vLLM >= 0.32). Prefix caching
# serves the trie's deeper nodes from their prompt's cache; echoed prompt
# log-probabilities would bypass it.
set -euo pipefail
cd "$(dirname "$0")/../../.."
model=${MODEL:-Qwen/Qwen3.6-27B}
read -ra gpus <<< "${GPUS:-0}"
read -ra ports <<< "${PORTS:-8011}"
tp=${TP:-1}
if (( tp > 1 )); then  # one server over the GPUs, in groups of tp
  groups=(); for ((i = 0; i < ${#gpus[@]}; i += tp)); do groups+=("$(IFS=,; echo "${gpus[*]:i:tp}")"); done
  gpus=("${groups[@]}")
fi
mkdir -p logs
# Do not JIT-compile FlashInfer's sampler with the system nvcc.
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
