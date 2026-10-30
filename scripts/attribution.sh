#!/usr/bin/env bash
# Compares ContextCite (full masking, FA2) with AttnLRP (logit and contrast
# targets) on the eval rows that have gold evidence spans. Runs in the foreground.
#
#   scripts/attribution.sh [model] [out dir] [gpu] [rows per dataset]
#   scripts/attribution.sh Athekunal/llm-jev-qwen3-0.6b runs/attribution/qwen3_0.6b 0 50
#
# Output: <out dir>/rows.jsonl (per row and method) and summary.json (means).
set -euo pipefail
cd "$(dirname "$0")/.."
model=${1:-Athekunal/llm-jev-qwen3-0.6b}; out=${2:-runs/attribution/qwen3_0.6b}
gpu=${3:-0}; per=${4:-50}
export CUDA_VISIBLE_DEVICES=$gpu
.venv/bin/python -m llm_jev.evaluation.attribution --model "$model" \
    --data Athekunal/llm-jev-reason-long --split eval --per "$per" --out "$out"
