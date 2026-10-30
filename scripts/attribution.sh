#!/usr/bin/env bash
# Measures ContextCite (full masking on FA2, fused kernels) on the eval rows
# that have gold evidence spans: span AP / top-1, top-k removal drop, LDS and
# seconds per row. Runs in the foreground; extra arguments go to the module
# (e.g. --no-fuse, --spans --split train).
#
#   scripts/attribution.sh [model] [out dir] [gpu] [rows per dataset] [extra args...]
#   scripts/attribution.sh Athekunal/llm-jev-qwen3-0.6b runs/attribution/qwen3_0.6b 0 50
#
# Output: <out dir>/rows.jsonl (per row) and summary.json (means per dataset).
set -euo pipefail
cd "$(dirname "$0")/.."
model=${1:-Athekunal/llm-jev-qwen3-0.6b}; out=${2:-runs/attribution/qwen3_0.6b}
gpu=${3:-0}; per=${4:-50}
export CUDA_VISIBLE_DEVICES=$gpu
.venv/bin/python -m llm_jev.evaluation.attribution --model "$model" \
    --data Athekunal/llm-jev-reason-long --split eval --per "$per" --out "$out" "${@:5}"
