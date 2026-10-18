#!/usr/bin/env bash
# Evaluates a run's adapter (or the base model) on the eval split and on the
# external benchmarks (BoolQ, MMLU, MMLU-Pro, ARC-C, WinoGrande, SciQ,
# Banking77, JevBench). Runs in the foreground.
#
#   scripts/evaluate.sh <model> <run dir> [gpu] [tag]
#   scripts/evaluate.sh Qwen/Qwen3-0.6B runs/qwen3_0.6b 1          # trained
#   scripts/evaluate.sh Qwen/Qwen3-0.6B runs/qwen3_0.6b 1 base     # base model
#
# Output: <run dir>/eval_<tag>.json and benchmarks_<tag>.json (+ examples).
set -euo pipefail
cd "$(dirname "$0")/.."
model=${1:?model id}; out=${2:?run dir}; gpu=${3:-0}; tag=${4:-final}
adapter=()
[[ $tag != base ]] && adapter=(--adapter "$out/adapter")
export CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=src
.venv/bin/python -m evaluation.evaluate --model "$model" "${adapter[@]}" \
    --out "$out" --tag "$tag"
.venv/bin/python -m evaluation.benchmarks --model "$model" "${adapter[@]}" \
    --out "$out" --tag "$tag"
