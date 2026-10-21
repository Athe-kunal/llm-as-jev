#!/usr/bin/env bash
# Launches one DiffMask probe training run in the background (one GPU).
#
#   scripts/train_diffmask.sh <variant fade|interp> <run dir> [gpu] [extra train_diffmask.py args ...]
#   scripts/train_diffmask.sh fade runs/diffmask_qwen3_0.6b/fade_m0.05 0
#   MARGIN=0.2 scripts/train_diffmask.sh interp runs/diffmask_qwen3_0.6b/interp_m0.2 0
#
# Output: <run dir>/ (probe.pt, train_diffmask.log) and <run dir>.log
# (stdout/stderr). Prints the training process id.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-Qwen/Qwen3-0.6B}
ADAPTER=${ADAPTER:-runs/qwen3_0.6b_lam0.01/adapter}
MARGIN=${MARGIN:-0.05}       # KL budget (nats) between the masked and unmasked decision
STEPS=${STEPS:-2000}         # rows, one per step
LR=${LR:-3e-4}               # probe (and baseline) learning rate
LAM_LR=${LAM_LR:-1e-2}       # dual ascent step for the multiplier

variant=${1:?variant fade|interp}; out=${2:?run dir}; gpu=${3:-0}; shift $(( $# < 3 ? $# : 3 ))
mkdir -p "$(dirname "$out")"
CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=src setsid nohup .venv/bin/python -m train.train_diffmask \
  --model "$MODEL" --adapter "$ADAPTER" --variant "$variant" --margin "$MARGIN" --steps "$STEPS" \
  --lr "$LR" --lam-lr "$LAM_LR" --out "$out" "$@" > "$out.log" 2>&1 &
echo $!
