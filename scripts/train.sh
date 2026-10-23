#!/usr/bin/env bash
# Launches one training run in the background with Accelerate (FSDP2 by default).
#
#   scripts/train.sh <model> <run dir> [gpus] [extra train.py args ...]
#   scripts/train.sh Qwen/Qwen3-0.6B runs/qwen3_0.6b 2
#   scripts/train.sh Qwen/Qwen3-4B runs/qwen3_4b 0,1,2,3
#   LR=5e-5 EVAL_EVERY=75 scripts/train.sh Qwen/Qwen3.5-4B runs/qwen3.5_4b 4,5,6,7
#
# Every setting below can be overridden from the environment; the defaults
# are the paper's (Li & Wagle, arXiv 2610.02076v3, Table 2).
# Output: <run dir>/ (adapter, eval_*.json, train.log, tb/) and <run dir>.log
# (stdout/stderr). Prints the launcher's process id.
set -euo pipefail
cd "$(dirname "$0")/.."

DATA=${DATA:-data/splits/reason_long_nimble}  # local folder (Parquet) or Hub dataset id
BATCH_SIZE=${BATCH_SIZE:-32}               # rows per optimizer step, over all GPUs
LR=${LR:-1e-4}                             # peak learning rate
WARMUP_STEPS=${WARMUP_STEPS:-100}          # linear warmup
MIN_LR_RATIO=${MIN_LR_RATIO:-0.1}          # then linear decay to this x peak
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}         # AdamW
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}        # gradient clipping
LAM=${LAM:-1.0}                            # KL anchor weight (lambda)
LORA_RANK=${LORA_RANK:-16}                 # alpha = 2 x rank
MAX_TOKENS=${MAX_TOKENS:-16384}            # longer rows are dropped
EVAL_EVERY=${EVAL_EVERY:-0}                # 0: evaluate before and after only
EVAL_PER=${EVAL_PER:-100}                  # eval rows per dataset
SEED=${SEED:-20260922}
ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-configs/accelerate_fsdp2.yaml}  # or accelerate_ddp.yaml
PORT=${PORT:-$((29500 + RANDOM % 1000))}

model=${1:?model id}; out=${2:?run dir}; gpus=${3:-0}; shift $(( $# < 3 ? $# : 3 ))
nproc=$(( $(tr -cd , <<< "$gpus" | wc -c) + 1 ))
mkdir -p "$(dirname "$out")"
CUDA_VISIBLE_DEVICES=$gpus nohup .venv/bin/accelerate launch \
    --config_file "$ACCELERATE_CONFIG" --num_processes "$nproc" \
    --main_process_port "$PORT" -m llm_jev.train.train \
    --model "$model" --out "$out" --data "$DATA" \
    --batch-size "$BATCH_SIZE" --lr "$LR" --warmup-steps "$WARMUP_STEPS" \
    --min-lr-ratio "$MIN_LR_RATIO" --weight-decay "$WEIGHT_DECAY" \
    --max-grad-norm "$MAX_GRAD_NORM" --lam "$LAM" --lora-rank "$LORA_RANK" \
    --max-tokens "$MAX_TOKENS" --eval-every "$EVAL_EVERY" --eval-per "$EVAL_PER" \
    --seed "$SEED" "$@" > "$out.log" 2>&1 &
echo "training $model -> $out on GPUs $gpus (pid $!, log $out.log)"
