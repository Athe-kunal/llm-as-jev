#!/usr/bin/env bash
# Trains the loss-term ablations of Qwen3-0.6B, one epoch each, and evaluates
# every adapter on the external benchmarks. One run per GPU at a time; runs
# are claimed by whichever GPU is free. Runs in the foreground.
#
#   scripts/ablate_loss.sh <gpu> [<gpu> ...]
#   scripts/ablate_loss.sh 1 2 3
#
# Output: runs/ablation/<name>/ (adapter, eval_*.json, train.log, tb/,
# eval_benchmarks_final.json), logs in runs/ablation/<name>.log. A run whose
# eval_benchmarks_final.json exists is skipped, so the script can be restarted.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-Qwen/Qwen3-0.6B}
OUT=${OUT:-runs/ablation}
# name:anchors, first the full loss and L_tree alone, then each term added
# alone, then each term left out.
RUNS=(
  full:mass,out,pos  tree_only:none
  mass_only:mass  out_only:out  pos_only:pos
  no_mass:out,pos  no_out:mass,pos  no_pos:mass,out
)
gpus=("$@")
[[ ${#gpus[@]} -gt 0 ]] || { echo "usage: $0 <gpu> [<gpu> ...]" >&2; exit 2; }

worker() {
  local gpu=$1
  local spec name anchors dir pid
  for spec in "${RUNS[@]}"; do
    name=${spec%%:*}; anchors=${spec#*:}; dir=$OUT/$name
    [[ -f $dir/eval_benchmarks_final.json ]] && continue
    mkdir "$OUT/.claim_$name" 2>/dev/null || continue  # another GPU has it
    base=(--skip-base-eval); [[ $name == full ]] && base=()
    pid=$(ACCELERATE_CONFIG=configs/accelerate_ddp.yaml scripts/train.sh \
        "$MODEL" "$dir" "$gpu" --anchors "$anchors" ${base[@]+"${base[@]}"} | grep -o 'pid [0-9]*' | cut -d' ' -f2)
    while kill -0 "$pid" 2>/dev/null; do sleep 30; done
    [[ -d $dir/adapter ]] || { echo "$name: training failed, see $dir.log" >&2; continue; }
    CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python -m llm_jev.evaluation.benchmarks \
        --model "$MODEL" --adapter "$dir/adapter" --out "$dir" --tag final \
        > "$dir/benchmarks_final.log" 2>&1
    echo "$name: done"
  done
}

mkdir -p "$OUT"
rmdir "$OUT"/.claim_* 2>/dev/null || true
for gpu in "${gpus[@]}"; do worker "$gpu" & done
wait
