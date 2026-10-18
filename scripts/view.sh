#!/usr/bin/env bash
# Trained-vs-base viewer for a run, at http://127.0.0.1:<port> (local only).
#
#   scripts/view.sh <run dir> [gpu] [port]
#   scripts/view.sh runs/qwen3.5_4b 2 8782
set -euo pipefail
cd "$(dirname "$0")/.."
run=${1:?run dir}; gpu=${2:-0}; port=${3:-8781}
CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=src exec .venv/bin/python \
    -m viewers.model --run "$run" --port "$port"
