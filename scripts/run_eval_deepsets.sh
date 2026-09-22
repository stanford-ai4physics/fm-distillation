#!/bin/bash
# One-shot test-split evaluation of a DeepSets top-tagging checkpoint.
# Grabs a 1-node x 4-GPU interactive allocation and runs
# evaluate_top_distill_deepsets.sh against it, tee'd to a per-tag log dir.
#
#   run_eval_deepsets.sh <save_tag> [size]
#
#   size defaults to "small"; pass "distillnet" for the 10,981-param student.
#
# Run inside a screen session so it survives a disconnect:
#   screen -dmS eval_deepsets bash scripts/run_eval_deepsets.sh <save_tag> [size]
#
# After it finishes, score it the same way every DeepSets run was scored:
#   /global/homes/t/twamorka/omnilearned-clean/env/bin/python tools/metrics/compute_metrics_top.py \
#       --indir /pscratch/sd/t/twamorka/omnilearned/eval/top_distill_deepsets/ \
#       --tag <save_tag>
#
# Known checkpoint tags (see docs/EXPERIMENTS_deepsets_kd.md):
#   distill_top_deepsets_small_scratch_a05_T4_archfix0804   (reference)
#   distill_top_deepsets_small_scratch_a00_b10_T4
#   distill_top_deepsets_small_scratch_a05_wd005_T4
#   distill_top_deepsets_small_scratch_a05_ewpool_T4
#   distill_top_deepsets_distillnet_scratch_a05_T4           (size distillnet)
#   distill_top_deepsets_small_scratch_teachS_a05_T4
#   train_top_deepsets_small_ce_scratch                      (CE-only baseline)

set -euo pipefail

export SAVE_TAG="${1:?Usage: $0 <save_tag> [size]}"
export SIZE="${2:-small}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR=/pscratch/sd/t/twamorka/omnilearned/logs/eval_deepsets_$SAVE_TAG
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date '+%Y-%m-%d_%H-%M-%S')

salloc \
    -C gpu \
    -q interactive \
    -t 60 \
    --nodes 1 \
    --ntasks-per-node 4 \
    --gpus-per-node 4 \
    -A m3246 \
    bash "$SCRIPT_DIR/evaluate_top_distill_deepsets.sh" \
    2>&1 | tee "$LOG_DIR/session_${TIMESTAMP}.out"
