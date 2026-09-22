#!/bin/bash
# One-off: post-training quantization (8/6/4-bit, percentile-calibrated,
# inputs + weights) of the distilled distillnet+GNN student, to fill the
# missing PTQ cells next to its 8-bit QAT result for the ML4PS paper.
# Same tool and settings as the plain-distillnet PTQ run in
# scripts/paper_compare_job.sh, plus the interaction flags the float run used.
#
#   screen -dmS ptq_gnn bash scripts/ptq_deepsets_gnn_interactive.sh
set -uo pipefail

REPO=/global/cfs/cdirs/m3246/twamorka/omnilearned_test/OmniLearned
TAG=distill_top_deepsets_distillnet_scratch_a05_T4_gnn1_k64
OUT=/pscratch/sd/t/twamorka/omnilearned/logs/ptq_deepsets_gnn
mkdir -p "$OUT"
TS=$(date '+%Y-%m-%d_%H-%M-%S')

echo "[$(date '+%F %T')] requesting 1 GPU node for PTQ of $TAG" | tee -a "$OUT/driver.log"
salloc -C gpu -q interactive -t 45 --nodes 1 --ntasks-per-node 1 --gpus-per-node 1 -A m3246 \
  bash -c "
    module load conda
    conda activate /global/homes/t/twamorka/omnilearned-clean/env
    module load pytorch
    export HDF5_USE_FILE_LOCKING=FALSE
    export PYTHONPATH=$REPO/tools/quantize:\${PYTHONPATH:-}
    cd $REPO
    /global/homes/t/twamorka/omnilearned-clean/env/bin/python tools/quantize/ptq_deepsets.py \
      --tag $TAG --size distillnet --num-interaction-layers 1 --interaction-k 64 \
      --bits 8,6,4
  " 2>&1 | tee "$OUT/ptq_gnn_${TS}.out"
echo "[$(date '+%F %T')] salloc exited ${PIPESTATUS[0]}" | tee -a "$OUT/driver.log"

{
  echo "==== $(date '+%F %T')  $TAG  (8/6/4-bit PTQ, inputs+weights, percentile 0.999) ===="
  grep -E "^Loaded|^Wrapped|-bit PTQ +acc=|^float32" "$OUT/ptq_gnn_${TS}.out"
} | tee -a "$OUT/results.log"
