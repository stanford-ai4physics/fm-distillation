#!/bin/bash
# Evaluate a DeepSets top-tagging student on the test split. Writes per-rank
# logits to OUTPUT_DIR as outputs_<tag>_top_test_<rank>.npz.
#
# Parameterized via env vars: SAVE_TAG (checkpoint tag), SIZE (small |
# distillnet, default small), DATASET_TYPE (default test). Works unchanged
# for CE-only and distilled checkpoints of the same architecture.
# run_eval_deepsets.sh wraps this in the salloc call.
#
# Run inside an salloc GPU interactive job, e.g.:
#   salloc -C gpu -q interactive -t 60 --nodes 1 --ntasks-per-node 4 \
#          --gpus-per-node 4 -A m3246 bash scripts/evaluate_top_distill_deepsets.sh

module load conda
conda activate /global/homes/t/twamorka/omnilearned-clean/env
module load pytorch

export MASTER_ADDR=$(hostname)

CHECKPOINT_DIR=/pscratch/sd/t/twamorka/omnilearned/checkpoints/
OUTPUT_DIR=/pscratch/sd/t/twamorka/omnilearned/eval/top_distill_deepsets/
SAVE_TAG=${SAVE_TAG:-distill_top_deepsets_small_scratch_a05_T4}
SIZE=${SIZE:-small}
DATASET_TYPE=${DATASET_TYPE:-test}

mkdir -p "$OUTPUT_DIR"

cmd="omnilearned evaluate \
    -i $CHECKPOINT_DIR \
    -o $OUTPUT_DIR \
    --save-tag $SAVE_TAG \
    --dataset top \
    --path /global/cfs/cdirs/m4567/www/ \
    --arch deep-sets --size $SIZE \
    --num-classes 2 \
    --batch 128 \
    --num-workers 4 \
    --dataset-type $DATASET_TYPE"

set -x
srun -l -u \
    bash -c "
    source export_ddp.sh
    $cmd
    "
