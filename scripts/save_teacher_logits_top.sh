#!/bin/bash
# Save teacher logits from best_model_fine_tune_top_l.pt (large model fine-tuned
# on top tagging) for train and val splits, then build per-sample companion H5 files.
#
# Phase 1: omnilearned evaluate writes sharded NPZ files.
# Phase 2: build_teacher_h5.py merges them into per-source companion H5 files
#          that the distill dataloader reads lazily during training.
#
# Run inside an salloc GPU interactive job:
#   salloc -C gpu -q interactive -t 240 --nodes 1 --ntasks-per-node 4 \
#          --gpus-per-node 4 -A m3246 bash save_teacher_logits_top.sh
#
# Paths are env-overridable so a collaborator can run this against their own
# conda env / checkout / scratch without editing the file:
#   OMNILEARNED_ENV     conda env prefix         (default: twamorka's)
#   OMNILEARNED_REPO    repo checkout            (default: shared m3246 checkout)
#   OMNILEARNED_SCRATCH per-user scratch root    (default: twamorka's pscratch)
#   CHECKPOINT_DIR      dir holding best_model_${TAG}.pt (default: $OMNILEARNED_SCRATCH/checkpoints/)
#   DATA_PATH          source datasets          (default: /global/cfs/cdirs/m4567/www/, world-readable)

set -euo pipefail

module load conda
conda activate "${OMNILEARNED_ENV:-/global/homes/t/twamorka/omnilearned-clean/env}"
module load pytorch

export MASTER_ADDR=$(hostname)
export NCCL_TIMEOUT=600000
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO="${OMNILEARNED_REPO:-/global/cfs/cdirs/m3246/twamorka/omnilearned_test/OmniLearned}"
SCRATCH="${OMNILEARNED_SCRATCH:-/pscratch/sd/t/twamorka/omnilearned}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$SCRATCH/checkpoints/}"
NPZ_DIR="${NPZ_DIR:-$SCRATCH/teacher_logits/top}"
TAG="${TAG:-fine_tune_top_l}"
# Tag-namespaced by default to match run_train.sh's TEACHER_DIR default
# ($TEACHER_ROOT/companion_$TEACHER_TAG) -- without this, two different TAGs
# would collide in one shared dir and --skip-existing would silently keep
# whichever teacher's logits got there first.
COMPANION_DIR="${COMPANION_DIR:-$SCRATCH/teacher_logits/companion_$TAG}"
DATA_PATH="${DATA_PATH:-/global/cfs/cdirs/m4567/www/}"
DATASET=top

mkdir -p "$NPZ_DIR"

evaluate_split() {
    local SPLIT=$1
    echo "=== $(date '+%F %T')  evaluating $DATASET/$SPLIT ==="
    srun -l -u bash -c "
        source $REPO/export_ddp.sh
        omnilearned evaluate \
          -i $CHECKPOINT_DIR \
          -o $NPZ_DIR \
          --save-tag $TAG \
          --dataset $DATASET \
          --path $DATA_PATH \
          --size large \
          --interaction \
          --local-interaction \
          --mode classifier \
          --num-classes 2 \
          --batch 32 \
          --num-workers 4 \
          --dataset-type $SPLIT
    "
    echo "=== $(date '+%F %T')  $DATASET/$SPLIT done ==="
}

# --- Phase 1: generate NPZ shards for train and val ---
evaluate_split train
evaluate_split val

# --- Phase 2: convert NPZ shards -> companion H5 files ---
echo "=== $(date '+%F %T')  building companion H5 files ==="
python3 "$REPO/tools/preprocess/build_teacher_h5.py" \
    --npz-dir "$NPZ_DIR" \
    --tag "$TAG" \
    --data-path "$DATA_PATH" \
    --out-dir "$COMPANION_DIR" \
    --dataset "$DATASET" \
    --split "train,val" \
    --skip-existing

echo "=== $(date '+%F %T')  all done — companions in $COMPANION_DIR/$DATASET/ ==="
