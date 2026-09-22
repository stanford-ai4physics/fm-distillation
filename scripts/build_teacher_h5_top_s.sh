#!/bin/bash
# Build teacher-logit companion h5 files for the SMALL top-tagging teacher
# (fine_tune_top_s), so it can be used as a KD teacher for the DeepSets student.
#
# Why this is needed: the sharded npz teacher outputs for fine_tune_top_s have
# existed since 2026-05-22/23 (outputs_fine_tune_top_s_top_{train,val,test}_*.npz,
# 16 ranks each, verified to carry `logits` (N,2) fp16 + `sample_keys` (N,2)
# int64, all finite), but training reads the per-source-h5 COMPANION format via
# --teacher-labels-dir, and only companion_fine_tune_top_l was ever built.
# This converts the existing npz shards -- no GPU re-inference needed.
#
# Only train+val are built: train.py's KD path needs teacher logits for the
# splits it trains and validates on. The test split is not read during training
# (evaluation scores the student against true labels, not teacher logits).
#
# The `top` dataset has just 2 source files (train_ttbar.h5, val_ttbar.h5), so
# this is a short CPU job -- uses -C cpu, NOT the GPU interactive allowance, so
# it does not compete with the DeepSets training runs for GPU nodes.
#
# build_teacher_h5.py initializes companions to NaN and runs a coverage pass
# that fails loudly if any row is left unfilled, so a silent partial conversion
# (which would poison the KD loss with NaN) is not possible. --skip-existing
# makes this safe to re-run after an interruption.
#
# Usage (driver runs on the login node; it allocates its own job):
#   screen -dmS build_teacher_s bash build_teacher_h5_top_s.sh
#   screen -r build_teacher_s

set -uo pipefail

REPO=/global/cfs/cdirs/m3246/twamorka/omnilearned_test/OmniLearned
NPZ_DIR=/pscratch/sd/t/twamorka/omnilearned/teacher_logits
OUT_DIR=/pscratch/sd/t/twamorka/omnilearned/teacher_logits/companion_fine_tune_top_s
DATA_PATH=/global/cfs/cdirs/m4567/www/
TAG=fine_tune_top_s

LOG_DIR=/pscratch/sd/t/twamorka/omnilearned/logs/build_teacher_h5_top_s
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date '+%Y-%m-%d_%H-%M-%S')

salloc -A m3246 -C cpu -q interactive -t 01:00:00 --ntasks=1 --cpus-per-task=16 \
    srun --ntasks=1 --cpus-per-task=16 \
    bash -c "export PATH=/global/homes/t/twamorka/omnilearned-clean/env/bin:\$PATH; \
        python3 '$REPO/tools/preprocess/build_teacher_h5.py' \
            --npz-dir '$NPZ_DIR' \
            --tag '$TAG' \
            --data-path '$DATA_PATH' \
            --out-dir '$OUT_DIR' \
            --dataset top \
            --split train,val \
            --skip-existing" \
    2>&1 | tee "$LOG_DIR/build_${TIMESTAMP}.out"
