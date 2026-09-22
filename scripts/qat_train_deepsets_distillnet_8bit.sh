#!/bin/bash
# Real QAT fine-tune of the DeepSets "distillnet"-size (10,981 param)
# top-tagging student: warm-started from the existing float KD checkpoint
# (distill_top_deepsets_distillnet_scratch_a05_T4), Linear layers replaced
# with Brevitas QuantLinear (8-bit weight + activation), fine-tuned for 15
# short epochs against the SAME distillation recipe (alpha=0.5/beta=0.5/T=4
# vs fine_tune_top_l) the base checkpoint was trained with -- quantization is
# the only new variable. See qat_deepsets.py for the full procedure/rationale
# (standalone script, does not edit network.py/train.py/cli.py/utils.py, so
# it can't affect the ongoing spread-queue training in the other env).
#
# CRITICAL: uses omnilearned-fpga/env (has Brevitas), NOT omnilearned-clean/env.
#
# Timing: single-GPU login-node smoke test measured ~504s/epoch at
# --iterations 100 -> extrapolated ~2hrs for the full 1000-iter x 15-epoch
# run. Wall time per epoch is roughly independent of node/GPU count in this
# framework (--iterations is per-rank, DDP ranks step in lockstep), so 4
# nodes doesn't speed this up -- it's used for recipe parity (same aggregate
# per-epoch data coverage as the original a05_T4 training), not speed.
#
# NOTE: qat_deepsets.py always warm-starts from the float --tag checkpoint,
# not from its own --save-tag output -- there is no resume-from-partial-QAT
# logic yet. ~2hrs estimated total, comfortably under the 240-min interactive
# cap, so this is run as a single one-shot salloc (no retry-loop wrapper): if
# it gets preempted/killed early, rerunning this script restarts the 15-epoch
# QAT fine-tune from the float checkpoint again, wasting the partial progress
# but not corrupting anything.
#
# Run inside an salloc GPU interactive job, e.g.:
#   salloc -C gpu -q interactive -t 240 --nodes 4 --ntasks-per-node 4 \
#          --gpus-per-node 4 -A m3246 bash qat_train_deepsets_distillnet_8bit.sh

module load conda
conda activate /global/homes/t/twamorka/omnilearned-fpga/env
module load pytorch

export MASTER_ADDR=$(hostname)
export NCCL_TIMEOUT=600000
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

cmd="/global/homes/t/twamorka/omnilearned-fpga/env/bin/python tools/quantize/qat_deepsets.py \
  --tag distill_top_deepsets_distillnet_scratch_a05_T4 \
  --size distillnet --bits 8 \
  --save-tag qat_top_deepsets_distillnet_a05_T4_8bit \
  --epochs 15 --warmup-epoch 1 --lr 5e-5 --wd 0.5 \
  --batch 128 --iterations 1000 --num-workers 4 \
  --teacher-dir /pscratch/sd/t/twamorka/omnilearned/teacher_logits/companion_fine_tune_top_l \
  --teacher-tag fine_tune_top_l \
  --distill-alpha 0.5 --distill-beta 0.5 --distill-t 4.0"

set -x
srun -l -u \
    bash -c "
    source export_ddp.sh
    $cmd
    "
