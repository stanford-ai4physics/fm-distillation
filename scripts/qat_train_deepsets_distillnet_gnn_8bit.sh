#!/bin/bash
# Real QAT fine-tune of the DeepSets "distillnet"+GNN top-tagging student
# (~17,446 params: distillnet body + 1 EdgeConv/Interaction message-passing
# block, leading-pT k=64). Warm-started from the float KD checkpoint
# distill_top_deepsets_distillnet_scratch_a05_T4_gnn1_k64 (93.98% acc / AUC
# 0.9859 / rej50 449.7 -- the best DeepSets student so far), Linear layers
# (phi + rho + the edge MLP) replaced with Brevitas QuantLinear (8-bit
# weight + activation), fine-tuned for 15 short epochs against the SAME
# distillation recipe (alpha=0.5/beta=0.5/T=4 vs fine_tune_top_l) the base
# checkpoint was trained with -- quantization is the only new variable.
#
# One-variable change from qat_train_deepsets_distillnet_8bit.sh: adds
# --num-interaction-layers 1 --interaction-k 64 so qat_deepsets.py rebuilds
# the GNN architecture before restore_checkpoint (state_dict keys must match).
# The edge feature extraction (log m / log dR / log kT) has no Linear layers
# so it is left in float -- QAT only touches the MLPs, same as the plain run.
#
# CRITICAL: uses omnilearned-fpga/env (has Brevitas), NOT omnilearned-clean/env.
#
# Timing: plain distillnet QAT measured ~504s/epoch at --iterations 100 ->
# ~2hrs for 1000-iter x 15-epoch. The k=64 O(N^2) message passing adds edge
# MLP cost, so budget a bit more -- still expected under the 240-min
# interactive cap as a single one-shot salloc (no retry loop; qat_deepsets.py
# always warm-starts from the float --tag, so a rerun just restarts cleanly).
#
# Wall time per epoch is roughly independent of node/GPU count here
# (--iterations is per-rank, DDP ranks step in lockstep), so 4 nodes is for
# recipe parity (same aggregate per-epoch data coverage as the float run),
# not speed.
#
# Run inside an salloc GPU interactive job, e.g.:
#   salloc -C gpu -q interactive -t 240 --nodes 4 --ntasks-per-node 4 \
#          --gpus-per-node 4 -A m3246 bash scripts/qat_train_deepsets_distillnet_gnn_8bit.sh

module load conda
conda activate /global/homes/t/twamorka/omnilearned-fpga/env
module load pytorch

export MASTER_ADDR=$(hostname)
export NCCL_TIMEOUT=600000
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

cmd="/global/homes/t/twamorka/omnilearned-fpga/env/bin/python tools/quantize/qat_deepsets.py \
  --tag distill_top_deepsets_distillnet_scratch_a05_T4_gnn1_k64 \
  --size distillnet --bits 8 \
  --num-interaction-layers 1 --interaction-k 64 \
  --save-tag qat_top_deepsets_distillnet_gnn_a05_T4_8bit \
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
