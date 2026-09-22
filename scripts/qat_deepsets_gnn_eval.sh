#!/bin/bash
# One-shot interactive launcher for qat_deepsets_eval.py on the distillnet+GNN
# QAT checkpoint. Uses the omnilearned-fpga env's python by ABSOLUTE PATH
# (conda activate + bare python has failed here before with
# ModuleNotFoundError; absolute path sidesteps it).
#
# Passes the SAME --num-interaction-layers / --interaction-k the QAT run used
# so the arch (and Brevitas-wrapped state_dict) match before restore.
module load pytorch
# Do NOT set MASTER_ADDR here -- ddp_setup() branches on it; setting it without
# srun/export_ddp.sh also setting RANK/WORLD_SIZE crashes the single-GPU eval.
cd /global/cfs/cdirs/m3246/twamorka/omnilearned_test/OmniLearned
/global/homes/t/twamorka/omnilearned-fpga/env/bin/python tools/quantize/qat_deepsets_eval.py \
    --tag qat_top_deepsets_distillnet_gnn_a05_T4_8bit --size distillnet --bits 8 \
    --num-interaction-layers 1 --interaction-k 64
