#!/bin/bash
# One-shot interactive launcher for qat_deepsets_eval.py. Uses the
# omnilearned-fpga env's python by ABSOLUTE PATH (not `conda activate` +
# bare `python`) -- that pattern failed for hint_kd_top_smoketest.py earlier
# today with ModuleNotFoundError, root cause not yet nailed down; absolute
# path sidesteps it entirely and is the same pattern the QAT
# training/timing smoke tests used successfully earlier in this session.
module load pytorch
# NOTE: do NOT set MASTER_ADDR here -- ddp_setup() branches on whether it's
# already in the environment. Setting it (without srun/export_ddp.sh also
# setting RANK/WORLD_SIZE) pushes it into the multi-process env:// rendezvous
# path, which then crashes looking for RANK. Leaving it unset triggers the
# correct single-process branch for this single-GPU eval.
cd /global/cfs/cdirs/m3246/twamorka/omnilearned_test/OmniLearned
/global/homes/t/twamorka/omnilearned-fpga/env/bin/python tools/quantize/qat_deepsets_eval.py \
    --tag qat_top_deepsets_distillnet_a05_T4_8bit --size distillnet --bits 8
