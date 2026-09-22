# Test-split eval for the CE-only (no teacher) distillnet+GNN control
# (counterpart of train config top_deepsets_distillnet_gnn_ce). Interaction
# flags must match training so evaluate.py rebuilds the same architecture.
#
# Score:
#   /global/homes/t/twamorka/omnilearned-clean/env/bin/python tools/metrics/compute_metrics_top.py \
#     --indir /pscratch/sd/t/twamorka/omnilearned/eval/top_distill_deepsets/ \
#     --tag train_top_deepsets_distillnet_gnn1_k64_ce_scratch
ARCH=deep-sets
SIZE=distillnet
INTERACTION=0
LOCAL_INTERACTION=0
EXTRA_FLAGS="--num-interaction-layers 1 --interaction-k 64"
SAVE_TAG=train_top_deepsets_distillnet_gnn1_k64_ce_scratch
OUTDIR=/pscratch/sd/t/twamorka/omnilearned/eval/top_distill_deepsets/
