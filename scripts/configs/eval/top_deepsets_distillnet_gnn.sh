# Test-split eval for the distillnet DeepSets student WITH the optional GNN
# message-passing layer (counterpart of train config top_deepsets_distillnet_gnn).
# The interaction flags must match training exactly so evaluate.py rebuilds the
# same architecture before restoring the checkpoint.
#
# Score it the same way as the rest of the DeepSets family:
#   /global/homes/t/twamorka/omnilearned-clean/env/bin/python tools/metrics/compute_metrics_top.py \
#     --indir /pscratch/sd/t/twamorka/omnilearned/eval/top_distill_deepsets/ \
#     --tag distill_top_deepsets_distillnet_scratch_a05_T4_gnn1_k64
ARCH=deep-sets
SIZE=distillnet
INTERACTION=0
LOCAL_INTERACTION=0
EXTRA_FLAGS="--num-interaction-layers 1 --interaction-k 64"
SAVE_TAG=distill_top_deepsets_distillnet_scratch_a05_T4_gnn1_k64
OUTDIR=/pscratch/sd/t/twamorka/omnilearned/eval/top_distill_deepsets/
