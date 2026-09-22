# DeepSets "distillnet" width + one message-passing block, CE-only control:
# the same architecture as top_deepsets_distillnet_gnn.sh trained on hard
# labels with no teacher (DISTILL=0 drops --distill and every --teacher-*/
# --distill-* arg). Answers "how much of distillnet+GNN's 93.98% is KD and
# how much is the block" -- the KD-vs-scratch comparison for the paper.
#
# Recipe otherwise identical to the KD run: wd 0.5, K=64, one GNN block.
# --wandb ON (inherited from _defaults.sh).
ARCH=deep-sets
SIZE=distillnet
INTERACTION=0
LOCAL_INTERACTION=0
DISTILL=0
EXTRA_FLAGS="--num-interaction-layers 1 --interaction-k 64"
SAVE_TAG=train_top_deepsets_distillnet_gnn1_k64_ce_scratch
