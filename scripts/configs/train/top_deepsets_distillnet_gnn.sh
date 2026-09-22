# DeepSets/PFN "distillnet"-width top-tagging student (~11k params) PLUS the
# optional particle-particle message-passing layers -- one-variable change
# from top_deepsets_distillnet.sh (adds EXTRA_FLAGS). KD a0.5/b0.5/T4,
# teacher fine_tune_top_l, wd 0.5.
#
#   --num-interaction-layers 1  one EdgeConv/Interaction-Network GNN block
#                               (log mass / log dR / log kT edge features)
#                               inserted before the phi blocks
#   --interaction-k 64          keep the leading-pT 64 constituents so the
#                               O(N^2) message passing stays cheap (inputs
#                               are stored pT-descending -> plain slice)
#
# Baseline to beat: distill_top_deepsets_distillnet_scratch_a05_T4 = 92.85%.
#
# --wandb ON (inherited from _defaults.sh).
ARCH=deep-sets
SIZE=distillnet
INTERACTION=0
LOCAL_INTERACTION=0
EXTRA_FLAGS="--num-interaction-layers 1 --interaction-k 64"
SAVE_TAG=distill_top_deepsets_distillnet_scratch_a05_T4_gnn1_k64
