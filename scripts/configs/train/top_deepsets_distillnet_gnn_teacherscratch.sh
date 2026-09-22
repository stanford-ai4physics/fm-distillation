# Same recipe as top_deepsets_distillnet_gnn.sh (DeepSets/PFN "distillnet"-width
# student, ~17k params, +1 GNN message-passing block) but pointed at the
# from-scratch large teacher (top_l_scratch, 93.30% acc / 0.9819 AUC) instead
# of the JetClass-pretrained-then-fine-tuned one (fine_tune_top_l, 94.44%/0.9880).
# Comparison arm: does the FPGA student do better/worse distilling from a
# weaker-but-not-JetClass-biased teacher vs the stronger pretrained one?
# Requires teacher logits already built via:
#   TAG=top_l_scratch bash scripts/save_teacher_logits_top.sh
#
# Baseline to beat (same student, pretrained teacher):
#   distill_top_deepsets_distillnet_scratch_a05_T4_gnn1_k64 = 93.98%.
ARCH=deep-sets
SIZE=distillnet
INTERACTION=0
LOCAL_INTERACTION=0
EXTRA_FLAGS="--num-interaction-layers 1 --interaction-k 64"
TEACHER_TAG=top_l_scratch
SAVE_TAG=distill_top_deepsets_distillnet_scratch_a05_T4_gnn1_k64_teacherscratch
