# Large PET2 model trained from scratch directly on top tagging: no JetClass
# pretraining, no --fine-tune, no distillation. Comparison arm for whether the
# JetClass-pretrained-then-fine-tuned large teacher (fine_tune_top_l, used by
# save_teacher_logits_top.sh) actually beats training the same architecture
# straight on top tagging.
#
# Hyperparams are the author's own "from scratch, large, top" recipe (see the
# commented-out cmd in scripts/train.sh), matched to fine_tune_top_l's
# --interaction --local-interaction so it's evaluable with the same
# save_teacher_logits_top.sh (just override TAG=top_l_scratch).
SIZE=large
BATCH=8
EPOCH=10
LR=5e-6
WD=10.0
DISTILL=0
SAVE_TAG=top_l_scratch
