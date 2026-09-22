# DeepSets/PFN top-tagging student at "distillnet" width (~11k params, ~27x
# smaller than "small"), the size-scan point that came 2nd-best (92.85% vs
# 93.30% at small). One-variable change from top_deepsets_a05 (SIZE); KD
# a0.5/b0.5/T4, teacher fine_tune_top_l, wd 0.5.
#
# For other DeepSets widths, copy this file and set SIZE to one of
# nano | distillnet | micro | tiny | small | medium | large
# (see get_deepsets_parameters in src/omnilearned/utils.py), with a matching
# SAVE_TAG. This replaces the old SIZE_OVERRIDE env knob.
#
# --wandb ON (inherited from _defaults.sh).
# Was: distill_train_top_deepsets.sh CONFIG=distillnet.
ARCH=deep-sets
SIZE=distillnet
INTERACTION=0
LOCAL_INTERACTION=0
SAVE_TAG=distill_top_deepsets_distillnet_scratch_a05_T4
