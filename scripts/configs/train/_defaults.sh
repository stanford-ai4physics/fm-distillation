# Defaults for scripts/run_train.sh. run_train.sh sources this first, then the
# named config under configs/train/, so a config only lists what differs from
# here. Env vars in run_train.sh's override whitelist (SAVE_TAG ALPHA BETA
# DISTILL_T) still win over both, for replicate / sweep loops.
#
# These defaults reproduce the old distill_train_top.sh (large fine-tuned top
# teacher -> PET2-small student, KD alpha=beta=0.5, T=4).

OUTDIR=/pscratch/sd/t/twamorka/omnilearned/checkpoints/
DATA_PATH=/global/cfs/cdirs/m4567/www/
TEACHER_ROOT=/pscratch/sd/t/twamorka/omnilearned/teacher_logits

DATASET=top
MODE=classifier
NUM_CLASSES=2

ARCH=                       # empty -> PET2 default; else: deep-sets | mlp
SIZE=small
NUM_FEAT=                    # empty -> omnilearned default (4); jetclass 9, dctr 5
INTERACTION=1
LOCAL_INTERACTION=1

BATCH=128
ITERATIONS=1000             # empty -> omit --iterations
EPOCH=50
LR=5e-4
LR_FACTOR=                  # empty -> omit --lr-factor (fine-tune uses it)
WARMUP_EPOCH=               # empty -> omit --warmup-epoch
WD=0.5
NUM_WORKERS=4

# Fine-tune from a pretrained/distilled checkpoint instead of training from
# scratch. FINETUNE=1 adds --fine-tune --pretrain-tag $PRETRAIN_TAG.
FINETUNE=0
PRETRAIN_TAG=

DISTILL=1
ALPHA=0.5
BETA=0.5
DISTILL_T=4
TEACHER_TAG=fine_tune_top_l
TEACHER_DIR=                # empty -> $TEACHER_ROOT/companion_$TEACHER_TAG

WANDB=1
EXTRA_FLAGS=

SAVE_TAG=                   # required; every config must set it
