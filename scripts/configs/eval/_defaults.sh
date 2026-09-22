# Defaults for scripts/run_eval.sh. run_eval.sh sources this first, then the
# named config under configs/eval/, so a config only lists what differs from
# here. Env vars in run_eval.sh's override whitelist (SAVE_TAG DATASET_TYPE
# OUTDIR) still win over both.
#
# These defaults reproduce the old evaluate_top_distill.sh (small PET2 student,
# top tagging, interaction on, test split).

CHECKPOINT_DIR=/pscratch/sd/t/twamorka/omnilearned/checkpoints/
DATA_PATH=/global/cfs/cdirs/m4567/www/
EVAL_ROOT=/pscratch/sd/t/twamorka/omnilearned/eval

DATASET=top
NUM_CLASSES=2
DATASET_TYPE=test

ARCH=                       # empty -> PET2 default; else: deep-sets | mlp
SIZE=small
INTERACTION=1
LOCAL_INTERACTION=1
NUM_FEAT=                   # empty -> omnilearned default; jetclass needs 9

BATCH=128
NUM_WORKERS=4

EXTRA_FLAGS=                # appended verbatim to the CLI, e.g. "--use-pid"

SAVE_TAG=                   # required; every config must set it
OUTDIR=                     # empty -> $EVAL_ROOT/<config name>/
