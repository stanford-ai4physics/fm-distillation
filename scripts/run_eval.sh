#!/bin/bash
# Config-driven launcher for `omnilearned evaluate`.
#
#   scripts/run_eval.sh <config> [--dry-run]
#   SAVE_TAG=... scripts/run_eval.sh top_distill_micro      # env override
#   DATASET_TYPE=val scripts/run_eval.sh top_distill        # pick a split
#
# <config> names a file under scripts/configs/eval/ (without .sh).
# run_eval.sh sources configs/eval/_defaults.sh, then that config, assembles
# the CLI, and (unless --dry-run) runs `omnilearned evaluate` under srun across
# the current SLURM allocation. Per-rank logits land in
# $EVAL_ROOT/<config>/outputs_<save-tag>_<dataset>_<split>_rank*.npz.
#
# Env vars a caller may override per run: SAVE_TAG DATASET_TYPE OUTDIR.
#
# Not migrated (structurally different, still their own scripts):
#   evaluate_{chunk,train,top_large,top_for_distill}.sh  -- sharded/chunked
#     teacher-logit dumps
#   evaluate_metrics_top_{,T_}sweep.sh                   -- multi-tag metrics
#     tables with inline scoring
#   evaluate_top_distill_deepsets.sh + run_eval_deepsets.sh  -- DeepSets family
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONFIG="${1:?usage: run_eval.sh <config> [--dry-run]}"
DRY_RUN="${2:-}"

OVERRIDES="SAVE_TAG DATASET_TYPE OUTDIR"
for _v in $OVERRIDES; do eval "_ovr_${_v}=\${${_v}-}"; done

# shellcheck source=configs/eval/_defaults.sh
source "$SCRIPT_DIR/configs/eval/_defaults.sh"

CFG="$SCRIPT_DIR/configs/eval/${CONFIG}.sh"
[ -f "$CFG" ] || { echo "run_eval.sh: no such config: $CFG" >&2; exit 2; }
# shellcheck source=/dev/null
source "$CFG"

for _v in $OVERRIDES; do
    eval "_o=\${_ovr_${_v}}"
    [ -n "$_o" ] && eval "${_v}=\$_o"
done

: "${SAVE_TAG:?config must set SAVE_TAG}"
[ -n "$OUTDIR" ] || OUTDIR="$EVAL_ROOT/$CONFIG/"

args=(
    -i "$CHECKPOINT_DIR"
    -o "$OUTDIR"
    --save-tag "$SAVE_TAG"
    --dataset "$DATASET"
    --path "$DATA_PATH"
)
[ -n "$ARCH" ]               && args+=(--arch "$ARCH")
args+=(--size "$SIZE")
[ -n "$NUM_FEAT" ]           && args+=(--num-feat "$NUM_FEAT")
[ "$INTERACTION" = 1 ]       && args+=(--interaction)
[ "$LOCAL_INTERACTION" = 1 ] && args+=(--local-interaction)
args+=(
    --num-classes "$NUM_CLASSES"
    --batch "$BATCH"
    --num-workers "$NUM_WORKERS"
    --dataset-type "$DATASET_TYPE"
)
# shellcheck disable=SC2206
[ -n "$EXTRA_FLAGS" ] && args+=($EXTRA_FLAGS)

echo "run_eval.sh: config=$CONFIG  save_tag=$SAVE_TAG  out=$OUTDIR"
echo "+ omnilearned evaluate ${args[*]}"

[ "$DRY_RUN" = "--dry-run" ] && exit 0

mkdir -p "$OUTDIR"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"
run_ddp omnilearned evaluate "${args[@]}"
