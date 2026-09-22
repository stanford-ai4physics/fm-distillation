#!/bin/bash
# Generic salloc resubmit loop. Keeps grabbing a 4-node x 4-GPU interactive
# allocation and running <body...> inside it until the body exits 0 (all
# epochs done -> stop) or MAX_LOOPS is hit. A non-zero exit (walltime limit
# or preemption) triggers a resubmit; the body scripts run with --resuming so
# each session picks up where the last left off.
#
#   LOOP_LOG_DIR=/pscratch/.../logs/<name> [LOOP_TAG=<tag>] \
#   [NODES=4] [WALLTIME=240] [MAX_LOOPS=20] \
#     scripts/lib/resubmit_loop.sh bash scripts/run_train.sh <config>
#
# LOOP_TAG, when set, prefixes the per-session log files and summary.log so
# several replicate loops can share one LOOP_LOG_DIR (matches the old
# distill_loop_top_*_rep.sh "${SAVE_TAG}_" naming).
#
# Note: unlike the old per-experiment loops this does NOT set `-e`. Those
# aborted on the first non-zero salloc (with `pipefail` the `| tee` pipeline
# failed before the resubmit check), so resubmit-on-timeout never actually
# fired; the old distill_loop_top_deepsets.sh worked around it with `set +e`.
# Dropping `-e` here makes every loop behave like that fixed one.
set -uo pipefail

LOOP_LOG_DIR="${LOOP_LOG_DIR:?LOOP_LOG_DIR required}"
PREFIX="${LOOP_TAG:+${LOOP_TAG}_}"
NODES="${NODES:-4}"
WALLTIME="${WALLTIME:-240}"
MAX_LOOPS="${MAX_LOOPS:-20}"
SUMMARY="$LOOP_LOG_DIR/${PREFIX}summary.log"

mkdir -p "$LOOP_LOG_DIR"

LOOP=0
while [ "$LOOP" -lt "$MAX_LOOPS" ]; do
    LOOP=$((LOOP + 1))
    TS=$(date '+%Y-%m-%d_%H-%M-%S')
    echo "[${TS}] === ${PREFIX}session ${LOOP} / ${MAX_LOOPS} ===" | tee -a "$SUMMARY"

    salloc \
        -C gpu \
        -q interactive \
        -t "$WALLTIME" \
        --nodes "$NODES" \
        --ntasks-per-node 4 \
        --gpus-per-node 4 \
        -A m3246 \
        "$@" \
        2>&1 | tee "$LOOP_LOG_DIR/${PREFIX}session_${LOOP}_${TS}.out"
    EXIT="${PIPESTATUS[0]}"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${PREFIX}session ${LOOP} exited with code ${EXIT}" \
        | tee -a "$SUMMARY"

    if [ "$EXIT" -eq 0 ]; then
        echo "Training completed. Stopping loop." | tee -a "$SUMMARY"
        exit 0
    fi

    echo "Non-zero exit (likely time limit or preemption). Resubmitting in 15s..." \
        | tee -a "$SUMMARY"
    sleep 15
done

echo "Reached MAX_LOOPS=${MAX_LOOPS}. Edit MAX_LOOPS and rerun to continue." \
    | tee -a "$SUMMARY"
exit 1
