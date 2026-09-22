#!/bin/bash
# Inner job for the arXiv:2512.17011 comparison (#1-#4). Runs single-process
# (no srun / no MASTER_ADDR -- these analysis scripts call ddp_setup() in
# single-rank mode; see fpga-deepsets memory). Meant to be launched by
# scripts/paper_compare_loop.sh inside a 1-node GPU salloc.
#
# Produces, under $OUTDIR:
#   flops_probe.json                    (#2 input)
#   ptq_inputs_only_{small,distillnet}.log , ptq_both_{...}.log   (#3 input)
#   nconst_results.tsv                  (#4 input)
#   params_vs_rejection.{png,pdf}       (#1)
#   energy_model.{txt,png,pdf}          (#2)
#   tables_3_4.txt                      (#3 + #4)
set -uo pipefail

module load conda
conda activate /global/homes/t/twamorka/omnilearned-clean/env
module load pytorch

PY=/global/homes/t/twamorka/omnilearned-clean/env/bin/python
REPO=/global/cfs/cdirs/m3246/twamorka/omnilearned_test/OmniLearned
PC=$REPO/analysis/paper_compare
QZ=$REPO/tools/quantize
OUTDIR=/pscratch/sd/t/twamorka/omnilearned/results/paper_compare
mkdir -p "$OUTDIR"

export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONPATH="$PC:$QZ:${PYTHONPATH:-}"

DS_SMALL_TAG=distill_top_deepsets_small_scratch_a05_T4_archfix0804
DS_DNET_TAG=distill_top_deepsets_distillnet_scratch_a05_T4

echo "[$(date '+%F %T')] === paper_compare job start ==="

# Fresh start each session so a resubmit doesn't double-append.
rm -f "$OUTDIR/nconst_results.tsv" \
      "$OUTDIR"/ptq_inputs_only_*.log "$OUTDIR"/ptq_both_*.log

# ---- #2: fill missing FLOP / param counts -------------------------------
echo "[$(date '+%F %T')] flops_probe"
$PY "$PC/flops_probe.py" 2>&1 | tee "$OUTDIR/flops_probe.log"

# ---- #3: quantization table cells (DeepSets students) ------------------
for pair in "small:$DS_SMALL_TAG" "distillnet:$DS_DNET_TAG"; do
    size=${pair%%:*}; tag=${pair##*:}
    echo "[$(date '+%F %T')] PTQ inputs-only  $size  ($tag)"
    $PY "$QZ/ptq_deepsets.py" --tag "$tag" --size "$size" --bits 8 --weights-float \
        2>&1 | tee "$OUTDIR/ptq_inputs_only_${size}.log"
    echo "[$(date '+%F %T')] PTQ inputs+weights  $size"
    $PY "$QZ/ptq_deepsets.py" --tag "$tag" --size "$size" --bits 8 \
        2>&1 | tee "$OUTDIR/ptq_both_${size}.log"
done

# ---- #4: constituent-count robustness (full-N vs N=30) -----------------
#   tag                                          arch     size
NCONST_TARGETS=(
  "distill_top_small_scratch_a00_b10_T4          pet2     small"
  "distill_top_small_scratch_a05_T4              pet2     small"
  "distill_top_micro_scratch_a05_T4              pet2     micro"
  "${DS_SMALL_TAG}                               deepsets small"
  "${DS_DNET_TAG}                                deepsets distillnet"
)
for row in "${NCONST_TARGETS[@]}"; do
    set -- $row; tag=$1; arch=$2; size=$3
    for N in 0 30; do
        echo "[$(date '+%F %T')] nconst  $tag  N=$N"
        $PY "$PC/eval_nconst_top.py" --tag "$tag" --arch "$arch" --size "$size" \
            --nconst "$N" --outdir "$OUTDIR" 2>&1 | tail -n 12
    done
done

# ---- #1 + #2 + #3 + #4 : figures and tables (CPU) ---------------------
cd "$PC"
echo "[$(date '+%F %T')] fig_params_vs_rejection"
$PY fig_params_vs_rejection.py --outdir "$OUTDIR" 2>&1 | tee "$OUTDIR/fig1.log"
echo "[$(date '+%F %T')] energy_model"
$PY energy_model.py --outdir "$OUTDIR" 2>&1 | tee "$OUTDIR/energy.log"
echo "[$(date '+%F %T')] make_tables"
$PY make_tables.py --outdir "$OUTDIR" 2>&1 | tee "$OUTDIR/tables.log"

echo "[$(date '+%F %T')] === paper_compare job done ==="
echo "outputs in $OUTDIR"
ls -la "$OUTDIR"
