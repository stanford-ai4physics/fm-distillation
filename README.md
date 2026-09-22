# Towards foundation models on hardware accelerators for particle physics

Code accompanying the paper *"Towards foundation models on hardware
accelerators for particle physics"* (ML4PS 2026): distilling the
OmniLearned foundation model, fine-tuned on top tagging, into small
Deep Sets students, and quantizing those students to 8 bits.

This repo is **not standalone**. It contains only the files that are new
or modified relative to the base
[OmniLearned](https://github.com/ViniciusMikuni/OmniLearned) repository
that this work builds on. Everything else (dataset handling, the PET
transformer teacher architecture, the generic training/eval CLI) is
unchanged from the base repo — clone that repo first, then overlay the
files here on top of it at the matching paths.

## Setup

```bash
git clone https://github.com/ViniciusMikuni/OmniLearned.git
cp -r ml4ps-deepsets-fpga/{src,scripts,tools,analysis} OmniLearned/
```

Dependency setup (base `omnilearned` install, plus Brevitas for the
quantization scripts) is intentionally out of scope for this file list —
see the base repo's own install instructions.

All scripts below assume the layout above (i.e. you're running from the
`OmniLearned/` checkout with these files overlaid on top) and were run on
Perlmutter/NERSC: they `module load`, `conda activate`, and `salloc`/`srun`
against a specific account (`-A m3246`) and specific absolute scratch paths
(`/pscratch/sd/t/twamorka/...`, `/global/cfs/cdirs/m4567/www/`). None of
that has been genericized — treat the commands below as the exact recipe
that produced the paper's numbers, and edit the paths/account/env at the
top of each script (or export `OMNILEARNED_ENV`/`OMNILEARNED_REPO`/
`OMNILEARNED_SCRATCH` where a script already reads them) for your own
cluster.

## Training

Every training/eval run goes through `omnilearned train` / `omnilearned
evaluate` under the hood; the scripts here just assemble the CLI flags and
launch it via `srun` across an `salloc` allocation. Run these from the
`OmniLearned/` repo root.

### 1. Fine-tune the teachers

The teacher (OmniLearned-Large fine-tuned on top tagging, 423M params) and
the OmniLearned-Small reference row in Table 1 are trained with the
commented-out example commands in `scripts/train.sh` — uncomment the line
you need and run it inside a GPU allocation:

```bash
salloc -C gpu -q interactive -t 240 --nodes 1 --ntasks-per-node 4 \
       --gpus-per-node 4 -A <your-account> bash scripts/train.sh
```

- `--save-tag fine_tune_top_l` (large, pretrained then fine-tuned) — the
  teacher.
- `--save-tag fine_tune_top_s` (small, pretrained then fine-tuned) — the
  "Reference: OmniLearned-Small" row.

The from-scratch large teacher used by the `_teacherscratch` arm (Table
"pretrain") is trained through the config-driven launcher instead:

```bash
scripts/run_train.sh top_l_scratch          # --dry-run first to sanity-check the CLI
```

### 2. Cache the teacher's logits

Student training reads cached teacher logits rather than re-running the
teacher, via `scripts/save_teacher_logits_top.sh` (evaluates the teacher on
`train`+`val`, then merges the per-rank NPZ shards into companion H5 files):

```bash
TAG=fine_tune_top_l bash scripts/save_teacher_logits_top.sh   # pretrained teacher
TAG=top_l_scratch   bash scripts/save_teacher_logits_top.sh   # from-scratch teacher
```

Companion files land at `$SCRATCH/teacher_logits/companion_$TAG/`, which is
exactly where `run_train.sh`'s default `TEACHER_DIR` (`$TEACHER_ROOT/companion_$TEACHER_TAG`)
expects them.

### 3. Train the Deep Sets students

Each row of Table 1 / Table "pretrain" is one config under
`scripts/configs/train/`, launched the same way:

```bash
scripts/run_train.sh top_deepsets_distillnet                    # no GNN layer, KD          -> 92.85%
scripts/run_train.sh top_deepsets_distillnet_gnn                # +GNN layer, KD             -> 93.98%
scripts/run_train.sh top_deepsets_distillnet_gnn_ce              # +GNN layer, CE-only        -> 93.82%
scripts/run_train.sh top_deepsets_distillnet_gnn_teacherscratch  # +GNN layer, from-scratch teacher -> 93.65%
```

Each config only overrides what differs from `scripts/configs/train/_defaults.sh`
(dataset, KD alpha/beta/T=0.5/0.5/4, weight decay, teacher tag, etc.) — read
the comment block at the top of a config for the exact recipe and baseline
it's compared against. `run_train.sh` prints the full assembled
`omnilearned train ...` command before launching; pass `--dry-run` as a
second argument to see it without submitting anything. For a run that needs
to survive walltime limits, wrap the same call in
`scripts/lib/resubmit_loop.sh`:

```bash
LOOP_LOG_DIR=/path/to/logs/top_deepsets_distillnet_gnn \
    scripts/lib/resubmit_loop.sh bash scripts/run_train.sh top_deepsets_distillnet_gnn
```

### 4. Evaluate on the test split

The plain (no-GNN) student goes through its own pair of scripts:

```bash
scripts/run_eval_deepsets.sh distill_top_deepsets_distillnet_scratch_a05_T4 distillnet
```

The GNN-layer students go through the generic config-driven `run_eval.sh`,
one config per training config above:

```bash
scripts/run_eval.sh top_deepsets_distillnet_gnn
scripts/run_eval.sh top_deepsets_distillnet_gnn_ce
scripts/run_eval.sh top_deepsets_distillnet_gnn_teacherscratch
```

### 5. Score a checkpoint

Every eval run above writes per-rank NPZ prediction files; score them
(accuracy / AUC / background rejection at 30% and 50% signal efficiency)
with:

```bash
python tools/metrics/compute_metrics_top.py \
    --indir /pscratch/sd/t/twamorka/omnilearned/eval/top_distill_deepsets/ \
    --tag distill_top_deepsets_distillnet_scratch_a05_T4_gnn1_k64
```

Pass `--tag` multiple times (one per replicate run) to get a mean +/- std
summary instead of a single number.

### 6. Quantize (Table "quant")

QAT fine-tunes the float checkpoint above with 8-bit Brevitas layers; PTQ
quantizes it with no retraining. Both need the separate `omnilearned-fpga`
env with Brevitas installed.

```bash
# QAT: plain student, then GNN student
bash scripts/qat_train_deepsets_distillnet_8bit.sh
bash scripts/qat_train_deepsets_distillnet_gnn_8bit.sh
bash scripts/qat_deepsets_eval.sh          # score the plain QAT checkpoint
bash scripts/qat_deepsets_gnn_eval.sh      # score the GNN QAT checkpoint

# PTQ (no retraining): GNN student at 8/6/4 bits
bash scripts/ptq_deepsets_gnn_interactive.sh
```

The plain student's PTQ numbers come from the `ptq_deepsets.py` call inside
`scripts/paper_compare_job.sh` (that script also drives an unrelated
comparison table — only its PTQ block for `DS_DNET_TAG` is relevant here).

### 7. Regenerate Figure 1

```bash
python analysis/plot_roc_deepsets_paper.py
```

Recomputes all five ROC curves directly from the saved test-split NPZ
files above (no GPU needed) and prints each model's acc/AUC/rejection so
you can check them against the tables before trusting the figure. The
input globs and output path are hardcoded at the top of the script —
update them to your own scratch layout.

## What's in here

- `src/omnilearned/{network,layers,train,evaluate,dataloader,utils,cli}.py`
  — modified: adds the Deep Sets student, the optional message-passing
  (GNN) layer, and the knowledge-distillation loss/CLI flags.
- `export_ddp.sh` — modified: fixes the DDP rendezvous (`MASTER_ADDR`/
  `MASTER_PORT`) to use the actual allocation's first node and a
  per-job port, required by every multi-node run below.
- `scripts/train.sh` — the teacher fine-tune commands (`fine_tune_top_l`,
  `top_l` from-scratch, `fine_tune_top_s`) used to produce the teacher and
  reference rows in Table 1.
- `scripts/save_teacher_logits_top.sh`, `scripts/build_teacher_h5*`,
  `tools/preprocess/{build_teacher_h5,concat_logits}.py` — caching the
  teacher's logits so it is never re-invoked during student training.
- `scripts/configs/{train,eval}/top_deepsets_distillnet*.sh`,
  `top_l_scratch.sh` + `scripts/run_train.sh`, `run_eval.sh`,
  `evaluate_top_distill_deepsets.sh`, `run_eval_deepsets.sh`,
  `lib/{common,resubmit_loop}.sh` — the config-driven train/eval launchers
  for every Deep Sets student in Table 1 and Table "pretrain".
- `tools/quantize/{ptq_deepsets,qat_deepsets,qat_deepsets_eval}.py`,
  `scripts/{ptq_deepsets_gnn_interactive,qat_train_deepsets_distillnet*,
  qat_deepsets*eval}.sh`, `scripts/paper_compare_job.sh` — the 8-bit
  PTQ/QAT study in Table "quant".
- `tools/metrics/{compute_metrics_top,_common}.py` — Acc/AUC/rejection
  scoring shared by every table row.
- `analysis/plot_roc_deepsets_paper.py` — Figure 1 (ROC/rejection curves).
- `analysis/benchmark_inference.py`, `analysis/paper_compare/
  {flops_probe,calflops_crosscheck}.py` — the MAC-count numbers reported
  alongside Table "quant".

Perlmutter/NERSC-specific paths and SLURM assumptions in these scripts
(scratch paths, `module load`, `salloc -A ...`) reflect how the runs were
actually launched and have not been genericized for other clusters.
