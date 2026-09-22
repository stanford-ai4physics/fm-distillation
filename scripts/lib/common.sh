#!/bin/bash
# Shared setup for the config-driven drivers (run_train.sh, and later
# evaluate.sh / finetune.sh). Source this, then call run_ddp with the command
# to launch.

# Load the conda env + pytorch module and set the NCCL knobs every OmniLearned
# job on Perlmutter needs. Runs on the salloc head node before srun; srun
# propagates the resulting environment to every task.
_omnilearned_load_env() {
    module load conda
    # Env prefix is overridable so collaborators can point at their own conda
    # env (any readable absolute path works; another user's $HOME is not
    # readable on Perlmutter, so put a shared one under /global/cfs/cdirs/m3246).
    conda activate "${OMNILEARNED_ENV:-/global/homes/t/twamorka/omnilearned-clean/env}"
    module load pytorch

    export MASTER_ADDR="$(hostname)"
    export NCCL_TIMEOUT=600000
    export NCCL_DEBUG=WARN
    export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
}

# run_ddp <cmd> [args...]
#
# Launch one DDP job across the current SLURM allocation. Mirrors what every
# distill_train_top_*.sh did by hand: one srun task per GPU, each task
# re-deriving its rank from SLURM via export_ddp.sh (resolved against the
# repo-root CWD) before exec'ing the command.
run_ddp() {
    _omnilearned_load_env
    local quoted
    printf -v quoted '%q ' "$@"
    set -x
    srun -l -u bash -c "source export_ddp.sh && ${quoted}"
}
