export RANK=$SLURM_PROCID
export WORLD_RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID
export WORLD_SIZE=$SLURM_NTASKS
# Rendezvous host = first node in the allocation. Without this every task falls
# into the single-process branch of ddp_setup and collides on the default port.
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# Vary port per job so concurrent/leftover jobs don't collide (EADDRINUSE).
export MASTER_PORT=$(( 29500 + SLURM_JOB_ID % 10000 ))
export PYTHONUNBUFFERED=1
