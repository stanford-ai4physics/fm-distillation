"""
Real inference latency + analytic FLOP benchmark for the top-tagging teacher
and students (PET2 teacher/small, DeepSets KD variants, MLP KD), averaged
over a real sample of test-set jets spanning the natural particle-count (N)
distribution -- not a single event, since both FLOPs and latency scale with
N and a single jet is not representative.

No FPGA/Vivado access exists yet (see fpga-deepsets-distillation-progress
memory) -- this uses the A100 that's directly attached to this Perlmutter
login node (confirmed via nvidia-smi / torch.cuda.is_available(), no salloc
needed) as the GPU proxy, plus CPU timing as a second, more embedded-like
proxy (mirrors the CPU-vs-FPGA comparison style used in the DistillNet
reference paper, arXiv:2311.12551).

Methodology:
- Latency: measured per-event (batch=1, each jet trimmed to its own real,
  unpadded particle count) over N_EVENTS real test-set jets. GPU: 3 timed
  reps/event after a one-time model warmup; CPU: 1 rep/event (bounded for
  cost -- the 373M-param teacher is ~280ms/call on CPU). Reported as
  mean +/- std across events (population std, i.e. jet-to-jet variation +
  measurement noise combined, not repeated-measurement noise on one jet).
- FLOPs: analytic op-graph count via torch.utils.flop_counter.FlopCounterMode
  (built into PyTorch, no extra dependency), computed once per event at 5
  representative particle counts (min/p25/median/p75/max of the sampled
  N distribution) rather than for every event, since FLOPs are a
  deterministic function of N for a fixed architecture and computing it
  ~100 times added cost without new information. Reported as the value at
  median N, with the min/max-N range for context.

Shared login node etiquette: torch.set_num_threads(4) throughout, and rep
counts kept modest -- an earlier unbounded version pegged ~3000% CPU for
20+ minutes timing the large teacher with excessive reps and padded N.

Usage:
    python3 benchmark_inference.py
"""

import os
import time

import numpy as np
import torch

# Shared login node -- keep CPU thread usage bounded regardless of how many
# cores are visible (this ran unbounded once and pegged ~3000% CPU for 20+
# minutes; not acceptable on a shared node).
torch.set_num_threads(4)

from torch.utils.flop_counter import FlopCounterMode

from omnilearned.dataloader import load_data
from omnilearned.network import PET2, DeepSets, MLPStudent
from omnilearned.utils import (
    ddp_setup,
    get_checkpoint_name,
    get_deepsets_parameters,
    get_mlp_parameters,
    get_model_parameters,
    restore_checkpoint,
)

CHECKPOINT_DIR = "/pscratch/sd/t/twamorka/omnilearned/checkpoints/"
DATA_PATH = "/global/cfs/cdirs/m4567/www/"
OUT_FILE = "/pscratch/sd/t/twamorka/omnilearned/results/inference_benchmark.txt"

MODELS = [
    dict(
        name="Teacher-L (PET2 large)",
        arch="pet2",
        size="large",
        interaction=True,
        local_interaction=True,
        save_tag="fine_tune_top_l",
    ),
    dict(
        name="Teacher-S (PET2 small, CE)",
        arch="pet2",
        size="small",
        interaction=True,
        local_interaction=True,
        save_tag="fine_tune_top_s",
    ),
    dict(
        name="DeepSets-KD (a05)",
        arch="deep-sets",
        size="small",
        interaction=False,
        local_interaction=False,
        save_tag="distill_top_deepsets_small_scratch_a05_T4_archfix0804",
    ),
    # a00_b10 excluded: its checkpoint's state_dict keys (token,
    # local_physics.*, attn.in_proj_weight) are PET2, not DeepSets --
    # trained with the pre-arch-forwarding-fix cli.py despite the
    # --arch deep-sets flag (same bug class as the original mislabeled
    # a05 run). Needs retraining before it can go in this sweep. See
    # fpga-deepsets-distillation-progress memory.
    dict(
        name="DeepSets-KD (a05, wd0.05)",
        arch="deep-sets",
        size="small",
        interaction=False,
        local_interaction=False,
        save_tag="distill_top_deepsets_small_scratch_a05_wd005_T4",
    ),
    dict(
        name="MLP-KD (failed run)",
        arch="mlp",
        size="small",
        interaction=False,
        local_interaction=False,
        save_tag="distill_top_mlp_small_scratch_a05_T4",
    ),
]


def build_model(cfg, num_feat=4, num_classes=2):
    if cfg["arch"] == "pet2":
        params = get_model_parameters(cfg["size"])
        return PET2(
            input_dim=num_feat,
            use_int=cfg["interaction"],
            local_int=cfg["local_interaction"],
            mode="classifier",
            num_classes=num_classes,
            **params,
        )
    elif cfg["arch"] == "deep-sets":
        params = get_deepsets_parameters(cfg["size"])
        return DeepSets(
            input_dim=num_feat,
            num_classes=num_classes,
            mode="classifier",
            **params,
        )
    elif cfg["arch"] == "mlp":
        params = get_mlp_parameters(cfg["size"])
        return MLPStudent(
            input_dim=num_feat,
            num_classes=num_classes,
            mode="classifier",
            **params,
        )
    raise ValueError(cfg["arch"])


N_EVENTS = 100  # real test-set jets sampled for the benchmark
MIN_SLICE_N = 20  # PET2 default K=15 needs >= K+1=16 real+padded particles for topk


def get_real_batch(device, batch_size=16):
    loader = load_data(
        "top",
        DATA_PATH,
        batch=batch_size,
        dataset_type="test",
        distributed=False,
        use_cond=True,
        num_workers=0,
        rank=0,
        size=1,
        mode="classifier",
        shuffle=False,
    )
    batch = next(iter(loader))
    X = batch["X"].to(device, dtype=torch.float)
    y = batch["y"].to(device)
    kwargs = {
        k: (batch[k].to(device) if batch.get(k) is not None else None)
        for k in ["cond", "pid", "add_info"]
    }
    return X, y, kwargs


def slice_batch(X, y, kwargs, n):
    return (
        X[:n],
        y[:n],
        {k: (v[:n] if v is not None else None) for k, v in kwargs.items()},
    )


def single_trimmed_event(X, y, kwargs, idx):
    """One event, trimmed to its own real (unpadded) particle count -- a
    realistic single-jet inference input rather than the batch's max-padded
    N, which can be an outlier and wastes compute on zeros for every other
    model in the sweep."""
    n_real = int((X[idx, :, 2] != 0).sum().item())
    n_real = max(n_real, 1)
    # PET2's local-interaction kNN does topk(k=K+1); trimming an event to
    # fewer than K+1 real particles makes that topk crash ("index k out of
    # range"), which never happens in production because batches aren't
    # trimmed to a single event's own count. Floor the slice length at
    # MIN_SLICE_N, using the source tensor's own existing zero-padding
    # beyond n_real (same padding convention production batches already use).
    n_slice = min(max(n_real, MIN_SLICE_N), X.shape[1])
    Xe = X[idx : idx + 1, :n_slice]
    ye = y[idx : idx + 1]
    kw = {k: (v[idx : idx + 1] if v is not None else None) for k, v in kwargs.items()}
    return Xe, ye, kw, n_real


def representative_n_indices(n_reals):
    """Event indices at min/p25/median/p75/max of the sampled N (real
    particle count) distribution -- used for FLOP counting, which is a
    deterministic function of N and so doesn't need per-event recomputation."""
    order = np.argsort(n_reals)
    n = len(order)
    return {
        "min": int(order[0]),
        "p25": int(order[int(round(0.25 * (n - 1)))]),
        "median": int(order[int(round(0.5 * (n - 1)))]),
        "p75": int(order[int(round(0.75 * (n - 1)))]),
        "max": int(order[-1]),
    }


def time_forward(model, X, y, kwargs, device, n_warmup=10, n_reps=100):
    model.eval()
    on_cuda = device.type == "cuda"
    with torch.no_grad():
        for _ in range(n_warmup):
            model(X, y, **kwargs)
        if on_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_reps):
            model(X, y, **kwargs)
        if on_cuda:
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    return 1000.0 * dt / n_reps  # ms / inference


def count_flops(model, X, y, kwargs):
    model.eval()
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(X, y, **kwargs)
    return fc.get_total_flops()


def to_device(kwargs, device):
    return {k: (v.to(device) if v is not None else None) for k, v in kwargs.items()}


def main():
    ddp_setup()
    has_gpu = torch.cuda.is_available()
    gpu = torch.device("cuda:0" if has_gpu else "cpu")
    cpu = torch.device("cpu")

    fetch_device = gpu if has_gpu else cpu
    X, y, kwargs = get_real_batch(fetch_device, batch_size=N_EVENTS)
    X_cpu, y_cpu, kwargs_cpu = X.cpu(), y.cpu(), to_device(kwargs, cpu)
    n_events = X.shape[0]

    n_reals = np.array(
        [int((X_cpu[i, :, 2] != 0).sum().item()) for i in range(n_events)]
    )
    picks = representative_n_indices(n_reals)
    print(
        f"Benchmark sample: {n_events} real test-set jets. N particles: "
        f"min={n_reals.min()} p25={int(np.percentile(n_reals, 25))} "
        f"median={int(np.median(n_reals))} p75={int(np.percentile(n_reals, 75))} "
        f"max={n_reals.max()}\n",
        flush=True,
    )

    rows = []
    header = (
        f"{'Model':30s} {'Params':>10s} {'FLOPs@medianN':>14s} "
        f"{'GPU b=1 ms (mean+/-std)':>26s} {'CPU b=1 ms (mean+/-std)':>26s}"
    )
    print(header)

    for cfg in MODELS:
        ckpt_name = get_checkpoint_name(cfg["save_tag"])
        ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_name)
        if not os.path.isfile(ckpt_path):
            print(f"[skip] {cfg['name']}: no checkpoint at {ckpt_path}")
            continue

        print(f"[running] {cfg['name']}...", flush=True)
        model = build_model(cfg)
        restore_checkpoint(model, CHECKPOINT_DIR, ckpt_name, 0, is_main_node=True)
        n_params = sum(p.numel() for p in model.parameters())

        # FLOPs at 5 representative N (min/p25/median/p75/max) -- deterministic
        # given N, so no need to recompute per event.
        flops_by_pos = {}
        model_cpu = model.cpu()
        for pos, idx in picks.items():
            Xe, ye, kwe, nreal = single_trimmed_event(X_cpu, y_cpu, kwargs_cpu, idx)
            flops_by_pos[pos] = (nreal, count_flops(model_cpu, Xe, ye, kwe))
        flops_median = flops_by_pos["median"][1]
        print(
            f"  flops: min-N({flops_by_pos['min'][0]})={flops_by_pos['min'][1] / 1e6:.2f}M  "
            f"median-N({flops_by_pos['median'][0]})={flops_median / 1e6:.2f}M  "
            f"max-N({flops_by_pos['max'][0]})={flops_by_pos['max'][1] / 1e6:.2f}M",
            flush=True,
        )

        # CPU per-event latency, 1 rep/event (bounded for cost on the largest model)
        cpu_lats = []
        for i in range(n_events):
            Xe, ye, kwe, _ = single_trimmed_event(X_cpu, y_cpu, kwargs_cpu, i)
            cpu_lats.append(
                time_forward(model_cpu, Xe, ye, kwe, cpu, n_warmup=0, n_reps=1)
            )
            if (i + 1) % 25 == 0:
                print(f"  cpu latency: {i + 1}/{n_events} events", flush=True)
        cpu_lats = np.array(cpu_lats)

        # GPU per-event latency, 3 reps/event after one model-level warmup
        if has_gpu:
            model_gpu = model.to(gpu)
            med_idx = picks["median"]
            Xe, ye, kwe, _ = single_trimmed_event(X, y, kwargs, med_idx)
            model_gpu.eval()
            with torch.no_grad():
                for _ in range(5):
                    model_gpu(Xe, ye, **kwe)
            torch.cuda.synchronize()

            gpu_lats = []
            for i in range(n_events):
                Xe, ye, kwe, _ = single_trimmed_event(X, y, kwargs, i)
                gpu_lats.append(
                    time_forward(model_gpu, Xe, ye, kwe, gpu, n_warmup=0, n_reps=3)
                )
                if (i + 1) % 25 == 0:
                    print(f"  gpu latency: {i + 1}/{n_events} events", flush=True)
            gpu_lats = np.array(gpu_lats)
        else:
            gpu_lats = np.array([float("nan")])

        row = dict(
            name=cfg["name"],
            params=n_params,
            flops_median=flops_median,
            flops_min=flops_by_pos["min"][1],
            flops_max=flops_by_pos["max"][1],
            n_median=flops_by_pos["median"][0],
            gpu_mean=gpu_lats.mean(),
            gpu_std=gpu_lats.std(),
            cpu_mean=cpu_lats.mean(),
            cpu_std=cpu_lats.std(),
        )
        rows.append(row)
        print(
            f"{cfg['name']:30s} {n_params / 1e6:9.3f}M {flops_median / 1e6:13.2f}M "
            f"{row['gpu_mean']:9.3f}+/-{row['gpu_std']:<8.3f} "
            f"{row['cpu_mean']:9.3f}+/-{row['cpu_std']:<8.3f}",
            flush=True,
        )

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    with open(OUT_FILE, "w") as f:
        f.write(
            f"Benchmark sample: {n_events} real test-set jets. N particles: "
            f"min={n_reals.min()} p25={int(np.percentile(n_reals, 25))} "
            f"median={int(np.median(n_reals))} p75={int(np.percentile(n_reals, 75))} "
            f"max={n_reals.max()}\n"
        )
        f.write(f"GPU: {torch.cuda.get_device_name(0) if has_gpu else 'none'}\n")
        f.write(
            "Latency: per-event batch=1 (each jet trimmed to its own real N), "
            "GPU=3 reps/event, CPU=1 rep/event, mean+/-std over events.\n"
        )
        f.write(
            "FLOPs: analytic op-graph count (FlopCounterMode) at median-N event; "
            "min-N/max-N given for range context.\n\n"
        )
        f.write(header + "\n")
        for row in rows:
            f.write(
                f"{row['name']:30s} {row['params'] / 1e6:9.3f}M "
                f"{row['flops_median'] / 1e6:13.2f}M "
                f"{row['gpu_mean']:9.3f}+/-{row['gpu_std']:<8.3f} "
                f"{row['cpu_mean']:9.3f}+/-{row['cpu_std']:<8.3f}\n"
            )
            f.write(
                f"{'':30s} (FLOPs range: {row['flops_min'] / 1e6:.2f}M @minN "
                f"-- {row['flops_max'] / 1e6:.2f}M @maxN)\n"
            )
    print(f"\nWrote {OUT_FILE}")


if __name__ == "__main__":
    main()
