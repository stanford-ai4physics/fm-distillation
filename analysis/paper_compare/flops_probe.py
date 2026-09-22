"""Fill in the missing forward-pass FLOP / param counts needed by the energy
model (comparison #2). Analytic op-graph count via torch FlopCounterMode at
the top-tagging median particle count N=47 -- same convention as
analysis/benchmark_inference.py, just for the sizes that file didn't cover
(PET2 micro, DeepSets distillnet) plus a re-check of the ones it did.

FlopCounterMode only traces the op graph of one forward pass; it does no
training and needs no real data. Run it inside a GPU salloc anyway (never on
the login node) for env parity.

Writes analysis/paper_compare/flops_probe.json.

  python analysis/paper_compare/flops_probe.py
"""

import json
import os

import torch
from torch.utils.flop_counter import FlopCounterMode

from omnilearned.network import PET2, DeepSets
from omnilearned.utils import get_model_parameters, get_deepsets_parameters

N = 47                # top-tagging median particle multiplicity
NUM_FEAT = 4
NUM_CLASSES = 2
NUM_COND = 2          # top tagging: use_cond -> cond dim 2
OUT = os.path.join(os.path.dirname(__file__), "flops_probe.json")

# (key, arch, size)
TARGETS = [
    ("pet2_small",          "pet2",     "small"),
    ("pet2_micro",          "pet2",     "micro"),
    ("deepsets_small",      "deepsets", "small"),
    ("deepsets_distillnet", "deepsets", "distillnet"),
]


def build(arch, size, device):
    if arch == "pet2":
        p = get_model_parameters(size)
        # matches distill_train_top_*.sh: interaction on, local_interaction on
        # for micro/small students, num_coord=2, K=10.
        m = PET2(input_dim=NUM_FEAT, use_int=True, local_int=True,
                 num_classes=NUM_CLASSES, mode="classifier",
                 num_coord=2, K=10, **p)
    else:
        p = get_deepsets_parameters(size)
        m = DeepSets(input_dim=NUM_FEAT, num_classes=NUM_CLASSES,
                     mode="classifier", **p)
    return m.to(device).eval()


def make_input(arch, device):
    x = torch.randn(1, N, NUM_FEAT, device=device)
    x[:, :, 2] = x[:, :, 2].abs() + 0.1     # nonzero -> valid-particle mask
    y = torch.randint(0, NUM_CLASSES, (1,), device=device)
    cond = torch.randn(1, NUM_COND, device=device)
    return x, y, cond


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  N={N}")
    out = {}
    for key, arch, size in TARGETS:
        model = build(arch, size, device)
        n_params = sum(p.numel() for p in model.parameters())
        x, y, cond = make_input(arch, device)
        try:
            with FlopCounterMode(model, display=False) as fc:
                model(x, y, cond=cond)
            flops = fc.get_total_flops()
        except TypeError:
            with FlopCounterMode(model, display=False) as fc:
                model(x, y)
            flops = fc.get_total_flops()
        out[key] = {"arch": arch, "size": size, "params": n_params,
                    "fwd_flops_at_N47": int(flops), "macs_at_N47": int(flops // 2)}
        print(f"  {key:22s} params={n_params:>12,}  fwd_flops={flops:>16,}  (MACs={flops//2:,})")
        del model

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
