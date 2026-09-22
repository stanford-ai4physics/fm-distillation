"""Independent cross-check of the FLOPs (Table 2) and MACs (added to Table 2,
Cost accounting, FPGA Readiness) numbers in
docs/paper_draft/ml4ps2026_deepsets_fpga.tex, both originally from
analysis/benchmark_inference.py / analysis/paper_compare/flops_probe.py's
torch.utils.flop_counter.FlopCounterMode trace (MACs = FLOPs/2, since that
tracer counts a matmul's multiply and add as two ops). This script
re-measures the same forward passes with the third-party `calflops`
package, which hooks nn.Module.forward directly (a different instrumentation
path than FlopCounterMode's __torch_dispatch__ trace) and reports FLOPs and
MACs as independent, natively-computed numbers rather than one derived from
the other.

calflops.__init__ unconditionally imports `transformers` for its HF helper,
which this env doesn't have and doesn't need (we only use
calflops.flops_counter.calculate_flops on plain nn.Module instances) -- a
minimal stub module satisfies the import instead of pulling in the real
dependency.

CPU only, seconds to run:
    /global/homes/t/twamorka/omnilearned-clean/env/bin/python \
        analysis/paper_compare/calflops_crosscheck.py
"""
import sys
import types
import importlib.machinery

_fake_transformers = types.ModuleType("transformers")
_fake_transformers.__spec__ = importlib.machinery.ModuleSpec("transformers", loader=None)
_fake_transformers.AutoTokenizer = object
sys.modules.setdefault("transformers", _fake_transformers)

import torch
torch.set_num_threads(2)
from calflops.flops_counter import calculate_flops

from omnilearned.network import PET2, DeepSets
from omnilearned.utils import get_model_parameters, get_deepsets_parameters

N = 47                # top-tagging median particle multiplicity (matches the paper)
NUM_FEAT = 4
NUM_CLASSES = 2
NUM_COND = 2

# (label, paper's FLOPs/2 MACs figure, builder)
def build_pet2(size):
    # Matches analysis/benchmark_inference.py's build_model() exactly for
    # "pet2" -- no K or num_coord override, so both take PET2's own
    # defaults (K=15, num_coord=3). That script produced the teacher FLOPs
    # numbers quoted in the paper; flops_probe.py's K=10/num_coord=2
    # override is for a *different* set of small/micro PET2 students and
    # does not apply to Teacher-L/Teacher-S here. Verified this matters:
    # K=10 for the large teacher understates the local-physics block and
    # gives 59.1G FLOPs instead of the correct 75.9G.
    p = get_model_parameters(size)
    return PET2(input_dim=NUM_FEAT, use_int=True, local_int=True,
                num_classes=NUM_CLASSES, mode="classifier", **p).eval()


def build_deepsets(size, gnn=False):
    p = get_deepsets_parameters(size)
    kw = dict(num_interaction_layers=1, interaction_k=64) if gnn else {}
    return DeepSets(input_dim=NUM_FEAT, num_classes=NUM_CLASSES,
                     mode="classifier", **p, **kw).eval()


def make_pet2_input():
    x = torch.randn(1, N, NUM_FEAT)
    x[:, :, 2] = x[:, :, 2].abs() + 0.1
    y = torch.zeros(1, dtype=torch.long)
    cond = torch.randn(1, NUM_COND)
    return x, y, cond


def make_deepsets_input():
    x = torch.randn(1, N, NUM_FEAT)
    x[:, :, 2] = x[:, :, 2].abs() + 0.1
    y = torch.zeros(1, dtype=torch.long)
    return x, y


# (label, paper FLOPs (Table 2 / benchmark_inference.py), paper MACs (FLOPs/2), builder)
TARGETS = [
    ("teacher-L (pet2 large)", 75.9e9, 37.94e9, lambda: (build_pet2("large"), *make_pet2_input())),
    ("teacher-S (pet2 small)", 550.8e6, 275.4e6, lambda: (build_pet2("small"), *make_pet2_input())),
    ("deepsets small", 15.8e6, 7.9e6, lambda: (build_deepsets("small"), *make_deepsets_input())),
    ("distillnet", 0.61e6, 0.305e6, lambda: (build_deepsets("distillnet"), *make_deepsets_input())),
    ("distillnet+GNN", 28.6e6, 14.3e6, lambda: (build_deepsets("distillnet", gnn=True), *make_deepsets_input())),
]


def main():
    print(f"{'model':24s} {'calflops FLOPs':>15s} {'calflops MACs':>15s}")
    for label, paper_flops, paper_macs, factory in TARGETS:
        model, x, y, *rest = factory()
        cond = rest[0] if rest else None
        kwargs = {"cond": cond} if cond is not None else {}
        flops, macs, params = calculate_flops(
            model=model,
            args=[x, y],
            kwargs=kwargs,
            print_results=False,
            print_detailed=False,
            output_as_string=False,
        )
        print(f"{label:24s} {flops:15,d} {macs:15,d}")
        del model


if __name__ == "__main__":
    main()
