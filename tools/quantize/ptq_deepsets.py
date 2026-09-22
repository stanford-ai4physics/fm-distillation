"""Post-training quantization (PTQ) probe for the DeepSets top-tagging student.

Simulates fixed-point PTQ (symmetric per-tensor, static calibrated activation
ranges) at each nn.Linear boundary inside the DeepSets body/head -- weight
and input-activation quantization, everything else (GELU, DynamicTanh norms,
masked pooling) left in float. This is a software-level proxy for what
hls4ml/Brevitas would enforce on real fixed-point hardware, meant to answer
"how much does PTQ alone cost vs. the float checkpoint" before investing in
QAT (retraining with quantization noise injected).

No retraining happens here -- this loads an existing checkpoint, calibrates
activation ranges on the val split, then evaluates the test split at each
requested bit width.

Usage:
    python ptq_deepsets.py --tag distill_top_deepsets_distillnet_scratch_a05_T4 \
        --size distillnet --bits 8,6,4

    # student with the message-passing block: pass the SAME interaction flags
    # the float run used so the checkpoint restores into the same graph
    python ptq_deepsets.py --tag distill_top_deepsets_distillnet_scratch_a05_T4_gnn1_k64 \
        --size distillnet --num-interaction-layers 1 --interaction-k 64 --bits 8,6,4
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm.auto import tqdm

from omnilearned.dataloader import load_data
from omnilearned.network import DeepSets
from omnilearned.utils import (
    ddp_setup,
    get_checkpoint_name,
    get_deepsets_parameters,
    restore_checkpoint,
)

CHECKPOINT_DIR = "/pscratch/sd/t/twamorka/omnilearned/checkpoints/"
DATA_PATH = "/global/cfs/cdirs/m4567/www/"
SIGNAL_EFFS = [0.50, 0.30]
_QUANTILE_MAX = 2**24  # torch.quantile's input-size limit


class PTQLinear(nn.Module):
    """Wraps an nn.Linear with static-range fake quantization on its input
    activation and a per-tensor symmetric fake quantization on its weight.
    Bias stays float (standard PTQ practice -- bias correction matters much
    less than weight/activation precision, and hls4ml keeps bias at higher
    precision by default too)."""

    def __init__(self, linear, percentile=0.999):
        super().__init__()
        self.linear = linear
        self.calibrating = False
        self.percentile = percentile
        self._low_samples = []
        self._high_samples = []
        self.act_min = None
        self.act_max = None
        self.bits = None  # None => float passthrough
        self.quant_weights = True  # False => quantize inputs only, weights stay float

    def forward(self, x):
        if self.calibrating:
            flat = x.detach().reshape(-1)
            # torch.quantile refuses inputs above 2**24 elements; the all-pairs
            # input of the message-passing block (B x K x K x 67) exceeds that,
            # so estimate the percentiles on a uniform random subsample.
            if flat.numel() > _QUANTILE_MAX:
                idx = torch.randint(flat.numel(), (_QUANTILE_MAX,), device=flat.device)
                flat = flat[idx]
            # Per-batch percentile clipping instead of raw min/max: a single
            # extreme-pT jet in the calibration set otherwise blows up the
            # scale and crushes the useful dynamic range into a handful of
            # quantization levels near zero (empirically what happened with
            # raw min/max here -- 8-bit PTQ dropped to 77.79% acc / 0.8841
            # AUC vs the 92.85%/0.9800 float baseline).
            q = torch.quantile(
                flat.float(),
                torch.tensor(
                    [1.0 - self.percentile, self.percentile], device=flat.device
                ),
            )
            self._low_samples.append(q[0].item())
            self._high_samples.append(q[1].item())
            return self.linear(x)

        if self.bits is None:
            return self.linear(x)

        xq = _fake_quant_static(x, self.act_min, self.act_max, self.bits)
        wq = _fake_quant_weight(self.linear.weight, self.bits) if self.quant_weights else self.linear.weight
        return F.linear(xq, wq, self.linear.bias)

    def finalize_calibration(self):
        """Average the per-batch percentile estimates into a single static
        range (more robust than min/max-of-quantiles against one noisy
        batch)."""
        self.act_min = sum(self._low_samples) / len(self._low_samples)
        self.act_max = sum(self._high_samples) / len(self._high_samples)


def _fake_quant_static(x, xmin, xmax, bits):
    qmax = 2 ** (bits - 1) - 1
    scale = max(abs(xmin), abs(xmax), 1e-8) / qmax
    return torch.clamp(torch.round(x / scale), -qmax - 1, qmax) * scale


def _fake_quant_weight(w, bits):
    qmax = 2 ** (bits - 1) - 1
    scale = max(w.abs().max().item(), 1e-8) / qmax
    return torch.clamp(torch.round(w / scale), -qmax - 1, qmax) * scale


def wrap_linears(model, percentile=0.999):
    """Replace every nn.Linear in the model with a PTQLinear wrapper in place.

    Collects (parent, child_name) pairs before mutating -- setattr'ing a
    module's children while still inside model.named_modules()'s recursive
    traversal of that same module is unsafe."""
    targets = []
    for module in model.modules():
        for child_name, child in module.named_children():
            if isinstance(child, nn.Linear):
                targets.append((module, child_name, child))
    wrappers = []
    for module, child_name, child in targets:
        wrapper = PTQLinear(child, percentile=percentile)
        setattr(module, child_name, wrapper)
        wrappers.append(wrapper)
    return wrappers


def run_epoch(model, loader, device, desc):
    preds, labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, total=len(loader)):
            X = batch["X"].to(device, dtype=torch.float)
            y = batch["y"].to(device)
            cond = batch["cond"].to(device) if batch.get("cond") is not None else None
            out = model(X, y, cond=cond)
            preds.append(out["y_pred"].float().softmax(-1).cpu())
            labels.append(y.cpu())
    return torch.cat(preds).numpy(), torch.cat(labels).numpy()


def compute_metrics(preds, labels):
    scores = preds[:, 1]
    pred_classes = preds.argmax(axis=1)
    acc = (pred_classes == labels).mean()
    auc = roc_auc_score(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)
    rej = {}
    for eff in SIGNAL_EFFS:
        idx = tpr.searchsorted(eff)
        rej[eff] = float("inf") if idx >= len(fpr) or fpr[idx] == 0 else 1.0 / fpr[idx]
    return acc, auc, rej


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--size", required=True, help="e.g. small, distillnet, tiny, micro, nano")
    ap.add_argument("--bits", default="8,6,4", help="comma-separated bit widths to test")
    ap.add_argument("--calib-batches", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument(
        "--percentile", type=float, default=0.999,
        help="calibration clip percentile (e.g. 0.999 keeps the 0.1%%-99.9%% range)",
    )
    ap.add_argument("--num-interaction-layers", type=int, default=0,
                    help="message-passing blocks; must match the float run (0 = plain DeepSets)")
    ap.add_argument("--interaction-k", type=int, default=0,
                    help="leading-pT constituents kept by the message-passing block")
    ap.add_argument(
        "--weights-float", action="store_true",
        help="quantize only the linear-layer INPUTS to the target bit width; "
             "leave weights in float. Matches the 'i8' (inputs-only) column of "
             "Petitjean et al. arXiv:2512.17011 Table 6.",
    )
    args = ap.parse_args()

    local_rank, rank, size = ddp_setup()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    ds_params = get_deepsets_parameters(args.size)
    model = DeepSets(
        input_dim=4, num_classes=2, mode="classifier",
        num_interaction_layers=args.num_interaction_layers,
        interaction_k=args.interaction_k,
        **ds_params,
    )
    restore_checkpoint(
        model, CHECKPOINT_DIR, get_checkpoint_name(args.tag), local_rank, is_main_node=True
    )
    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded {args.tag} ({args.size}): {n_params:,} params")

    wrappers = wrap_linears(model, percentile=args.percentile)
    for w in wrappers:
        w.quant_weights = not args.weights_float
    mode = "inputs-only (weights float)" if args.weights_float else "inputs + weights"
    print(f"Wrapped {len(wrappers)} nn.Linear layers for PTQ (calib percentile={args.percentile}, quant: {mode})")

    calib_loader = load_data(
        "top", dataset_type="val", use_cond=True, path=DATA_PATH, batch=args.batch,
        num_workers=args.num_workers, rank=0, size=1, mode="classifier", shuffle=True,
    )
    for w in wrappers:
        w.calibrating = True
    with torch.no_grad():
        for i, batch in enumerate(calib_loader):
            if i >= args.calib_batches:
                break
            X = batch["X"].to(device, dtype=torch.float)
            y = batch["y"].to(device)
            cond = batch["cond"].to(device) if batch.get("cond") is not None else None
            model(X, y, cond=cond)
    for w in wrappers:
        w.calibrating = False
        w.finalize_calibration()
        print(f"  calibrated range: [{w.act_min:.4f}, {w.act_max:.4f}]")

    test_loader = load_data(
        "top", dataset_type="test", use_cond=True, path=DATA_PATH, batch=args.batch,
        num_workers=args.num_workers, rank=0, size=1, mode="classifier", shuffle=False,
    )

    bit_list = [int(b) for b in args.bits.split(",")]
    results = {}

    for w in wrappers:
        w.bits = None
    preds, labels = run_epoch(model, test_loader, device, "float32 (sanity check)")
    acc, auc, rej = compute_metrics(preds, labels)
    results["float32"] = (acc, auc, rej)
    print(f"float32        acc={acc:.4f} auc={auc:.4f} 1/FPR@50%={rej[0.5]:.1f} 1/FPR@30%={rej[0.3]:.1f}")

    for bits in bit_list:
        for w in wrappers:
            w.bits = bits
        preds, labels = run_epoch(model, test_loader, device, f"{bits}-bit PTQ")
        acc, auc, rej = compute_metrics(preds, labels)
        results[f"{bits}bit"] = (acc, auc, rej)
        print(f"{bits}-bit PTQ    acc={acc:.4f} auc={auc:.4f} 1/FPR@50%={rej[0.5]:.1f} 1/FPR@30%={rej[0.3]:.1f}")

    print("\n=== Summary ===")
    print(f"{'config':<12} {'acc':>8} {'auc':>8} {'1/FPR@50%':>10} {'1/FPR@30%':>10}")
    for k, (acc, auc, rej) in results.items():
        print(f"{k:<12} {acc*100:7.2f}% {auc:8.4f} {rej[0.5]:10.1f} {rej[0.3]:10.1f}")


if __name__ == "__main__":
    main()
