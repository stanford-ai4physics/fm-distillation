"""Evaluate a QAT-trained DeepSets checkpoint (produced by qat_deepsets.py)
on the top-tagging test split.

The checkpoint's body/classifier_head state dicts contain Brevitas
QuantLinear submodules (weights + quantizer proxy buffers), not plain
nn.Linear -- so the model must be wrapped with the same wrap_linears_qat
call (same bit width) BEFORE restore_checkpoint, or the state_dict keys
won't match.

Must run with the omnilearned-fpga/env python (has Brevitas).

Usage:
    /global/homes/t/twamorka/omnilearned-fpga/env/bin/python qat_deepsets_eval.py \
        --tag qat_top_deepsets_distillnet_a05_T4_8bit --size distillnet --bits 8
"""

import argparse

import torch
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm.auto import tqdm

from omnilearned.dataloader import load_data
from omnilearned.network import DeepSets, ACT_LAYERS
from omnilearned.utils import (
    ddp_setup,
    get_checkpoint_name,
    get_deepsets_parameters,
    restore_checkpoint,
)
from qat_deepsets import wrap_linears_qat

CHECKPOINT_DIR = "/pscratch/sd/t/twamorka/omnilearned/checkpoints/"
DATA_PATH = "/global/cfs/cdirs/m4567/www/"
SIGNAL_EFFS = [0.50, 0.30]


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
    ap.add_argument("--tag", required=True, help="QAT checkpoint save-tag")
    ap.add_argument("--size", required=True)
    ap.add_argument("--bits", type=int, required=True, help="must match the bit width used in training")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--num-interaction-layers", type=int, default=0,
                    help="must match the QAT checkpoint's message-passing depth")
    ap.add_argument("--interaction-k", type=int, default=0,
                    help="must match the QAT checkpoint's leading-pT constituent cap")
    ap.add_argument("--act-layer", default="gelu", choices=sorted(ACT_LAYERS),
                    help="must match the QAT checkpoint's activation")
    ap.add_argument("--deepsets-fixed-n", type=int, default=0,
                    help="must match the QAT checkpoint's fixed-N/no-mask body (0 = masked-mean)")
    args = ap.parse_args()

    local_rank, rank, size = ddp_setup()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    ds_params = get_deepsets_parameters(args.size)
    model = DeepSets(
        input_dim=4, num_classes=2, mode="classifier",
        num_interaction_layers=args.num_interaction_layers,
        interaction_k=args.interaction_k,
        act_layer=ACT_LAYERS[args.act_layer],
        fixed_n=args.deepsets_fixed_n,
        **ds_params,
    )

    wrap_linears_qat(model, weight_bits=args.bits, act_bits=args.bits)
    n_qlin = sum(1 for m in model.modules() if type(m).__name__ == "QuantLinear")
    print(f"Wrapped {n_qlin} nn.Linear layers as QuantLinear ({args.bits}-bit) before loading checkpoint")

    restore_checkpoint(
        model, CHECKPOINT_DIR, get_checkpoint_name(args.tag), local_rank, is_main_node=True
    )
    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded {args.tag} ({args.size}, {args.bits}-bit QAT): {n_params:,} params")

    test_loader = load_data(
        "top", dataset_type="test", use_cond=True, path=DATA_PATH, batch=args.batch,
        num_workers=args.num_workers, rank=0, size=1, mode="classifier", shuffle=False,
    )

    preds, labels = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="QAT eval", total=len(test_loader)):
            X = batch["X"].to(device, dtype=torch.float)
            y = batch["y"].to(device)
            cond = batch["cond"].to(device) if batch.get("cond") is not None else None
            out = model(X, y, cond=cond)
            preds.append(out["y_pred"].float().softmax(-1).cpu())
            labels.append(y.cpu())
    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()

    acc, auc, rej = compute_metrics(preds, labels)
    print(f"\n=== {args.tag} ({args.bits}-bit QAT) ===")
    print(f"  N events  : {len(labels):,}")
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  AUC       : {auc:.4f}")
    for eff in SIGNAL_EFFS:
        print(f"  1/FPR @ {int(eff*100)}% sig eff : {rej[eff]:.1f}")


if __name__ == "__main__":
    main()
