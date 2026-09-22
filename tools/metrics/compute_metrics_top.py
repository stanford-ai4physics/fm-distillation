"""Compute AUC, accuracy, and background rejection at fixed signal efficiencies for top tagging.

Binary classification: label 0 = QCD (background), label 1 = top (signal).

Metrics:
  - AUC
  - Accuracy
  - 1/FPR at signal efficiency (TPR) = 50% and 30%

Usage:
    python compute_metrics_top.py --indir /path/to/eval/dir --tag <save_tag>
    python compute_metrics_top.py --indir /path/to/eval/dir --tag <tag1> --tag <tag2> --tag <tag3>
        (multiple --tag also prints a mean +- std summary across reps)
"""

import argparse

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from _common import SIGNAL_EFFS, load_rank_files, rej_at_eff

SIGNAL_CLASS = 1


def compute_and_print(preds, labels, tag=""):
    scores = preds[:, SIGNAL_CLASS]
    pred_classes = preds.argmax(axis=1)

    acc = (pred_classes == labels).mean()
    auc = roc_auc_score(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)
    rejs = [rej_at_eff(fpr, tpr, eff) for eff in SIGNAL_EFFS]

    header = f"=== {tag} ===" if tag else "=== Results ==="
    print(f"\n{header}")
    print(f"  N events  : {len(labels):,}")
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  AUC       : {auc:.4f}")
    for eff, rej in zip(SIGNAL_EFFS, rejs):
        print(f"  1/FPR @ {int(eff*100)}% sig eff : {rej:.1f}")
    print()

    return {"acc": acc, "auc": auc, **{f"rej@{int(eff*100)}": rej for eff, rej in zip(SIGNAL_EFFS, rejs)}}


def print_spread(tags, results):
    keys = results[0].keys()
    print(f"=== Spread across {len(tags)} reps: {', '.join(tags)} ===")
    for key in keys:
        vals = np.array([r[key] for r in results])
        finite = np.isfinite(vals)
        if not finite.all():
            print(f"  {key:10s} : {vals.tolist()} (inf present, skipping mean/std)")
            continue
        print(f"  {key:10s} : mean={vals.mean():.4f}  std={vals.std(ddof=1):.4f}  values={vals.tolist()}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True, help="Directory with per-rank npz files")
    ap.add_argument(
        "--tag",
        required=True,
        action="append",
        help="save_tag used during evaluate; repeat --tag for multiple reps to get a spread summary",
    )
    args = ap.parse_args()

    results = []
    for tag in args.tag:
        preds, labels = load_rank_files(args.indir, tag, "top")
        results.append(compute_and_print(preds, labels, tag))

    if len(args.tag) > 1:
        print_spread(args.tag, results)


if __name__ == "__main__":
    main()
