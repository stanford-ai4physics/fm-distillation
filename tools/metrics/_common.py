"""Shared helpers for compute_metrics_{top,jetclass}.py.

Only the parts that are byte-identical between the two: the rejection-at-
efficiency reader and the rank/concat file loaders. The per-task reporting
(binary vs. 10-class OvR) stays in each script.
"""

import glob
import os

import numpy as np

SIGNAL_EFFS = [0.50, 0.30]


def rej_at_eff(fpr, tpr, eff):
    """Return 1/FPR at the first point where TPR >= eff."""
    idx = np.searchsorted(tpr, eff)
    if idx >= len(fpr) or fpr[idx] == 0:
        return float("inf")
    return 1.0 / fpr[idx]


def load_rank_files(indir, tag, dataset):
    """Concatenate prediction/pid arrays from every per-rank npz for one tag.

    dataset picks the glob infix: outputs_<tag>_<dataset>_test_rank*.npz.
    """
    pattern = os.path.join(indir, f"outputs_{tag}_{dataset}_test_rank*.npz")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matching: {pattern}")
    print(f"Found {len(paths)} rank files")
    preds, labels = [], []
    for p in paths:
        z = np.load(p)
        preds.append(z["prediction"].astype(np.float32))
        labels.append(z["pid"])
    return np.concatenate(preds), np.concatenate(labels)


def load_concat_file(path):
    """Load prediction/pid from a single pre-concatenated npz (concat_logits.py)."""
    z = np.load(path)
    return z["prediction"].astype(np.float32), z["pid"]
