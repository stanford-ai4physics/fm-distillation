"""ROC (background rejection vs. signal efficiency) figure for the ML4PS Deep Sets
KD paper, docs/paper_draft/ml4ps2026_deepsets_fpga.tex (fig:roc).

Curves are recomputed from saved test-split prediction files; no GPU needed.
Every model's (acc, AUC, 1/FPR@50%, 1/FPR@30%) is printed so it can be checked
against Table 2 / Table 3 of the draft before the figure is used.

Usage:  python analysis/plot_roc_deepsets_paper.py
Writes: docs/paper_draft/roc_deepsets.{pdf,png}
"""
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

SP = "/pscratch/sd/t/twamorka/omnilearned"
DS = f"{SP}/eval/top_distill_deepsets"
OUT = "/global/cfs/cdirs/m3246/twamorka/omnilearned_test/OmniLearned/docs/paper_draft/roc_deepsets"

# Five curves: the two OmniLearned models (purple / grey) and the three
# DistillNet+GNN variants, one colour each. Solid = distilled from the
# pretrained teacher, dashed = trained from scratch on hard labels,
# dotted = distilled from the from-scratch teacher.
BLUE, ORANGE, AQUA, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#7b3fb8"
INK, INK2 = "#0b0b0b", "#7a7975"

# label, glob, color, linestyle, linewidth
CURVES = [
    ("OmniLearned-Large, fine-tuned to top tagging (teacher)",
     f"{SP}/teacher_logits/TEST/outputs_fine_tune_top_l_top_test_rank*.npz", PURPLE, "-", 1.8),
    ("OmniLearned-Small, fine-tuned to top tagging",
     f"{SP}/teacher_logits/outputs_fine_tune_top_s_top_test_*.npz", INK2, "-", 1.2),
    ("DeepSets (including GNN); distilled from pretrained teacher",
     f"{DS}/outputs_distill_top_deepsets_distillnet_scratch_a05_T4_gnn1_k64_top_test_rank*.npz", AQUA, "-", 2.2),
    ("DeepSets (including GNN); trained from scratch",
     f"{DS}/outputs_train_top_deepsets_distillnet_gnn1_k64_ce_scratch_top_test_rank*.npz", BLUE, "--", 1.6),
    ("DeepSets (including GNN); distilled from a from-scratch teacher",
     f"{DS}/outputs_distill_top_deepsets_distillnet_scratch_a05_T4_gnn1_k64_teacherscratch_top_test_rank*.npz", ORANGE, ":", 1.8),
]
# A curve whose npz files do not exist is skipped with a warning.


def load(pattern):
    fs = sorted(glob.glob(pattern))
    assert fs, pattern
    key = "labels" if "labels" in np.load(fs[0]).files else "pid"
    logits = np.concatenate([np.load(f)["logits"].astype(np.float32) for f in fs])
    y = np.concatenate([np.load(f)[key] for f in fs])
    return logits, y


plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 10, "legend.fontsize": 8,
    "axes.edgecolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.labelcolor": INK, "text.color": INK,
})
# Single panel: linear-scale rejection over the 25-55% working region where
# the 30%/50% operating points live. The full log-scale ROC (0.1-1.0
# efficiency) added no information beyond this region for this figure, so
# it has been dropped rather than kept as a second panel or inset.
fig = plt.figure(figsize=(5.0, 3.6), layout="constrained")
ax = fig.subplots(1, 1)
ZOOM_X, ZOOM_Y = (0.25, 0.55), (0, 4500)

GRID = np.linspace(0.1, 0.99, 600)   # common signal-efficiency grid for printed ratios
REF_LABEL = "DeepSets (including GNN); trained from scratch"
curves = {}

for label, pattern, color, ls, lw in CURVES:
    if not glob.glob(pattern):
        print(f"SKIP (no files yet): {label}")
        continue
    logits, y = load(pattern)
    d = logits[:, 1] - logits[:, 0]
    s = 1.0 / (1.0 + np.exp(-d))
    fpr, tpr, _ = roc_curve(y, s)
    acc = ((d > 0) == y).mean() * 100
    auc = roc_auc_score(y, s)
    r50 = 1.0 / fpr[np.searchsorted(tpr, 0.5)]
    r30 = 1.0 / fpr[np.searchsorted(tpr, 0.3)]
    print(f"{label:56s} acc={acc:.2f} AUC={auc:.4f} 1/FPR@50={r50:.1f} 1/FPR@30={r30:.1f}")
    ok = (fpr > 0) & (tpr >= 0.1)
    zz = ok & (tpr >= ZOOM_X[0]) & (tpr <= ZOOM_X[1])
    ax.plot(tpr[zz], 1.0 / fpr[zz], color=color, ls=ls, lw=lw, label=label)
    # log-rejection on the common grid; tpr is non-decreasing, keep the
    # first point of each plateau so np.interp sees an increasing abscissa
    tpr_u, idx = np.unique(tpr[ok], return_index=True)
    logrej = np.interp(GRID, tpr_u, np.log10(1.0 / fpr[ok][idx]))
    curves[label] = logrej

ref = curves[REF_LABEL]
for label, logrej in curves.items():
    r = 10 ** (np.interp([0.3, 0.5], GRID, logrej) - np.interp([0.3, 0.5], GRID, ref))
    print(f"  ratio to scratch @30%={r[0]:.2f} @50%={r[1]:.2f}  {label}")

for e in (0.3, 0.5):
    ax.axvline(e, color=INK, lw=0.8, ls=(0, (2, 3)), zorder=1)
    ax.text(e, ZOOM_Y[1] * 0.97, f"{int(e*100)}%", ha="center", va="top", fontsize=7, color=INK)

ax.set_xlim(*ZOOM_X)
ax.set_ylim(*ZOOM_Y)
ax.set_xticks([0.3, 0.4, 0.5])
ax.set_yticks([0, 1000, 2000, 3000, 4000])
ax.set_yticklabels(["0", "1k", "2k", "3k", "4k"])
ax.set_xlabel(r"Signal efficiency $\epsilon_s$")
ax.set_ylabel(r"Background rejection $1/\epsilon_b$")
ax.grid(True, color="#e6e5e1", lw=0.6)

for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
# Legend sits entirely above the axes so it never covers a curve.
ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), borderaxespad=0.2,
          frameon=False, handlelength=2.4)

fig.savefig(OUT + ".pdf")
fig.savefig(OUT + ".png", dpi=200)
print("wrote", OUT + ".{pdf,png}")
