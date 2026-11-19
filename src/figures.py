"""Build the figures and the summary table from the saved result files.

Usage: python src/figures.py
"""
import glob
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import DS1, load, mask
from prepare_beats import CLASSES, FS, PRE, STEP

NAMES = {"N": "Normal (N)", "S": "Supraventricular (S)", "V": "Ventricular (V)", "F": "Fusion (F)"}
COLORS = ["#0072B2", "#E69F00", "#D55E00", "#009E73"]          # colour-blind safe
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

base = json.load(open("results/baseline_metrics.json"))
inter = [json.load(open(f)) for f in sorted(glob.glob("results/cnn_inter_seed*.json"))]
notmpl = json.load(open("results/cnn_inter_notemplate_seed0.json"))
shape_only = json.load(open("results/cnn_inter_norr_notemplate_seed0.json"))
intra = json.load(open("results/cnn_intra_seed0.json"))

# 1. what the classes look like
X, D, rr, y, rec = load()
tr = mask(rec, DS1)
t = (np.arange(X.shape[1]) * STEP - PRE) / FS * 1000
fig, axes = plt.subplots(1, 4, figsize=(12, 2.8), sharey=True)
for k, (c, ax) in enumerate(zip(CLASSES, axes)):
    beats = X[tr & (y == k)]
    m, s = beats.mean(0), beats.std(0)
    ax.fill_between(t, m - s, m + s, color=COLORS[k], alpha=0.25, lw=0)
    ax.plot(t, m, color=COLORS[k], lw=1.8)
    ax.set_title(f"{NAMES[c]}\nn = {len(beats):,}", fontsize=10)
    ax.set_xlabel("Time from R peak (ms)")
axes[0].set_ylabel("Amplitude (z-score)")
fig.suptitle("Mean beat and one standard deviation by class, training patients (DS1)", y=1.12)
fig.savefig("results/figures/beat_classes.png", dpi=160, bbox_inches="tight"); plt.close(fig)

# 2. confusion matrices on the held-out patients
def draw_cm(ax, cm, title):
    cm = np.asarray(cm, float)
    frac = cm / cm.sum(1, keepdims=True)
    ax.imshow(frac, cmap="Blues", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{frac[i, j]:.2f}\n({int(cm[i, j]):,})", ha="center", va="center", fontsize=8,
                    color="white" if frac[i, j] > 0.5 else "black")
    ax.set_xticks(range(4), CLASSES); ax.set_yticks(range(4), CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Annotated"); ax.set_title(title, fontsize=10)

fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
draw_cm(axes[0], base["gradient_boosting_rr_shape_template"]["confusion"], "Gradient boosting (RR + shape + template)")
draw_cm(axes[1], inter[0]["test"]["confusion"], "BeatNet, seed 0")
fig.suptitle("Test patients (DS2): share of each annotated class, with beat counts")
fig.savefig("results/figures/confusion_matrices.png", dpi=160, bbox_inches="tight"); plt.close(fig)

# 3. the same network under a patient-wise and a random beat split
fig, ax = plt.subplots(figsize=(7, 3.6))
w = 0.38
f_inter = np.array([[r["test"][c]["f1"] for c in CLASSES] for r in inter])
f_intra = [intra["test"][c]["f1"] for c in CLASSES]
xs = np.arange(4)
ax.bar(xs - w / 2, f_inter.mean(0), w, yerr=f_inter.std(0), capsize=3, color="#0072B2",
       label="Patient-wise split (unseen patients)")
ax.bar(xs + w / 2, f_intra, w, color="#BBBBBB", label="Random beat split (same patients in train and test)")
for x, v, e in zip(xs - w / 2, f_inter.mean(0), f_inter.std(0)): ax.text(x, v + e + 0.02, f"{v:.2f}", ha="center", fontsize=8)
for x, v in zip(xs + w / 2, f_intra): ax.text(x, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
ax.set_xticks(xs, CLASSES); ax.set_ylabel("F1 on test beats"); ax.set_ylim(0, 1.12)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=1, fontsize=8)
fig.savefig("results/figures/split_comparison.png", dpi=160, bbox_inches="tight"); plt.close(fig)

# summary table
def row(name, rs):
    rs = rs if isinstance(rs, list) else [rs]
    def cell(get):
        v = np.array([get(r) for r in rs])
        return f"{v.mean():.3f} ± {v.std():.3f}" if len(v) > 1 else f"{v[0]:.3f}"
    cols = [cell(lambda r: r["macro_f1"]), cell(lambda r: r["accuracy"])]
    for c in CLASSES:
        cols += [cell(lambda r, c=c: r[c]["sensitivity"]), cell(lambda r, c=c: r[c]["ppv"])]
    return f"| {name} | " + " | ".join(cols) + " |"

head = "| Model | Macro F1 | Accuracy | " + " | ".join(f"{c} Se | {c} PPV" for c in CLASSES) + " |"
lines = [head, "|" + " --- |" * (2 + 1 + 8),
         row("Logistic regression, RR ratios only", base["logreg_rr_only"]),
         row("Gradient boosting, RR + beat shape", base["gradient_boosting_rr_shape"]),
         row("Gradient boosting, RR + beat shape + patient template", base["gradient_boosting_rr_shape_template"]),
         row("BeatNet, beat shape only (1 seed)", shape_only["test"]),
         row("BeatNet, beat shape + RR (1 seed)", notmpl["test"]),
         row(f"BeatNet, beat shape + RR + patient template ({len(inter)} seeds, mean ± SD)", [r["test"] for r in inter]),
         "", "Random beat split, for contrast only (same patients in train and test):", "", head,
         "|" + " --- |" * (2 + 1 + 8), row("BeatNet, beat shape + RR + patient template (1 seed)", intra["test"])]
open("results/summary.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
