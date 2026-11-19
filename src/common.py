"""Shared helpers: data splits and the metrics reported throughout."""
import numpy as np
from sklearn.metrics import confusion_matrix

from prepare_beats import CLASSES, DS1, DS2

# Five DS1 patients held out while choosing the architecture and epoch count, so DS2 is
# touched only once. They were picked to contain S and V beats without removing the
# few bundle branch block patients from training.
VAL = ["116", "201", "205", "215", "220"]
TRAIN = [r for r in DS1 if r not in VAL]


def load(path="data/beats.npz"):
    """Beats, deviation of each beat from its patient's median beat, RR ratios, labels, record ids."""
    d = np.load(path)
    return d["X"], d["X"] - d["template"], d["rr"], d["y"], d["record"]


def mask(record, ids):
    return np.isin(record, [int(i) for i in ids])


def report(y_true, y_pred):
    """Per-class sensitivity, positive predictive value and F1, plus macro F1."""
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))
    out = {"confusion": cm.tolist(), "accuracy": float(np.trace(cm) / cm.sum())}
    f1s = []
    for k, c in enumerate(CLASSES):
        tp = cm[k, k]
        se = tp / max(cm[k].sum(), 1)
        ppv = tp / max(cm[:, k].sum(), 1)
        f1 = 2 * se * ppv / max(se + ppv, 1e-12)
        out[c] = {"sensitivity": float(se), "ppv": float(ppv), "f1": float(f1)}
        f1s.append(f1)
    out["macro_f1"] = float(np.mean(f1s))
    return out
