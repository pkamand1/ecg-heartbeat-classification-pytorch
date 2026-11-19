"""Classical baselines on the inter-patient split.

1. Logistic regression on the four RR-interval ratios alone (rhythm only).
2. Gradient-boosted trees on RR ratios plus a downsampled beat shape.
3. The same, plus each beat's deviation from its patient's median beat.

Usage: python src/baseline.py
"""
import json

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import DS1, DS2, load, mask, report

X, D, rr, y, rec = load()
tr, te = mask(rec, DS1), mask(rec, DS2)
shape = X[:, ::3]                                   # 39 points per beat
feats = np.c_[rr, shape]

results = {}
lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
lr.fit(rr[tr], y[tr])
results["logreg_rr_only"] = report(y[te], lr.predict(rr[te]))

for name, F in (("gradient_boosting_rr_shape", feats),
                ("gradient_boosting_rr_shape_template", np.c_[feats, D[:, ::3]])):
    gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, class_weight="balanced", random_state=0)
    gb.fit(F[tr], y[tr])
    results[name] = report(y[te], gb.predict(F[te]))

json.dump(results, open("results/baseline_metrics.json", "w"), indent=1)
for name, r in results.items():
    print(f"{name:38s} macro F1 {r['macro_f1']:.3f}  accuracy {r['accuracy']:.3f}  "
          + "  ".join(f"{c} Se {r[c]['sensitivity']:.2f} PPV {r[c]['ppv']:.2f}" for c in "NSVF"))
