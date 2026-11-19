"""Train BeatNet and evaluate it.

--split inter --stage validate   train on 17 DS1 patients and score the 5 held-out
                                 DS1 patients after every epoch; used to choose the
                                 architecture and the number of epochs
--split inter --stage final      train on all 22 DS1 patients for that fixed number
                                 of epochs, then test once on the 22 DS2 patients
--split intra                    shuffle all beats and split at random, so the same
                                 patients appear in training and test (shown only
                                 to expose the leakage this causes)

Usage: python src/train_cnn.py --split inter --stage final --seed 0
"""
import argparse
import json
import time

import numpy as np
import torch
from torch import nn

from common import DS1, DS2, TRAIN, VAL, load, mask, report
from model import BeatNet
from prepare_beats import CLASSES

ap = argparse.ArgumentParser()
ap.add_argument("--split", choices=["inter", "intra"], default="inter")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--stage", choices=["validate", "final"], default="final")
ap.add_argument("--epochs", type=int, default=10)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--weight-decay", type=float, default=1e-2)
ap.add_argument("--width", type=int, default=16)
ap.add_argument("--dropout", type=float, default=0.5)
ap.add_argument("--class-weight-power", type=float, default=0.5, help="0 = unweighted loss, 1 = inverse frequency")
ap.add_argument("--no-rr", action="store_true", help="ablation: drop the RR-interval ratios")
ap.add_argument("--no-template", action="store_true", help="ablation: drop the patient-template channel")
a = ap.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed)

X, D, rr, y, rec = load()
X = X[:, None] if a.no_template else np.stack([X, D], axis=1)     # (beats, channels, samples)
if a.split == "inter" and a.stage == "validate":
    tr, va, te = mask(rec, TRAIN), mask(rec, VAL), None
elif a.split == "inter":
    tr, va, te = mask(rec, DS1), None, mask(rec, DS2)
else:
    order = np.random.permutation(len(y))
    n_tr, n_va = int(0.7 * len(y)), int(0.1 * len(y))
    tr, va, te = (np.zeros(len(y), bool) for _ in range(3))
    tr[order[:n_tr]] = True; va[order[n_tr:n_tr + n_va]] = True; te[order[n_tr + n_va:]] = True

mu, sd = rr[tr].mean(0), rr[tr].std(0)                 # scale RR features on training data only
t = lambda arr: torch.from_numpy(arr)
Xt, Rt, yt = t(X), t((rr - mu) / sd), t(y)

counts = np.bincount(y[tr], minlength=len(CLASSES))
weights = torch.tensor((counts.max() / np.maximum(counts, 1)) ** a.class_weight_power, dtype=torch.float32)
net = BeatNet(use_rr=not a.no_rr, in_channels=X.shape[1], width=a.width, dropout=a.dropout)
opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.weight_decay)
idx_tr = np.flatnonzero(tr)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * (len(idx_tr) // 256 + 1))
loss_fn = nn.CrossEntropyLoss(weight=weights)


def augment(b):
    """Random gain, small time shift and low-level noise."""
    b = b * (1 + 0.6 * (torch.rand(len(b), 1, 1) - 0.5))
    shift = int(np.random.randint(-3, 4))
    return torch.roll(b, shift, dims=2) + 0.02 * torch.randn_like(b)


@torch.no_grad()
def predict(m):
    net.eval()
    idx = np.flatnonzero(m)
    out = [net(Xt[idx[i:i + 2048]], Rt[idx[i:i + 2048]]).argmax(1) for i in range(0, len(idx), 2048)]
    return torch.cat(out).numpy()


history, best, best_state, t0 = [], -1, None, time.time()
for epoch in range(a.epochs):
    net.train()
    perm, total = np.random.permutation(idx_tr), 0.0
    for i in range(0, len(perm), 256):
        b = perm[i:i + 256]
        loss = loss_fn(net(augment(Xt[b]), Rt[b]), yt[b])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        total += loss.item() * len(b)
    row = {"epoch": epoch + 1, "train_loss": total / len(perm)}
    line = f"epoch {epoch + 1:2d}  loss {row['train_loss']:.4f}"
    if va is not None:
        val = report(y[va], predict(va))
        row.update(val_macro_f1=val["macro_f1"], **{f"val_f1_{c}": val[c]["f1"] for c in CLASSES})
        line += f"  val macro F1 {val['macro_f1']:.3f}  " + " ".join(f"{c} {val[c]['f1']:.2f}" for c in CLASSES)
        if a.split == "intra" and val["macro_f1"] > best:      # random split keeps its best epoch
            best, best_state = val["macro_f1"], {k: v.clone() for k, v in net.state_dict().items()}
    history.append(row)
    print(line + f"  {time.time() - t0:.0f}s", flush=True)

if best_state is not None:
    net.load_state_dict(best_state)
result = {"split": a.split, "stage": a.stage, "seed": a.seed, "use_rr": not a.no_rr,
          "use_template": not a.no_template, "history": history}
tag = (f"{a.split}{'_validate' if te is None else ''}{'_norr' if a.no_rr else ''}"
       f"{'_notemplate' if a.no_template else ''}_seed{a.seed}")
if te is not None:
    pred = predict(te)
    result["test"] = r = report(y[te], pred)
    if a.split == "inter":                              # per-patient results for the test set
        result["per_record"] = {str(k): report(y[te & (rec == k)], pred[rec[te] == k])
                                for k in sorted(set(rec[te]))}
        if not a.no_rr and not a.no_template and a.seed == 0:
            torch.save(net.state_dict(), "results/beatnet_seed0.pt")
    print(f"TEST {tag}  macro F1 {r['macro_f1']:.3f}  accuracy {r['accuracy']:.3f}  "
          + "  ".join(f"{c} Se {r[c]['sensitivity']:.2f} PPV {r[c]['ppv']:.2f}" for c in CLASSES))
json.dump(result, open(f"results/cnn_{tag}.json", "w"), indent=1)
