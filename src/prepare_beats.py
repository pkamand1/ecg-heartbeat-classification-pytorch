"""Segment MIT-BIH Arrhythmia Database records into labelled heartbeats.

Each annotated beat becomes one fixed-length window centred near its R peak,
with four RR-interval ratios that describe the rhythm around it. Labels
follow the AAMI EC57 grouping (N, S, V, F). Paced records are excluded, as
the standard recommends.

Usage: python src/prepare_beats.py --mitdb data/mitdb --out data/beats.npz
"""
import argparse
from pathlib import Path

import numpy as np
import wfdb
from scipy.signal import butter, filtfilt

FS = 360                      # sampling rate of the database, Hz
PRE, POST = 90, 144           # 0.25 s before and 0.40 s after the R peak
STEP = 2                      # keep every second sample (180 Hz); the signal is band-limited to 40 Hz
PACED = {"102", "104", "107", "217"}

# de Chazal et al. (2004) inter-patient division: no patient is in both sets
DS1 = ["101", "106", "108", "109", "112", "114", "115", "116", "118", "119", "122",
       "124", "201", "203", "205", "207", "208", "209", "215", "220", "223", "230"]
DS2 = ["100", "103", "105", "111", "113", "117", "121", "123", "200", "202", "210",
       "212", "213", "214", "219", "221", "222", "228", "231", "232", "233", "234"]

CLASSES = ["N", "S", "V", "F"]
AAMI = {**dict.fromkeys("NLRej", 0),   # normal and bundle branch block beats
        **dict.fromkeys("AaJS", 1),    # supraventricular ectopic beats
        **dict.fromkeys("VE", 2),      # ventricular ectopic beats
        "F": 3}                        # fusion of ventricular and normal
UNCLASSIFIED = set("/fQ")              # paced and unclassifiable beats (class Q), not modelled
BEATS = set(AAMI) | UNCLASSIFIED


def bandpass(x, low=0.5, high=40.0, order=3):
    """Zero-phase Butterworth filter that removes baseline wander and mains noise."""
    b, a = butter(order, [low / (FS / 2), high / (FS / 2)], btype="band")
    return filtfilt(b, a, x)


def load_record(path):
    rec = wfdb.rdrecord(str(path))
    names = [n.upper() for n in rec.sig_name]
    lead = names.index("MLII") if "MLII" in names else 0
    sig = bandpass(rec.p_signal[:, lead])
    sig = (sig - sig.mean()) / sig.std()           # per-record scaling, no labels involved
    ann = wfdb.rdann(str(path), "atr")
    keep = [i for i, s in enumerate(ann.symbol) if s in BEATS]
    return sig, ann.sample[keep], [ann.symbol[i] for i in keep], rec.sig_name[lead]


def rr_features(r_peaks):
    """Rhythm around each beat, expressed relative to the patient's own rhythm.

    Ratios are used instead of intervals in seconds because resting heart rate
    differs between patients, and absolute intervals do not transfer to new ones.
    Columns: previous RR / local mean, next RR / local mean, previous RR / record
    mean, next RR / previous RR. The local mean covers the ten preceding intervals.
    """
    rr = np.diff(r_peaks) / FS
    pre = np.r_[rr[0], rr]
    post = np.r_[rr, rr[-1]]
    local = np.array([pre[max(0, i - 9): i + 1].mean() for i in range(len(pre))])
    return np.c_[pre / local, post / local, pre / rr.mean(), post / pre].astype(np.float32)


def segment(mitdb):
    X, R, y, rec_id = [], [], [], []
    for name in DS1 + DS2:
        sig, peaks, symbols, lead = load_record(Path(mitdb) / name)
        feats = rr_features(peaks)
        n = 0
        for p, s, f in zip(peaks, symbols, feats):
            if s in UNCLASSIFIED or p - PRE < 0 or p + POST > len(sig):
                continue
            X.append(sig[p - PRE: p + POST: STEP]); R.append(f); y.append(AAMI[s]); rec_id.append(int(name))
            n += 1
        print(f"record {name}  lead {lead:5s}  beats {n}")
    X, rec_id = np.asarray(X, np.float32), np.asarray(rec_id, np.int64)
    return X, np.asarray(R, np.float32), np.asarray(y, np.int64), rec_id, patient_templates(X, rec_id)


def patient_templates(X, rec_id):
    """Each beat's own-patient reference: the median of every beat in that record.

    No labels are used, so the template can be computed for a new patient before
    any beat is classified. Subtracting it shows how far a beat departs from what
    is usual for that patient, which matters for patients whose every beat is wide
    (bundle branch block) and would otherwise look ventricular.
    """
    T = np.empty_like(X)
    for r in np.unique(rec_id):
        m = rec_id == r
        T[m] = np.median(X[m], axis=0)
    return T


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mitdb", default="data/mitdb")
    ap.add_argument("--out", default="data/beats.npz")
    a = ap.parse_args()
    X, R, y, rec, T = segment(a.mitdb)
    np.savez_compressed(a.out, X=X, rr=R, y=y, record=rec, template=T)
    for split, ids in (("DS1 (train)", DS1), ("DS2 (test)", DS2)):
        m = np.isin(rec, [int(i) for i in ids])
        print(split, {c: int((y[m] == k).sum()) for k, c in enumerate(CLASSES)})
