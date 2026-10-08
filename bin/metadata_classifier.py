"""Metadata-only detector: real vs fake from container fields alone, trained on one
benchmark and tested on another -- the same protocols the spectral detectors face.

This is the plan's "metadata-only classifier" sanity baseline. Within a benchmark it
measures how far file format alone goes; across benchmarks it shows whether format
conventions transfer (they need not: each benchmark encodes its reals and fakes its
own way). Any detector that does not beat this on a protocol is not demonstrably
detecting generation.

Inputs are the CSVs written by bin/format_audit.py (400 probed files per source).
"""

import argparse
from collections import defaultdict
from pathlib import Path

import csv

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATS = ["width", "height", "fps", "n_frames", "duration_s", "bit_rate_kbps", "bits_per_pixel", "gop"]
CODECS = ["h264", "mpeg4", "hevc", "vp9", "av1"]


def load(paths):
    rows = []
    for p in paths:
        rows.extend(csv.DictReader(open(p)))
    return rows


def to_float(v):
    try:
        f = float(v)
        return f if np.isfinite(f) else -1.0
    except (TypeError, ValueError):
        return -1.0


def X(rows):
    out = []
    for r in rows:
        v = [to_float(r[f]) for f in FEATS]
        v.append(np.log1p(max(to_float(r["bit_rate_kbps"]), 0.0)))
        v += [float(r["codec"] == c) for c in CODECS]
        out.append(v)
    return np.array(out, dtype=np.float64)


def y(rows):
    return np.array([int(r["label"]) for r in rows])


def models():
    return {"logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced")),
            "boosted": HistGradientBoostingClassifier(max_iter=300, class_weight="balanced", random_state=0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audits", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = load(args.audits)
    benches = sorted({r["pair"] for r in rows})
    print("benchmarks:", {b: f"{sum(r['pair'] == b for r in rows)} files, "
                             f"{sum(r['pair'] == b and r['label'] == '0' for r in rows)} real" for b in benches})

    protocols = [
        ("Pair1 (5-fold CV)", "Pair1", None),
        ("Pair1 -> Pair2", "Pair1", "Pair2"),
        ("Pair2 -> Pair1", "Pair2", "Pair1"),
        ("GenVideo train -> GenVideo-Val", "GenVideo:train", "GenVideo:test"),
        ("GenVideo train -> Pair2", "GenVideo:train", "Pair2"),
        ("Pair1 -> GenVideo-Val", "Pair1", "GenVideo:test"),
        ("Pair1 -> DeepAction", "Pair1", "DeepAction:*"),
        ("Pair1 -> DVF", "Pair1", "DVF:test"),
        ("GenVideo train -> DVF", "GenVideo:train", "DVF:test"),
        ("GenVideo train -> DeepAction", "GenVideo:train", "DeepAction:*"),
    ]
    sel = lambda b: [r for r in rows if (r["pair"].startswith(b[:-1]) if b.endswith("*") else r["pair"] == b)]
    lines = ["Metadata-only detector (container fields + codec one-hot). AUC; 0.5 = chance.", "",
             f"{'protocol':<34}{'n_test':>8}{'logreg':>9}{'boosted':>9}"]
    for name, tr, te in protocols:
        dtr = sel(tr)
        if len(set(y(dtr))) < 2:
            lines.append(f"{name:<34}  (no data)"); continue
        if te is None:
            res = []
            for m in models().values():
                pr = cross_val_predict(m, X(dtr), y(dtr), cv=StratifiedKFold(5, shuffle=True, random_state=0),
                                       method="predict_proba")[:, 1]
                res.append(roc_auc_score(y(dtr), pr))
            lines.append(f"{name:<34}{len(dtr):>8}{res[0]:>9.3f}{res[1]:>9.3f}"); continue
        dte = sel(te)
        if len(set(y(dte))) < 2:
            lines.append(f"{name:<34}  (test set lacks a class)"); continue
        res = [roc_auc_score(y(dte), m.fit(X(dtr), y(dtr)).predict_proba(X(dte))[:, 1]) for m in models().values()]
        lines.append(f"{name:<34}{len(dte):>8}{res[0]:>9.3f}{res[1]:>9.3f}")
    text = "\n".join(lines)
    print(text)
    Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
