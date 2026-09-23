"""Sanity check on the normalised cache: how well do trivial, non-spectral clip
statistics separate real from fake?

After build_cache.py every clip has the same resolution, frame rate, codec and
GOP, so container fields cannot separate the classes by construction. What can
still leak is low-level content: brightness, contrast, sharpness, frame-to-frame
motion, and how compressible the clip is under the fixed encoder. This fits a
logistic regression and a gradient-boosted classifier on those five numbers and
reports AUC on the same protocols the real model is judged on. Whatever these
reach is the floor the FFT model has to clear.

Stats are recomputed from the first T frames of every clip (clips shorter than T
are excluded), so clip length cannot leak through them -- including through
bytes/frame, where a shorter clip spends a larger share on its keyframe.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from build_cache import lowlevel_stats, reencode

FEATS = ["mean", "contrast", "sharpness", "motion", "bytes_per_frame"]


def load(cache: Path, T: int, crf: int):
    rows = []
    for idx_path in sorted(cache.glob("*.csv")):
        arr = np.load(idx_path.with_suffix(".npy"), mmap_mode="r")
        for r in csv.DictReader(open(idx_path)):
            if int(r["n_valid"]) < T:
                continue
            clip = np.ascontiguousarray(arr[int(r["row"]), :T])
            _, nbytes = reencode(clip, crf)
            r.update(lowlevel_stats(clip, nbytes))
            r["label"] = int(r["label"])
            rows.append(r)
        print(f"  loaded {idx_path.stem}", flush=True)
    return rows


def xy(rows):
    return np.array([[r[f] for f in FEATS] for r in rows], dtype=np.float64), np.array([r["label"] for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--crf", type=int, default=23)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = load(Path(args.cache), args.T, args.crf)
    sel = lambda pair, split: [r for r in rows if r["pair"] == pair and r["split"] == split]

    protocols = {
        "E1  Pair1 -> Pair1 test": ("Pair1", "Pair1"),
        "E3  Pair1 -> Pair2 test": ("Pair1", "Pair2"),
        "E3r Pair2 -> Pair1 test": ("Pair2", "Pair1"),
        "E4  Pair1 -> DeepAction test": ("Pair1", "DeepAction"),
    }
    lines = [f"Shortcut baseline on normalised cache, T={args.T}. AUC (0.5 = chance).",
             f"features: {', '.join(FEATS)}", ""]
    lines.append(f"{'protocol':<32}{'n_test':>8}{'logreg':>9}{'boosted':>9}   best single feature")
    for name, (tr_pair, te_pair) in protocols.items():
        tr, te = sel(tr_pair, "train"), sel(te_pair, "test")
        if len({r['label'] for r in te}) < 2 or len({r['label'] for r in tr}) < 2:
            lines.append(f"{name:<32}  (skipped: missing a class)"); continue
        Xtr, ytr = xy(tr); Xte, yte = xy(te)
        lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")).fit(Xtr, ytr)
        gb = HistGradientBoostingClassifier(max_iter=300, class_weight="balanced", random_state=0).fit(Xtr, ytr)
        a_lr = roc_auc_score(yte, lr.predict_proba(Xte)[:, 1])
        a_gb = roc_auc_score(yte, gb.predict_proba(Xte)[:, 1])
        singles = {f: max(a, 1 - a) for f, a in ((f, roc_auc_score(yte, Xte[:, i])) for i, f in enumerate(FEATS))}
        best = max(singles, key=singles.get)
        lines.append(f"{name:<32}{len(te):>8}{a_lr:>9.3f}{a_gb:>9.3f}   {best} ({singles[best]:.3f})")

    lines += ["", "Per-source medians (test split):", f"{'source':<30}{'lbl':>4}" + "".join(f"{f[:10]:>12}" for f in FEATS)]
    for key in sorted({(r['pair'], r['source']) for r in rows}):
        rs = [r for r in rows if (r['pair'], r['source']) == key and r['split'] == 'test']
        if rs:
            lines.append(f"{key[0] + '/' + key[1]:<30}{rs[0]['label']:>4}" +
                         "".join(f"{np.median([r[f] for r in rs]):>12.2f}" for f in FEATS))

    text = "\n".join(lines)
    print("\n" + text)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
