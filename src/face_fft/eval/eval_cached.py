"""Evaluate a detector on the clip cache under the seen/unseen protocols.

Test sets (always the TEST split of the manifest):
  Pair1, Pair2, DeepAction   fakes vs that dataset's own reals
  6.7m                       Sora/Kling/OpenSora fakes (no reals exist) vs the pooled
                             test reals of the other datasets -- flagged, because the
                             real side is then a different source from the fakes

Headline numbers use real clips at stride 1 (their native frame rate), which is
what a detector meets in practice. The stride-2/3/4 variants are reported
separately as a frame-rate invariance check: a frame-rate-agnostic model should
give the same real videos the same scores at every stride.

The decision threshold comes from the run's validation set (config.json) and is
never re-tuned on test. AUC confidence intervals resample content groups, not
clips, because clips within a group are not independent.

    PYTHONPATH=src python -m face_fft.eval.eval_cached --run_dir checkpoints/cached/b0_s42 \
        --cache cache/v1 --tests Pair1,Pair2,DeepAction,6.7m
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader

from face_fft.data.cached import CachedClipDataset
from face_fft.models.pipeline import FaceFFTPipeline


def group_bootstrap_auc(labels, probs, groups, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    by_group = defaultdict(list)
    for i, g in enumerate(groups):
        by_group[g].append(i)
    keys = list(by_group)
    aucs = []
    for _ in range(n_boot):
        idx = np.concatenate([by_group[keys[j]] for j in rng.integers(0, len(keys), len(keys))])
        if len(set(labels[idx])) == 2:
            aucs.append(roc_auc_score(labels[idx], probs[idx]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def safe_auc(y, p):
    return float(roc_auc_score(y, p)) if len(set(y)) == 2 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--tests", default="Pair1,Pair2,DeepAction,6.7m")
    ap.add_argument("--run_dir", help="output dir of train_cached.py (reads config.json + best.pt)")
    ap.add_argument("--ckpt", help="or: a bare state_dict (e.g. the old DeepAction checkpoints)")
    ap.add_argument("--mode"); ap.add_argument("--arch"); ap.add_argument("--T", type=int)
    ap.add_argument("--threshold", type=float, default=None, help="override; default = run's val threshold, else 0.5")
    ap.add_argument("--out", help="JSON path; default <run_dir>/eval.json")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=8)
    args = ap.parse_args()

    if args.run_dir:
        cfg = json.loads((Path(args.run_dir) / "config.json").read_text())
        mode, arch, T, ckpt = cfg["mode"], cfg["arch"], cfg["T"], Path(args.run_dir) / "best.pt"
        thr = cfg["best"]["threshold"]
        out_path = Path(args.out or Path(args.run_dir) / "eval.json")
    else:
        mode, arch, T, ckpt, thr = args.mode, args.arch, args.T, Path(args.ckpt), 0.5
        out_path = Path(args.out)
    if args.threshold is not None:
        thr = args.threshold

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FaceFFTPipeline(mode=mode, model_type=arch, temporal_frames=T, spatial_size=(256, 256))
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.to(device).eval()
    print(f"{ckpt}  mode={mode} arch={arch} T={T} threshold={thr:.3f}", flush=True)

    tests = [t for t in args.tests.split(",") if t]
    results = {"ckpt": str(ckpt), "mode": mode, "arch": arch, "T": T, "threshold": thr, "tests": {}}
    for name in tests:
        if name == "6.7m":
            ds = CachedClipDataset(args.cache, ["6.7m", "Pair1", "Pair2", "DeepAction"], "test", T)
            ds.rows = [r for r in ds.rows if r["pair"] == "6.7m" or r["label"] == 0]
            note = "reals pooled from Pair1/Pair2/DeepAction test: real source differs from fakes"
        else:
            ds = CachedClipDataset(args.cache, [name], "test", T)
            note = ""
        if not ds.rows:
            print(f"\n[{name}] no clips at T={T}; skipped"); continue
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        probs = []
        with torch.no_grad():
            for x, _ in loader:
                probs.append(torch.sigmoid(model(x.to(device)).squeeze(1)).float().cpu())
        p = torch.cat(probs).numpy()
        y = np.array([r["label"] for r in ds.rows])
        stride = np.array([r["stride"] for r in ds.rows])
        src = np.array([r["source"] for r in ds.rows])
        grp = np.array([r["group"] for r in ds.rows])

        head = (y == 1) | (stride == 1)                       # native-rate reals + all fakes
        yh, ph, pred = y[head], p[head], (p[head] >= thr).astype(int)
        res = {"note": note, "n_real": int(((y == 0) & (stride == 1)).sum()), "n_fake": int((y == 1).sum())}
        res["auc"] = safe_auc(yh, ph)
        if len(set(yh)) == 2:
            res["auc_ci95"] = group_bootstrap_auc(yh, ph, grp[head])
            tn, fp, fn, tp = confusion_matrix(yh, pred, labels=[0, 1]).ravel()
            res.update(confusion=dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp)),
                       f1=float(f1_score(yh, pred)), balanced_acc=float(0.5 * (tp / (tp + fn) + tn / (tn + fp))))
        res["auc_all_real_strides"] = safe_auc(y, p)
        real1 = (y == 0) & (stride == 1)
        res["per_fake_source"] = {s: dict(n=int((src == s).sum()),
                                          auc_vs_reals=safe_auc(np.r_[np.zeros(real1.sum()), np.ones((src == s).sum())],
                                                                np.r_[p[real1], p[src == s]]),
                                          detection_rate=float((p[src == s] >= thr).mean()))
                                  for s in sorted(set(src[y == 1]))}
        res["per_real_source"] = {s: dict(n=int((real1 & (src == s)).sum()),
                                          false_positive_rate=float((p[real1 & (src == s)] >= thr).mean()))
                                  for s in sorted(set(src[real1]))}
        res["real_stride_invariance"] = {int(k): dict(n=int(((y == 0) & (stride == k)).sum()),
                                                      mean_prob=float(p[(y == 0) & (stride == k)].mean()),
                                                      false_positive_rate=float((p[(y == 0) & (stride == k)] >= thr).mean()))
                                         for k in sorted(set(stride[y == 0]))}
        results["tests"][name] = res

        ci = res.get("auc_ci95")
        print(f"\n[{name}] reals={res['n_real']} fakes={res['n_fake']}  AUC {res['auc']:.3f}"
              + (f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "") + (f"  ({note})" if note else ""))
        if "confusion" in res:
            c = res["confusion"]
            print(f"  at val threshold: F1 {res['f1']:.3f}  balanced acc {res['balanced_acc']:.3f}  "
                  f"TN {c['tn']} FP {c['fp']} FN {c['fn']} TP {c['tp']}")
        for s, v in res["per_fake_source"].items():
            print(f"  fake {s:24s} n={v['n']:5d}  AUC vs reals {v['auc_vs_reals']:.3f}  detected {v['detection_rate']:.1%}")
        for s, v in res["per_real_source"].items():
            print(f"  real {s:24s} n={v['n']:5d}  false positives {v['false_positive_rate']:.1%}")
        print("  real clips by frame stride (same videos; flat = frame-rate invariant): " +
              "  ".join(f"s{k}: p={v['mean_prob']:.3f} FP={v['false_positive_rate']:.1%}"
                        for k, v in res["real_stride_invariance"].items()))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
