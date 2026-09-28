"""Two follow-ups on the per-generator experts, from the per-clip scores that
eval_ensemble.py --dump_dir saves.

1. Leave-one-dataset-out combiner. Can you predict which experts will transfer to a
   NEW dataset? For each unseen dataset H, a logistic regression over the 8 expert
   logits is fit on the OTHER unseen datasets and tested on H. Compared with the
   plain mean, the GenVidBench-validation stacker and the best single expert
   (an oracle: chosen by looking at H). Each fitting dataset gets equal total weight,
   and each class within it, so GenVideo's 13K clips do not drown out DeepAction.

   The Sora/Kling/OpenSora set ("6.7m") has no reals of its own; here it uses only
   Pair1/Pair2 reals, so its reals never overlap DeepAction's Pexels clips.

2. Frame-rate sensitivity. The same REAL clips are cached at frame strides 1-4 (every
   k-th frame, i.e. more motion per frame at larger k). For each expert: how much its
   logit on real clips rises from stride 1 to stride 4. If experts learned "more
   motion than the reals = fake", this should predict the flip between DeepAction
   (25 fps reals, little motion per frame) and GenVideo-Val (3 fps reals, a lot).
"""

import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

UNSEEN = ["DeepAction", "6.7m", "GenVideo", "DVF"]


def logit(P):
    p = np.clip(P, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).T


def load(run_dir: Path):
    d = {}
    for f in run_dir.glob("*.npz"):
        z = np.load(f, allow_pickle=True)
        d[f.stem] = {k: z[k] for k in z.files}
    return d


def head(ds, name):
    """Fakes + native-rate reals; for 6.7m keep only Pair1/Pair2 reals (no Pexels overlap)."""
    m = (ds["y"] == 1) | (ds["stride"] == 1)
    if name == "6.7m":
        m &= (ds["y"] == 1) | np.isin(ds["src"], ["vript", "hd_vg_130m"])
    return m


def lodo(data):
    rows = {}
    for H in UNSEEN:
        X, y, w = [], [], []
        for D in UNSEEN:
            if D == H:
                continue
            m = head(data[D], D)
            yd = data[D]["y"][m]
            X.append(logit(data[D]["P"][:, m])); y.append(yd)
            wd = np.where(yd == 1, 0.5 / max((yd == 1).sum(), 1), 0.5 / max((yd == 0).sum(), 1))
            w.append(wd)
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
        clf.fit(np.vstack(X), np.concatenate(y), logisticregression__sample_weight=np.concatenate(w))
        m = head(data[H], H)
        Ph, yh = data[H]["P"][:, m], data[H]["y"][m]
        rows[H] = dict(lodo=roc_auc_score(yh, clf.predict_proba(logit(Ph))[:, 1]),
                       mean=roc_auc_score(yh, Ph.mean(0)),
                       oracle=max(roc_auc_score(yh, Ph[e]) for e in range(len(Ph))),
                       weights=clf[-1].coef_[0].round(2).tolist())
    return rows


def stride_sensitivity(data, experts):
    """Per expert: mean over real sources of [mean logit at stride 4 - at stride 1]."""
    out = np.zeros(len(experts))
    for e in range(len(experts)):
        deltas = []
        for name, ds in data.items():
            if name == "6.7m":
                continue                                    # its reals duplicate other test sets
            for s in set(ds["src"][ds["y"] == 0]):
                r = (ds["y"] == 0) & (ds["src"] == s)
                L = logit(ds["P"][e:e + 1, r])[:, 0]
                st = ds["stride"][r]
                if (st == 1).any() and (st == 4).any():
                    deltas.append(L[st == 4].mean() - L[st == 1].mean())
        out[e] = np.mean(deltas)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="/standard/uva-mira-drive/3d-fft_results/ensembles/scores")
    ap.add_argument("--ensembles", default="/standard/uva-mira-drive/3d-fft_results/ensembles")
    ap.add_argument("--out", default="/standard/uva-mira-drive/3d-fft_results/expert_transfer_analysis.txt")
    args = ap.parse_args()

    lines = []
    by_mode = defaultdict(list)
    points = []                                             # (mode, expert, sensitivity, DA auc - GV auc)
    for run_dir in sorted(Path(args.scores).glob("ens_*_T8_s*")):
        mode = re.search(r"ens_(.+)_T8_s\d+", run_dir.name).group(1)
        data = load(run_dir)
        if not all(u in data for u in UNSEEN):
            continue
        experts = [re.sub(r"exp(_1d)?_(.+)_T8_s\d+", r"\2", str(n)) for n in data["DVF"]["experts"]]
        res = lodo(data)
        gvb = json.loads((Path(args.ensembles) / f"{run_dir.name}.json").read_text())
        for H in UNSEEN:
            res[H]["gvb_stack"] = gvb["tests"][H]["stack_lr"]["auc"]
        by_mode[mode].append(res)

        sens = stride_sensitivity(data, experts)
        for e, g in enumerate(experts):
            auc = lambda D: roc_auc_score(data[D]["y"][head(data[D], D)], data[D]["P"][e, head(data[D], D)])
            points.append((mode, g, sens[e], auc("DeepAction") - auc("GenVideo")))

    fmt = lambda a: f"{np.mean(a):.3f}±{np.std(a, ddof=1):.3f}" if len(a) > 1 else f"{a[0]:.3f}"
    lines.append("1. Leave-one-dataset-out combiner: fit on the other 3 unseen datasets, test on the held-out one.")
    lines.append("   AUC, mean ± std over seeds.\n")
    for mode, runs in by_mode.items():
        lines.append(f"{mode} experts ({len(runs)} seeds)")
        lines.append(f"  {'held-out':<12}{'mean':>14}{'GVB-val stack':>15}{'LODO stack':>14}{'oracle expert':>15}")
        for H in UNSEEN:
            g = lambda k: fmt([r[H][k] for r in runs])
            lines.append(f"  {H:<12}{g('mean'):>14}{g('gvb_stack'):>15}{g('lodo'):>14}{g('oracle'):>15}")
        lines.append("")

    lines.append("2. Frame-rate sensitivity vs the DeepAction / GenVideo-Val flip, one point per expert and seed.")
    lines.append("   sensitivity = logit(stride 4) - logit(stride 1) on the same real clips (+ = more motion looks more fake)")
    lines.append("   flip        = AUC(DeepAction) - AUC(GenVideo-Val)\n")
    for mode in sorted({p[0] for p in points}):
        pts = [p for p in points if p[0] == mode]
        per = defaultdict(list)
        for _, g, s, f in pts:
            per[g].append((s, f))
        lines.append(f"{mode}:  {'expert':<10}{'sensitivity':>13}{'flip':>9}")
        for g, v in per.items():
            lines.append(f"{'':<{len(mode) + 3}}{g:<10}{np.mean([a for a, _ in v]):>+13.2f}{np.mean([b for _, b in v]):>+9.3f}")
        s, f = np.array([p[2] for p in pts]), np.array([p[3] for p in pts])
        lines.append(f"  n={len(pts)}  Pearson r={pearsonr(s, f)[0]:+.2f} (p={pearsonr(s, f)[1]:.1e})  "
                     f"Spearman rho={spearmanr(s, f)[0]:+.2f} (p={spearmanr(s, f)[1]:.1e})\n")
    s, f = np.array([p[2] for p in points]), np.array([p[3] for p in points])
    lines.append(f"all experts: n={len(points)}  Pearson r={pearsonr(s, f)[0]:+.2f} (p={pearsonr(s, f)[1]:.1e})  "
                 f"Spearman rho={spearmanr(s, f)[0]:+.2f} (p={spearmanr(s, f)[1]:.1e})")

    text = "\n".join(lines)
    print(text)
    Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
