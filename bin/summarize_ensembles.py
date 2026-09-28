"""Combine ensemble/stacking results across seeds: AUC per test set, mean +- std,
for each way of combining the per-generator experts. The joint model trained on all
8 GenVidBench generators (Option A) is listed for reference."""

import glob
import json
import re
from collections import defaultdict

import numpy as np

TESTS = ["Pair1", "Pair2", "DeepAction", "6.7m", "GenVideo", "DVF"]
COMBOS = ["mean", "max", "stack_lr", "stack_mlp"]
fmt = lambda a: f"{np.mean(a):.3f}±{np.std(a, ddof=1):.3f}" if len(a) > 1 else (f"{a[0]:.3f}" if a else "–")

groups = defaultdict(list)
for f in sorted(glob.glob("/standard/uva-mira-drive/3d-fft_results/ensembles/ens_*_T8_s*.json")):
    mode = re.search(r"ens_(.+)_T8_s\d+", f).group(1)
    groups[mode].append(json.load(open(f)))

print(f"{'experts / combination':<34}{'seeds':>6}" + "".join(f"{t:>14}" for t in TESTS))
for mode, runs in sorted(groups.items()):
    for c in COMBOS:
        cells = [fmt([r["tests"][t][c]["auc"] for r in runs if t in r["tests"]]) for t in TESTS]
        print(f"{mode + ' / ' + c:<34}{len(runs):>6}" + "".join(f"{x:>14}" for x in cells))
    best = [fmt([max(v for v in r["tests"][t]["per_expert_auc"].values() if v == v) for r in runs if t in r["tests"]])
            for t in TESTS]
    print(f"{mode + ' / best expert (oracle)':<34}{len(runs):>6}" + "".join(f"{x:>14}" for x in best))
    print()

print("Joint model trained on all 8 generators (Option A), for reference:")
for mode in ("fft_no_mask", "fft_1d_temporal"):
    aucs = defaultdict(list)
    for f in glob.glob(f"/standard/uva-mira-drive/3d-fft_checkpoints/cached/gvb_{mode}_T8_s*/eval.json"):
        for t, v in json.load(open(f))["tests"].items():
            aucs[t].append(v["auc"])
    print(f"{mode + ' / joint':<34}{len(aucs['Pair1']):>6}" + "".join(f"{fmt(aucs[t]):>14}" for t in TESTS))
