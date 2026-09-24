"""Collect every cached-training run into one table: AUC per test set and per
generator, mean +- std over seeds for each configuration.

A configuration is everything except the seed (mode, arch, T, training pairs,
real strides, excluded sources). Differences smaller than the seed spread should
not be read as effects -- the same DeepAction script moved ~9 accuracy points
between seeds.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

KEY = ("mode", "arch", "T", "train_pairs", "real_strides", "exclude_sources")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/standard/uva-mira-drive/3d-fft_checkpoints/cached")
    ap.add_argument("--out", default="/standard/uva-mira-drive/3d-fft_results/runs_summary.md")
    args = ap.parse_args()

    groups = defaultdict(list)
    for run in sorted(Path(args.root).iterdir()):
        cfg_p, ev_p = run / "config.json", run / "eval.json"
        if not (cfg_p.exists() and ev_p.exists()):
            continue
        cfg, ev = json.loads(cfg_p.read_text()), json.loads(ev_p.read_text())
        groups[tuple(cfg.get(k) for k in KEY)].append((cfg["seed"], run.name, ev))

    tests = ["Pair1", "Pair2", "DeepAction", "6.7m", "GenVideo"]
    fmt = lambda a: f"{np.mean(a):.3f} ± {np.std(a, ddof=1):.3f}" if len(a) > 1 else (f"{a[0]:.3f}" if a else "–")
    lines = ["# Cached-training runs", "",
             "AUC, native-rate real clips vs all fakes of the test set; mean ± std over seeds.", "",
             "| mode | arch | T | train | strides | excl | seeds | " + " | ".join(tests) + " |",
             "|" + "---|" * (7 + len(tests))]
    for key, runs in sorted(groups.items(), key=lambda kv: (kv[0][2], kv[0][0])):
        mode, arch, T, pairs, strides, excl = key
        cells = [fmt([r[2]["tests"][t]["auc"] for r in runs if t in r[2]["tests"]]) for t in tests]
        lines.append(f"| {mode} | {arch} | {T} | {pairs} | {strides} | {excl or '–'} | "
                     f"{','.join(str(r[0]) for r in sorted(runs))} | " + " | ".join(cells) + " |")

    lines += ["", "## Per-generator AUC (vs that test set's native-rate reals)", ""]
    for key, runs in sorted(groups.items(), key=lambda kv: (kv[0][2], kv[0][0])):
        mode, arch, T = key[:3]
        per = defaultdict(list)
        for _, _, ev in runs:
            for t, res in ev["tests"].items():
                for g, v in res["per_fake_source"].items():
                    if v["auc_vs_reals"] == v["auc_vs_reals"]:
                        per[f"{t}/{g}"].append(v["auc_vs_reals"])
        lines.append(f"**{mode}, {arch}, T={T}** ({len(runs)} seed{'s' if len(runs) > 1 else ''}): " +
                     ", ".join(f"{g} {fmt(a)}" for g, a in sorted(per.items())))
        lines.append("")

    text = "\n".join(lines)
    Path(args.out).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
