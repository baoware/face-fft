"""Evaluate an ensemble of per-generator expert detectors on the cached test sets.

Each expert is a train_cached.py run trained on ONE generator against its own
benchmark's reals. Scores are combined per clip:
  max   a clip is fake if ANY expert thinks so (an expert that recognises its own
        generator should fire; the rest stay low)
  mean  average probability

Reported like eval_cached.py: AUC on native-rate reals vs all fakes, per-generator
AUC, and per-real-source mean score. Each expert's own AUC is listed too, so the
ensemble can be compared with its best member. Generators an expert was trained on
count as "seen" for the ensemble; everything else is unseen.

    PYTHONPATH=src python -m face_fft.eval.eval_ensemble --cache cache/v1 \
        --runs /standard/.../cached/exp_ms_T8_s42 ... --out ensemble.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from face_fft.data.cached import CachedClipDataset
from face_fft.models.pipeline import FaceFFTPipeline


def safe_auc(y, p):
    return float(roc_auc_score(y, p)) if len(set(y)) == 2 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="train_cached.py run directories")
    ap.add_argument("--tests", default="Pair1,Pair2,DeepAction,6.7m,GenVideo,DVF")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=8)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    experts = []
    for run in args.runs:
        cfg = json.loads((Path(run) / "config.json").read_text())
        m = FaceFFTPipeline(mode=cfg["mode"], model_type=cfg["arch"], temporal_frames=cfg["T"], spatial_size=(256, 256))
        m.load_state_dict(torch.load(Path(run) / "best.pt", map_location="cpu", weights_only=True))
        experts.append((Path(run).name, m.to(device).eval(), cfg))
    Ts = {cfg["T"] for _, _, cfg in experts}
    if len(Ts) != 1:
        raise SystemExit(f"experts disagree on T: {Ts}")
    T = Ts.pop()
    print(f"{len(experts)} experts, T={T}: {[n for n, _, _ in experts]}", flush=True)

    results = {"experts": [n for n, _, _ in experts], "T": T, "tests": {}}
    for name in [t for t in args.tests.split(",") if t]:
        if name == "6.7m":
            ds = CachedClipDataset(args.cache, ["6.7m", "Pair1", "Pair2", "DeepAction"], "test", T)
            ds.rows = [r for r in ds.rows if r["pair"] == "6.7m" or r["label"] == 0]
        else:
            ds = CachedClipDataset(args.cache, [name], "test", T)
        if not ds.rows:
            continue
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        P = np.zeros((len(experts), len(ds)), dtype=np.float32)
        with torch.no_grad():
            i = 0
            for x, _ in loader:
                x = x.to(device)
                for e, (_, m, _) in enumerate(experts):
                    P[e, i:i + len(x)] = torch.sigmoid(m(x).squeeze(1)).float().cpu().numpy()
                i += len(x)

        y = np.array([r["label"] for r in ds.rows])
        stride = np.array([r["stride"] for r in ds.rows])
        src = np.array([r["source"] for r in ds.rows])
        head = (y == 1) | (stride == 1)
        real1 = (y == 0) & (stride == 1)
        combos = {"max": P.max(0), "mean": P.mean(0)}
        res = {"per_expert_auc": {n: safe_auc(y[head], P[e, head]) for e, (n, _, _) in enumerate(experts)}}
        for cname, p in combos.items():
            res[cname] = dict(
                auc=safe_auc(y[head], p[head]),
                per_fake_source={s: safe_auc(np.r_[np.zeros(real1.sum()), np.ones((src == s).sum())],
                                             np.r_[p[real1], p[src == s]]) for s in sorted(set(src[y == 1]))},
                real_mean_score={s: float(p[real1 & (src == s)].mean()) for s in sorted(set(src[real1]))})
        results["tests"][name] = res

        best = max(res["per_expert_auc"].items(), key=lambda kv: kv[1] if kv[1] == kv[1] else -1)
        print(f"\n[{name}] ensemble AUC  max {res['max']['auc']:.3f}  mean {res['mean']['auc']:.3f}  "
              f"| best single expert {best[0]} {best[1]:.3f}")
        for s in res["max"]["per_fake_source"]:
            print(f"  fake {s:24s} max {res['max']['per_fake_source'][s]:.3f}  mean {res['mean']['per_fake_source'][s]:.3f}")
        for s in res["max"]["real_mean_score"]:
            print(f"  real {s:24s} mean score  max {res['max']['real_mean_score'][s]:.3f}  "
                  f"mean {res['mean']['real_mean_score'][s]:.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
