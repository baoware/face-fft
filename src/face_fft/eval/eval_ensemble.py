"""Evaluate an ensemble of per-generator expert detectors on the cached test sets.

Each expert is a train_cached.py run trained on ONE generator against its own
benchmark's reals. Scores are combined per clip:
  max   a clip is fake if ANY expert thinks so (an expert that recognises its own
        generator should fire; the rest stay low)
  mean  average probability

  stack_lr   logistic regression on the experts' logit scores
  stack_mlp  small MLP (one hidden layer) on the experts' concatenated penultimate
             features (the input to each model's final linear layer)
Both stacking layers are fit ONLY on the validation splits of the experts' own
benchmarks (--fit_pairs, default Pair1+Pair2): clips the experts never trained on,
and never any test clip. The unseen test sets therefore stay untouched.

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
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from face_fft.data.cached import CachedClipDataset
from face_fft.models.pipeline import FaceFFTPipeline


def safe_auc(y, p):
    return float(roc_auc_score(y, p)) if len(set(y)) == 2 else float("nan")


def last_linear(model: nn.Module) -> nn.Linear:
    return [m for m in model.modules() if isinstance(m, nn.Linear)][-1]


@torch.no_grad()
def score(experts, ds, device, batch_size, num_workers):
    """Per-expert probabilities (E, N) and concatenated penultimate features (N, D)."""
    feats = {i: [] for i in range(len(experts))}
    hooks = [last_linear(m).register_forward_pre_hook(lambda mod, inp, i=i: feats[i].append(inp[0].float().cpu()))
             for i, (_, m, _) in enumerate(experts)]
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    P = np.zeros((len(experts), len(ds)), dtype=np.float32)
    try:
        i = 0
        for x, _ in loader:
            x = x.to(device)
            for e, (_, m, _) in enumerate(experts):
                P[e, i:i + len(x)] = torch.sigmoid(m(x).squeeze(1)).float().cpu().numpy()
            i += len(x)
    finally:
        for h in hooks:
            h.remove()
    F = torch.cat([torch.cat(feats[i]) for i in range(len(experts))], dim=1).numpy()
    return P, F


def logit(P):
    p = np.clip(P, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).T          # (N, E)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="train_cached.py run directories")
    ap.add_argument("--tests", default="Pair1,Pair2,DeepAction,6.7m,GenVideo,DVF")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fit_pairs", default="Pair1,Pair2", help="val splits the stacking layers are fit on")
    ap.add_argument("--dump_dir", default=None, help="also save per-clip expert scores per test set (.npz)")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=8)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    experts = []
    for run in args.runs:
        cfg = json.loads((Path(run) / "config.json").read_text())
        m = FaceFFTPipeline(mode=cfg["mode"], model_type=cfg["arch"], temporal_frames=cfg["T"], spatial_size=(256, 256),
                            kinetics_norm=bool(cfg.get("pretrained", False)))
        m.load_state_dict(torch.load(Path(run) / "best.pt", map_location="cpu", weights_only=True))
        experts.append((Path(run).name, m.to(device).eval(), cfg))
    Ts = {cfg["T"] for _, _, cfg in experts}
    if len(Ts) != 1:
        raise SystemExit(f"experts disagree on T: {Ts}")
    T = Ts.pop()
    print(f"{len(experts)} experts, T={T}: {[n for n, _, _ in experts]}", flush=True)

    # stacking layers, fit on validation clips only
    val = CachedClipDataset(args.cache, [p for p in args.fit_pairs.split(",") if p], "val", T)
    Pv, Fv = score(experts, val, device, args.batch_size, args.num_workers)
    yv = np.array([r["label"] for r in val.rows])
    stack_lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced")).fit(logit(Pv), yv)
    stack_mlp = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-3, early_stopping=True,
                                                              max_iter=500, random_state=0)).fit(Fv, yv)
    lr_w = dict(zip([n for n, _, _ in experts], stack_lr[-1].coef_[0].round(3).tolist()))
    print(f"stacking fit on {len(val)} val clips ({int((yv == 0).sum())} real); LR weights per expert: {lr_w}", flush=True)

    results = {"experts": [n for n, _, _ in experts], "T": T, "fit_pairs": args.fit_pairs,
               "stack_lr_weights": lr_w, "tests": {}}
    for name in [t for t in args.tests.split(",") if t]:
        if name == "6.7m":
            ds = CachedClipDataset(args.cache, ["6.7m", "Pair1", "Pair2", "DeepAction"], "test", T)
            ds.rows = [r for r in ds.rows if r["pair"] == "6.7m" or r["label"] == 0]
        else:
            ds = CachedClipDataset(args.cache, [name], "test", T)
        if not ds.rows:
            continue
        P, F = score(experts, ds, device, args.batch_size, args.num_workers)

        y = np.array([r["label"] for r in ds.rows])
        stride = np.array([r["stride"] for r in ds.rows])
        src = np.array([r["source"] for r in ds.rows])
        head = (y == 1) | (stride == 1)
        real1 = (y == 0) & (stride == 1)
        combos = {"max": P.max(0), "mean": P.mean(0),
                  "stack_lr": stack_lr.predict_proba(logit(P))[:, 1],
                  "stack_mlp": stack_mlp.predict_proba(F)[:, 1]}
        res = {"per_expert_auc": {n: safe_auc(y[head], P[e, head]) for e, (n, _, _) in enumerate(experts)}}
        for cname, p in combos.items():
            res[cname] = dict(
                auc=safe_auc(y[head], p[head]),
                per_fake_source={s: safe_auc(np.r_[np.zeros(real1.sum()), np.ones((src == s).sum())],
                                             np.r_[p[real1], p[src == s]]) for s in sorted(set(src[y == 1]))},
                real_mean_score={s: float(p[real1 & (src == s)].mean()) for s in sorted(set(src[real1]))})
        results["tests"][name] = res
        if args.dump_dir:
            Path(args.dump_dir).mkdir(parents=True, exist_ok=True)
            np.savez_compressed(Path(args.dump_dir) / f"{name}.npz", P=P, y=y, stride=stride, src=src,
                                grp=np.array([r["group"] for r in ds.rows]),
                                experts=np.array([n for n, _, _ in experts]))

        best = max(res["per_expert_auc"].items(), key=lambda kv: kv[1] if kv[1] == kv[1] else -1)
        print(f"\n[{name}] AUC  " + "  ".join(f"{c} {res[c]['auc']:.3f}" for c in combos) +
              f"  | best single expert {best[0]} {best[1]:.3f}")
        for src_name in res["max"]["per_fake_source"]:
            print(f"  fake {src_name:24s} " + "  ".join(f"{c} {res[c]['per_fake_source'][src_name]:.3f}" for c in combos))
        for src_name in res["max"]["real_mean_score"]:
            print(f"  real {src_name:24s} mean score " + "  ".join(f"{c} {res[c]['real_mean_score'][src_name]:.3f}" for c in combos))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
