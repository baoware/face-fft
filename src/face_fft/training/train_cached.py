"""Train one detector on the format-normalised clip cache.

One run = one (mode, architecture, T, seed, training pairs, excluded sources)
configuration. The epoch with the best VALIDATION AUC is kept -- not the lowest
val loss, which rewards confident scores rather than ranking -- and the decision
threshold is chosen on that same validation set and saved, so unseen-generator
evaluation never re-tunes it.

    PYTHONPATH=src python -m face_fft.training.train_cached \
        --cache cache/v1 --train_pairs Pair1 --mode fft_whitened --arch compact \
        --T 16 --seed 42 --run_name b0_s42
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, WeightedRandomSampler

from face_fft.data.cached import CachedClipDataset
from face_fft.models.pipeline import FaceFFTPipeline


def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        probs.append(torch.sigmoid(model(x.to(device, non_blocking=True)).squeeze(1)).float().cpu())
        labels.append(y)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


def youden_threshold(labels, probs) -> float:
    """Threshold maximising TPR - FPR (= balanced accuracy) on validation."""
    fpr, tpr, thr = roc_curve(labels, probs)
    return float(np.clip(thr[np.argmax(tpr - fpr)], 0.0, 1.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--train_pairs", required=True, help="comma list, e.g. Pair1 or Pair1,Pair2")
    ap.add_argument("--val_pairs", default=None, help="defaults to --train_pairs")
    ap.add_argument("--exclude_sources", default="", help="comma list, dropped from train AND val (LOGO)")
    ap.add_argument("--mode", default="fft_whitened")
    ap.add_argument("--arch", default="compact")
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--real_strides", default="1,2,3,4", help="'1' disables frame-dropping augmentation")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--steps_per_epoch", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--out_root", default="checkpoints/cached")
    ap.add_argument("--run_name", required=True)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    split = lambda s: [p for p in s.split(",") if p]
    strides = [int(s) for s in split(args.real_strides)]
    exclude = split(args.exclude_sources)

    train_ds = CachedClipDataset(args.cache, split(args.train_pairs), "train", args.T, strides, exclude)
    val_ds = CachedClipDataset(args.cache, split(args.val_pairs or args.train_pairs), "val", args.T,
                               (1, 2, 3, 4), exclude)
    print(f"train: {len(train_ds)} rows  [{train_ds.summary()}]")
    print(f"val:   {len(val_ds)} rows  [{val_ds.summary()}]", flush=True)
    if len({r['label'] for r in train_ds.rows}) < 2 or len({r['label'] for r in val_ds.rows}) < 2:
        raise SystemExit("train and val must each contain both real and fake clips")

    g = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(train_ds.balanced_weights(), num_samples=args.steps_per_epoch * args.batch_size,
                                    replacement=True, generator=g)
    loader_kw = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True,
                     persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(train_ds, sampler=sampler, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)

    model = FaceFFTPipeline(mode=args.mode, model_type=args.arch, temporal_frames=args.T,
                            spatial_size=(256, 256)).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.BCEWithLogitsLoss()

    out = Path(args.out_root) / args.run_name
    out.mkdir(parents=True, exist_ok=True)
    print(f"mode={args.mode} arch={args.arch} T={args.T} params={n_params/1e6:.2f}M seed={args.seed} -> {out}", flush=True)

    best = {"val_auc": -1.0}
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device).float().unsqueeze(1)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y); n += len(y)
        sched.step()
        vp, vy = predict(model, val_loader, device)
        val_auc = roc_auc_score(vy, vp)
        val_loss = float(nn.functional.binary_cross_entropy(torch.tensor(vp).clamp(1e-7, 1 - 1e-7),
                                                            torch.tensor(vy).float()))
        rec = dict(epoch=epoch, train_loss=total / n, val_loss=val_loss, val_auc=val_auc, secs=time.time() - t0)
        history.append(rec)
        tag = ""
        if val_auc > best["val_auc"]:
            best = dict(epoch=epoch, val_auc=val_auc, threshold=youden_threshold(vy, vp))
            torch.save(model.state_dict(), out / "best.pt")
            tag = "  <- best"
        print(f"epoch {epoch:2d} | train loss {rec['train_loss']:.4f} | val loss {val_loss:.4f} | "
              f"val AUC {val_auc:.4f} | {rec['secs']:.0f}s{tag}", flush=True)

    config = dict(vars(args), n_params=n_params, best=best, history=history,
                  train_rows=len(train_ds), val_rows=len(val_ds))
    (out / "config.json").write_text(json.dumps(config, indent=2))
    print(f"best epoch {best['epoch']}: val AUC {best['val_auc']:.4f}, threshold {best['threshold']:.3f}")


if __name__ == "__main__":
    main()
