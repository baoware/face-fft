"""
Full ablation sweep: all (pipeline_mode × model_type) combinations.

Trains each combination on the combined DeepAction + GenVidBench dataset,
then evaluates on:
  - DeepAction test set  → accuracy, precision, recall, F1, AUC
  - GenVidBench fakes    → per-generator detection rate (fake recall)

Results are written incrementally to a CSV so a partial run is never lost.

Usage (interactive or via Slurm):
  PYTHONPATH=src python -m face_fft.training.run_sweep
  PYTHONPATH=src python -m face_fft.training.run_sweep --eval-only   # skip training
"""

import argparse
import csv
import os
import time
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import ConcatDataset, DataLoader

from face_fft.data.deepaction import get_deepaction_splits
from face_fft.data.genvidbench import (
    collect_mp4s,
    get_genvidbench_splits,
)
from face_fft.models.pipeline import FaceFFTPipeline
from face_fft.models.pipeline_mode import PipelineMode
from face_fft.training.trainer import Trainer


# ---------------------------------------------------------------------------
# Sweep configuration — edit to change the search space
# ---------------------------------------------------------------------------

PIPELINE_MODES = [
    PipelineMode.PIXEL_BASELINE,
    PipelineMode.FFT_NO_MASK,
    PipelineMode.FFT_LEARNABLE_MASK,
]

MODEL_TYPES = ["compact", "r3d_18", "mc3_18", "r2plus1d_18"]

DEEPACTION_ROOT = "/scratch/rjr6zk/face-fft/src/face_fft/data/deepaction_dataset"
GENVIDBENCH_ROOT = "/scratch/rjr6zk/face-fft/src/face_fft/data/genvidbench_dataset"

CHECKPOINT_DIR = "checkpoints/sweep"
CSV_PATH = "results/sweep_results.csv"

GVB_SYNTH_GENERATORS = ["Sora", "Kling", "OpenSora"]
GVB_MAX_VIDS_PER_GEN = 200  # cap per generator for GVB fake-recall eval
GVB_MIN_FILE_BYTES = 1024
GVB_EVAL_SEED = 10  # fixed seed for per-generator video subsampling

TARGET_FRAMES = 8
TARGET_SIZE = (256, 256)
BATCH_SIZE = 8
EPOCHS = 25
LEARNING_RATE = 1e-3
NUM_WORKERS = 4
SEED = 42

# ---------------------------------------------------------------------------


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(mode: PipelineMode, model_type: str) -> FaceFFTPipeline:
    return FaceFFTPipeline(
        log_scale=True,
        in_channels=3,
        base_channels=16,
        num_classes=1,
        model_type=model_type,
        mode=mode,
        temporal_frames=TARGET_FRAMES,
        spatial_size=TARGET_SIZE,
    )


def checkpoint_path(mode: PipelineMode, model_type: str) -> str:
    return os.path.join(CHECKPOINT_DIR, f"{mode.value}_{model_type}.pt")


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def build_combined_datasets(deepaction_root: str, genvidbench_root: str):
    """Return (train_dataset, val_dataset, deepaction_test_synth_models)."""
    all_folders = [
        d
        for d in os.listdir(deepaction_root)
        if os.path.isdir(os.path.join(deepaction_root, d))
    ]
    synth_models = [d for d in all_folders if d != "Pexels" and not d.startswith(".")]
    print(f"  DeepAction synthetic models: {synth_models}")

    da_train, da_val, _ = get_deepaction_splits(
        root_dir=deepaction_root,
        synth_models=synth_models,
        train_ratio=0.8,
        val_ratio=0.1,
        target_frames=TARGET_FRAMES,
        target_size=TARGET_SIZE,
        seed=SEED,
    )

    gvb_real_dir = Path(genvidbench_root) / "Real"
    real_override = None
    # Using a different seed gives GVB a different shuffle order over the Pexels
    # pool, reducing (not eliminating) overlap with the DeepAction training sample.
    # If the Pexels pool is small relative to the required n, some duplicates are
    # unavoidable in fallback mode — download GVB Real/ to avoid this entirely.
    gvb_seed = SEED
    if not gvb_real_dir.exists():
        print("  GVB Real/ absent — using DeepAction Pexels as real source for GVB.")
        real_override = collect_mp4s(str(Path(deepaction_root) / "Pexels"))
        if not real_override:
            raise FileNotFoundError(
                f"No .mp4 files found in {Path(deepaction_root) / 'Pexels'}. "
                "Check DEEPACTION_ROOT."
            )
        gvb_seed = SEED + 1  # different shuffle → different sample → less duplication

    gvb_train, gvb_val, _ = get_genvidbench_splits(
        root_dir=genvidbench_root,
        synth_generators=GVB_SYNTH_GENERATORS,
        real_folder="Real",
        real_paths_override=real_override,
        train_ratio=0.8,
        val_ratio=0.1,
        target_frames=TARGET_FRAMES,
        target_size=TARGET_SIZE,
        seed=gvb_seed,
    )

    train_ds = ConcatDataset([da_train, gvb_train])
    val_ds = ConcatDataset([da_val, gvb_val])
    return train_ds, val_ds, synth_models


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_run(
    mode: PipelineMode,
    model_type: str,
    train_ds,
    val_ds,
    device: str,
) -> float:
    """Train one combination; return wall-clock seconds."""
    model = build_model(mode, model_type)
    ckpt = checkpoint_path(mode, model_type)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    t0 = time.time()
    trainer.train(num_epochs=EPOCHS, save_path=ckpt)
    return time.time() - t0


# ---------------------------------------------------------------------------
# Evaluation — DeepAction test set
# ---------------------------------------------------------------------------


def eval_deepaction(
    model: torch.nn.Module,
    synth_models: List[str],
    device: str,
) -> Dict[str, float]:
    """Evaluate on the DeepAction held-out test split."""
    from face_fft.eval.deepaction_evaluator import evaluate_subset

    y_true, y_pred, y_prob = evaluate_subset(
        synth_models, DEEPACTION_ROOT, model, device, batch_size=BATCH_SIZE
    )

    return {
        "da_accuracy": float(accuracy_score(y_true, y_pred)),
        "da_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "da_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "da_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "da_auc": float(roc_auc_score(y_true, y_prob))
        if len(np.unique(y_true)) > 1
        else float("nan"),
    }


# ---------------------------------------------------------------------------
# Evaluation — GenVidBench fake-recall
# ---------------------------------------------------------------------------


def eval_genvidbench(model: torch.nn.Module, device: str) -> Dict[str, float]:
    """
    Measure per-generator fake detection rate (recall on synthetic-only subset).

    Uses GVB_FakesOnlyDataset (returns is_valid flag) so that corrupted videos
    are excluded from the denominator rather than counted as missed fakes.
    """
    from face_fft.eval.eval_genvidbench import GVB_FakesOnlyDataset

    model.eval()
    results: Dict[str, float] = {}
    total_caught = 0
    total_vids = 0

    rng = np.random.default_rng(GVB_EVAL_SEED)

    for gen in GVB_SYNTH_GENERATORS:
        gen_dir = os.path.join(GENVIDBENCH_ROOT, gen)
        if not os.path.exists(gen_dir):
            print(f"    GVB/{gen} not found, skipping.")
            results[f"gvb_{gen.lower()}_det"] = float("nan")
            continue

        fake_paths = [
            os.path.join(dp, f)
            for dp, _, files in os.walk(gen_dir)
            for f in files
            if f.endswith(".mp4")
        ]
        if not fake_paths:
            results[f"gvb_{gen.lower()}_det"] = float("nan")
            continue

        if len(fake_paths) > GVB_MAX_VIDS_PER_GEN:
            idx = rng.choice(len(fake_paths), GVB_MAX_VIDS_PER_GEN, replace=False)
            fake_paths = [fake_paths[i] for i in idx]

        ds = GVB_FakesOnlyDataset(fake_paths, TARGET_FRAMES, TARGET_SIZE)
        loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)

        caught = 0
        total = 0
        with torch.no_grad():
            for inputs, _, is_valid in loader:
                valid_mask = is_valid.bool()
                if not valid_mask.any():
                    continue
                inputs = inputs[valid_mask].to(device)
                logits = model(inputs).squeeze(-1)
                if logits.dim() == 0:
                    logits = logits.unsqueeze(0)
                caught += (logits > 0.0).float().sum().item()
                total += valid_mask.sum().item()

        rate = (caught / total) if total > 0 else float("nan")
        results[f"gvb_{gen.lower()}_det"] = rate
        total_caught += caught
        total_vids += total

    results["gvb_overall_det"] = (
        (total_caught / total_vids) if total_vids > 0 else float("nan")
    )
    return results


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

CSV_FIELDNAMES = [
    "pipeline_mode",
    "model_type",
    "num_params",
    "train_time_sec",
    "da_accuracy",
    "da_precision",
    "da_recall",
    "da_f1",
    "da_auc",
    "gvb_sora_det",
    "gvb_kling_det",
    "gvb_opensora_det",
    "gvb_overall_det",
]


def init_csv(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDNAMES).writeheader()


def append_csv(path: str, row: dict) -> None:
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
        writer.writerow(row)


def already_done(csv_path: str, mode: PipelineMode, model_type: str) -> bool:
    """Return True if this (mode, model_type) row already exists in the CSV."""
    if not os.path.exists(csv_path):
        return False
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["pipeline_mode"] == mode.value and row["model_type"] == model_type:
                return True
    return False


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-only",
        action="store_true",
        default=False,
        help="Skip training; evaluate existing checkpoints only.",
    )
    parser.add_argument(
        "--csv",
        default=CSV_PATH,
        help=f"Output CSV path (default: {CSV_PATH})",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    init_csv(args.csv)

    # Build datasets once — reused across all training runs
    if not args.eval_only:
        print("\nBuilding combined dataset...")
        train_ds, val_ds, synth_models = build_combined_datasets(
            DEEPACTION_ROOT, GENVIDBENCH_ROOT
        )
        print(f"Combined train: {len(train_ds)}, val: {len(val_ds)}\n")
    else:
        # For eval-only we still need synth_models for DeepAction eval
        all_folders = [
            d
            for d in os.listdir(DEEPACTION_ROOT)
            if os.path.isdir(os.path.join(DEEPACTION_ROOT, d))
        ]
        synth_models = [
            d for d in all_folders if d != "Pexels" and not d.startswith(".")
        ]
        train_ds = val_ds = None

    combos: List[Tuple[PipelineMode, str]] = list(product(PIPELINE_MODES, MODEL_TYPES))
    n_total = len(combos)
    print(
        f"Sweep: {n_total} combinations ({len(PIPELINE_MODES)} modes × {len(MODEL_TYPES)} archs)\n"
    )

    for run_idx, (mode, arch) in enumerate(combos, 1):
        tag = f"{mode.value}/{arch}"
        print(f"[{run_idx}/{n_total}] {tag}")
        print("-" * 60)

        ckpt = checkpoint_path(mode, arch)

        # CSV row is the source of truth — skip everything if it already exists.
        if already_done(args.csv, mode, arch):
            print("  Row already in CSV — skipping.\n")
            continue

        # --- Training ---
        train_time = float("nan")
        if not os.path.exists(ckpt):
            if args.eval_only:
                print(f"  Checkpoint not found: {ckpt}. Skipping.")
                continue
            print(f"  Training → {ckpt}")
            train_time = train_run(mode, arch, train_ds, val_ds, device)
            print(f"  Training complete in {train_time/60:.1f} min.")
        else:
            print("  Checkpoint found — skipping training.")

        # --- Load best checkpoint ---
        model = build_model(mode, arch)
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        n_params = count_parameters(model)

        # --- Evaluate DeepAction ---
        print("  Evaluating on DeepAction test set...")
        da_metrics = eval_deepaction(model, synth_models, device)
        print(
            f"  DA — Acc: {da_metrics['da_accuracy']*100:.1f}%  "
            f"F1: {da_metrics['da_f1']:.4f}  AUC: {da_metrics['da_auc']:.4f}"
        )

        # --- Evaluate GenVidBench ---
        print("  Evaluating on GenVidBench fakes...")
        gvb_metrics = eval_genvidbench(model, device)
        print(
            f"  GVB — Sora: {gvb_metrics.get('gvb_sora_det', float('nan'))*100:.1f}%  "
            f"Kling: {gvb_metrics.get('gvb_kling_det', float('nan'))*100:.1f}%  "
            f"OpenSora: {gvb_metrics.get('gvb_opensora_det', float('nan'))*100:.1f}%  "
            f"Overall: {gvb_metrics.get('gvb_overall_det', float('nan'))*100:.1f}%"
        )

        # --- Write row ---
        row = {
            "pipeline_mode": mode.value,
            "model_type": arch,
            "num_params": n_params,
            "train_time_sec": round(train_time, 1),
            **{
                k: round(v, 4) if not np.isnan(v) else "" for k, v in da_metrics.items()
            },
            **{
                k: round(v, 4) if not np.isnan(v) else ""
                for k, v in gvb_metrics.items()
            },
        }
        append_csv(args.csv, row)
        print(f"  Row written to {args.csv}\n")

    print("=" * 60)
    print(f"Sweep complete. Results: {args.csv}")
    _print_summary(args.csv)


def _print_summary(csv_path: str) -> None:
    """Print a formatted summary table from the completed CSV."""
    if not os.path.exists(csv_path):
        return
    rows = []
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return

    header = f"{'Mode':<22} {'Arch':<14} {'Params':>8} {'DA-F1':>7} {'DA-AUC':>7} {'GVB-Ovr':>8}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['pipeline_mode']:<22} {r['model_type']:<14} "
            f"{r['num_params']:>8} {r['da_f1']:>7} {r['da_auc']:>7} "
            f"{r['gvb_overall_det']:>8}"
        )


if __name__ == "__main__":
    main()
