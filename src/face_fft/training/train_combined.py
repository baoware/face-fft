"""
Combined training on DeepAction + GenVidBench datasets.

This script builds a mixed training set by concatenating:
  - DeepAction v1 (Pexels real videos + multiple generative model fakes)
  - GenVidBench   (Real/ folder + Sora/Kling/OpenSora fakes)

If the GenVidBench Real/ folder is absent, the script falls back to reusing
the DeepAction Pexels real videos as the "real" source for GenVidBench fakes.
This is valid for binary real/fake detection: the real-video distribution from
Pexels is representative enough to contrast against GenVidBench synthetics.

Ablation notes (edit PIPELINE_MODE / MODEL_TYPE below):
  pixel_baseline      -- raw pixels -> CNN (no FFT, no mask)
  fft_no_mask         -- 3D-FFT -> CNN (no learnable gate)
  fft_learnable_mask  -- 3D-FFT -> learnable per-voxel gate -> CNN  [default]

  model types: compact | r3d_18 | mc3_18 | r2plus1d_18
"""

import os
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader

from face_fft.data.deepaction import get_deepaction_splits
from face_fft.data.genvidbench import get_genvidbench_splits, collect_mp4s
from face_fft.models.pipeline import FaceFFTPipeline
from face_fft.models.pipeline_mode import PipelineMode
from face_fft.training.trainer import Trainer


# ---------------------------------------------------------------------------
# Configuration — edit these for your run / ablation study
# ---------------------------------------------------------------------------

DEEPACTION_ROOT = "/scratch/rjr6zk/face-fft/src/face_fft/data/deepaction_dataset"
GENVIDBENCH_ROOT = "/scratch/rjr6zk/face-fft/src/face_fft/data/genvidbench_dataset"

PIPELINE_MODE = PipelineMode.FFT_LEARNABLE_MASK  # change for ablation
MODEL_TYPE = "r3d_18"  # change for ablation

CHECKPOINT_DIR = "checkpoints/combined"
CHECKPOINT_NAME = f"best_combined_{MODEL_TYPE}_{PIPELINE_MODE.value}.pt"

BATCH_SIZE = 8
EPOCHS = 25
LEARNING_RATE = 1e-3
TARGET_FRAMES = 8
TARGET_SIZE = (256, 256)
NUM_WORKERS = 4
SEED = 42

GVB_SYNTH_GENERATORS = ["Sora", "Kling", "OpenSora"]

# ---------------------------------------------------------------------------


def _collect_deepaction_real_paths(deepaction_root: str) -> list:
    """Collect all Pexels real .mp4 paths from DeepAction for fallback use."""
    pexels_dir = Path(deepaction_root) / "Pexels"
    if not pexels_dir.exists():
        return []
    return collect_mp4s(str(pexels_dir))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Pipeline mode: {PIPELINE_MODE.value}")
    print(f"Model type:    {MODEL_TYPE}")
    print()

    # ------------------------------------------------------------------
    # 1. DeepAction splits
    # ------------------------------------------------------------------
    print("Loading DeepAction dataset...")
    all_folders = [
        d
        for d in os.listdir(DEEPACTION_ROOT)
        if os.path.isdir(os.path.join(DEEPACTION_ROOT, d))
    ]
    synth_models = [d for d in all_folders if d != "Pexels" and not d.startswith(".")]
    print(f"  Found {len(synth_models)} DeepAction synthetic model(s): {synth_models}")

    da_train, da_val, _ = get_deepaction_splits(
        root_dir=DEEPACTION_ROOT,
        synth_models=synth_models,
        train_ratio=0.8,
        val_ratio=0.1,
        target_frames=TARGET_FRAMES,
        target_size=TARGET_SIZE,
        seed=SEED,
    )

    # ------------------------------------------------------------------
    # 2. GenVidBench splits
    #    Try Real/ folder first; fall back to DeepAction Pexels videos.
    # ------------------------------------------------------------------
    print("\nLoading GenVidBench dataset...")
    gvb_real_dir = Path(GENVIDBENCH_ROOT) / "Real"
    real_override = None
    # Use SEED+1 in fallback mode so the GVB sample draws a different shuffle
    # from the Pexels list, reducing the chance of duplicating the exact same
    # videos that get_deepaction_splits() already put in training.
    gvb_seed = SEED
    if not gvb_real_dir.exists():
        print(
            f"  GenVidBench Real/ folder not found at {gvb_real_dir}.\n"
            "  Falling back to DeepAction Pexels videos as real source.\n"
            "  (Run scripts/download_genvidbench_full.slurm to fetch GVB reals.)"
        )
        real_override = _collect_deepaction_real_paths(DEEPACTION_ROOT)
        if not real_override:
            raise FileNotFoundError(
                "No real videos found in DeepAction Pexels either. "
                "Check DEEPACTION_ROOT and GENVIDBENCH_ROOT paths."
            )
        gvb_seed = SEED + 1

    gvb_train, gvb_val, _ = get_genvidbench_splits(
        root_dir=GENVIDBENCH_ROOT,
        synth_generators=GVB_SYNTH_GENERATORS,
        real_folder="Real",
        real_paths_override=real_override,
        train_ratio=0.8,
        val_ratio=0.1,
        target_frames=TARGET_FRAMES,
        target_size=TARGET_SIZE,
        seed=gvb_seed,
    )

    # ------------------------------------------------------------------
    # 3. Combine datasets
    # ------------------------------------------------------------------
    train_dataset = ConcatDataset([da_train, gvb_train])
    val_dataset = ConcatDataset([da_val, gvb_val])

    print(f"\nCombined — Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # ------------------------------------------------------------------
    # 4. Model
    # ------------------------------------------------------------------
    print("\nInitializing FaceFFTPipeline...")
    model = FaceFFTPipeline(
        log_scale=True,
        in_channels=3,
        base_channels=16,
        num_classes=1,
        model_type=MODEL_TYPE,
        mode=PIPELINE_MODE,
        temporal_frames=TARGET_FRAMES,
        spatial_size=TARGET_SIZE,
    )

    # ------------------------------------------------------------------
    # 5. Train
    # ------------------------------------------------------------------
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    save_path = os.path.join(CHECKPOINT_DIR, CHECKPOINT_NAME)
    print(f"\nStarting training — checkpoint: {save_path}")
    trainer.train(num_epochs=EPOCHS, save_path=save_path)
    print("Done.")


if __name__ == "__main__":
    main()
