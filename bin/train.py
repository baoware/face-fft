import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from face_fft.data.dataset import PairedVideoDataset, split_paired_dataset
from face_fft.data.deepaction import (
    DeepActionDataset,
    discover_deepaction_samples,
    split_deepaction_samples,
)
from face_fft.models.pipeline import FaceFFTPipeline
from face_fft.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train Face-FFT v1 (paired tensors or DeepAction v1)")
    parser.add_argument(
        "--real_dir",
        type=str,
        required=False,
        default=None,
        help="Path to real video tensors (.pt). Required unless --deepaction_dir is set.",
    )
    parser.add_argument(
        "--synth_dir",
        type=str,
        required=False,
        default=None,
        help="Path to synthetic video tensors (.pt). Required unless --deepaction_dir is set.",
    )

    # DeepAction (faridlab/deepaction_v1) support: unpaired real/synth videos.
    parser.add_argument(
        "--deepaction_dir",
        type=str,
        required=False,
        default=None,
        help=(
            "Path to a locally downloaded deepaction_v1 snapshot containing "
            "the top-level folders like `Pexels/` and synthetic model folders."
        ),
    )
    parser.add_argument(
        "--deepaction_hf_id",
        type=str,
        default="faridlab/deepaction_v1",
        help="Hugging Face dataset repo id to download (used with --deepaction_download).",
    )
    parser.add_argument(
        "--deepaction_download",
        action="store_true",
        default=False,
        help="If set, download the dataset into --deepaction_dir when missing.",
    )
    parser.add_argument(
        "--deepaction_real_folder",
        type=str,
        default="Pexels",
        help="Folder name for real videos inside deepaction_dir.",
    )
    parser.add_argument(
        "--deepaction_synth_folders",
        type=str,
        default=None,
        help=(
            "Comma-separated list of synthetic model folder names to include. "
            "If omitted, includes all non-real folders."
        ),
    )

    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--save_path", type=str, default="best_model.pt", help="Checkpoint save path"
    )

    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Val split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")

    # Video-to-tensor parameters (only used for DeepAction mode).
    parser.add_argument("--num_frames", type=int, default=16, help="Frames per clip")
    parser.add_argument(
        "--target_size", type=int, default=256, help="Center-crop resolution (square)"
    )
    parser.add_argument(
        "--max_duration_sec",
        type=float,
        default=2.0,
        help="Read only up to this many seconds per video (efficiency).",
    )
    parser.add_argument(
        "--skip_errors",
        action="store_true",
        default=False,
        help="Continue training if some videos fail to decode.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of discovered training samples (DeepAction mode).",
    )
    args = parser.parse_args()

    # 1. Load Data
    print("Loading datasets...")
    if args.deepaction_dir:
        if args.deepaction_download and not Path(args.deepaction_dir).exists():
            try:
                from huggingface_hub import snapshot_download
            except ImportError as e:
                raise ImportError(
                    "huggingface_hub is required for --deepaction_download. "
                    "Install it or download the dataset yourself and pass --deepaction_dir."
                ) from e

            # Download full snapshot to preserve directory layout for our folder-based labeling.
            snapshot_download(
                repo_id=args.deepaction_hf_id,
                repo_type="dataset",
                local_dir=args.deepaction_dir,
                local_dir_use_symlinks=False,
            )

        synth_folders = None
        if args.deepaction_synth_folders:
            synth_folders_raw = [
                s.strip() for s in args.deepaction_synth_folders.split(",") if s.strip()
            ]
            if len(synth_folders_raw) > 0 and synth_folders_raw[0].lower() != "all":
                synth_folders = synth_folders_raw

        samples = discover_deepaction_samples(
            args.deepaction_dir,
            real_folder_name=args.deepaction_real_folder,
            synth_folder_names=synth_folders,
            limit=args.limit,
        )
        train_samples, val_samples, _ = split_deepaction_samples(
            samples,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )

        train_set = DeepActionDataset(
            train_samples,
            target_size=(args.target_size, args.target_size),
            num_frames=args.num_frames,
            max_duration_sec=args.max_duration_sec,
            skip_errors=args.skip_errors,
        )
        val_set = DeepActionDataset(
            val_samples,
            target_size=(args.target_size, args.target_size),
            num_frames=args.num_frames,
            max_duration_sec=args.max_duration_sec,
            skip_errors=args.skip_errors,
        )
    else:
        if not args.real_dir or not args.synth_dir:
            raise ValueError(
                "Must provide either (real_dir & synth_dir) for paired .pt tensors, "
                "or --deepaction_dir for DeepAction video files."
            )

        # Load metadata explicitly without yielding pairs right away so we can split it safely.
        temp_dataset = PairedVideoDataset.from_directories(
            args.real_dir, args.synth_dir, yield_pairs=False
        )

        # Safely split without leakage
        train_pairs, val_pairs, _test_pairs = split_paired_dataset(
            temp_dataset.data_pairs,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )

        # We create the training and validation subsets focusing on single (vid, label) yields.
        train_set = PairedVideoDataset(train_pairs, yield_pairs=False)
        val_set = PairedVideoDataset(val_pairs, yield_pairs=False)

    print(f"Train size: {len(train_set)} | Val size: {len(val_set)}")

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # 2. Build Pipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    pipeline = FaceFFTPipeline()

    # 3. Setup Trainer
    trainer = Trainer(
        model=pipeline,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=args.lr,
    )

    # 4. Train Model
    print("Starting training...")
    trainer.train(num_epochs=args.epochs, save_path=args.save_path)

    print("Training complete!")


if __name__ == "__main__":
    main()
