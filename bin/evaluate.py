import argparse
import torch
from torch.utils.data import DataLoader

from face_fft.data.dataset import PairedVideoDataset, split_paired_dataset
from face_fft.models.pipeline import FaceFFTPipeline
from face_fft.models.pipeline_mode import PipelineMode
from face_fft.eval.evaluator import Evaluator


def main():
    parser = argparse.ArgumentParser(description="Evaluate Face-FFT generalization")
    parser.add_argument(
        "--real_dir", type=str, required=True, help="Path to real video tensors"
    )
    parser.add_argument(
        "--synth_dir", type=str, required=True, help="Path to synthetic video tensors"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained checkpoint"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--pipeline_mode",
        type=str,
        default=PipelineMode.FFT_LEARNABLE_MASK.value,
        choices=[m.value for m in PipelineMode],
        help="Must match the checkpoint (see train script).",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="compact",
        help="Must match training: compact | r3d_18 | mc3_18 | r2plus1d_18",
    )
    parser.add_argument(
        "--temporal_frames",
        type=int,
        default=8,
        help="T in (C,T,H,W); must match training for learnable-mask checkpoints.",
    )
    parser.add_argument(
        "--target_size",
        type=int,
        default=256,
        help="Square spatial size; must match training for learnable-mask checkpoints.",
    )
    args = parser.parse_args()

    print("Loading test datasets...")
    # Normally we load the full dataset, then extract the same test split to prevent leakage
    # For a real pipeline, the test splits should be statically saved or use fixed seed
    temp_dataset = PairedVideoDataset.from_directories(
        args.real_dir, args.synth_dir, yield_pairs=False
    )

    # Deterministic split via seed 42 to ensure we evaluate on the correct test partition
    _, _, test_pairs = split_paired_dataset(
        temp_dataset.data_pairs, train_ratio=0.8, val_ratio=0.1
    )

    test_set = PairedVideoDataset(test_pairs, yield_pairs=False)
    print(f"Test size: {len(test_set)}")

    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    pipeline = FaceFFTPipeline(
        mode=args.pipeline_mode,
        model_type=args.model_type,
        temporal_frames=args.temporal_frames,
        spatial_size=(args.target_size, args.target_size),
    )
    pipeline.load_state_dict(
        torch.load(args.model_path, weights_only=True, map_location=device)
    )

    evaluator = Evaluator(model=pipeline, device=device)

    print("Starting evaluation...")
    metrics = evaluator.evaluate(test_loader)

    print("\n======== Results ========")
    print(f"F1 Score: {metrics['f1_score']:.4f}")

    cm = metrics["confusion_matrix"]
    print("\nConfusion Matrix:")
    print("             Pred Real | Pred Synth")
    print(f"Actual Real       {cm[0][0]:<8} | {cm[0][1]:<8}")
    print(f"Actual Synth      {cm[1][0]:<8} | {cm[1][1]:<8}")
    print("=========================\n")


if __name__ == "__main__":
    main()
