import argparse
import torch
from torch.utils.data import DataLoader

from face_fft.data.dataset import PairedVideoDataset, split_paired_dataset
from face_fft.models.pipeline import FaceFFTPipeline
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

    pipeline = FaceFFTPipeline()
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
