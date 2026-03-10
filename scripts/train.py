import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from face_fft.data.dataset import PairedVideoDataset, split_paired_dataset
from face_fft.features.spectral import SpatiotemporalFFT
from face_fft.models.classifier import CompactSpectralCNN
from face_fft.training.trainer import Trainer


class FaceFFTPipeline(nn.Module):
    """
    Unified testing wrapper composing the feature extractor and the classifier.
    Moves processing directly onto the target device end-to-end.
    """

    def __init__(self):
        super().__init__()
        self.fft = SpatiotemporalFFT(log_scale=True)
        self.classifier = CompactSpectralCNN(
            in_channels=3, base_channels=16, num_classes=1
        )

    def forward(self, x):
        freq_vol = self.fft(x)
        logits = self.classifier(freq_vol)
        return logits


def main():
    parser = argparse.ArgumentParser(description="Train Face-FFT v1 on Paired Datasets")
    parser.add_argument(
        "--real_dir", type=str, required=True, help="Path to real video tensors"
    )
    parser.add_argument(
        "--synth_dir", type=str, required=True, help="Path to synthetic video tensors"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--save_path", type=str, default="best_model.pt", help="Checkpoint save path"
    )
    args = parser.parse_args()

    # 1. Load Data
    print("Loading datasets...")
    # Load metadata explicitly without yielding pairs right away so we can split it safely
    # This workaround creates a temporary dataset to parse directories
    temp_dataset = PairedVideoDataset.from_directories(
        args.real_dir, args.synth_dir, yield_pairs=False
    )

    # Safely split without leakage
    train_pairs, val_pairs, test_pairs = split_paired_dataset(
        temp_dataset.data_pairs, train_ratio=0.8, val_ratio=0.1
    )

    # We create the training and validation subsets focusing on single (vid, label) yields
    train_set = PairedVideoDataset(train_pairs, yield_pairs=False)
    val_set = PairedVideoDataset(val_pairs, yield_pairs=False)

    print(f"Train size: {len(train_set)} | Val size: {len(val_set)}")

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=2
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
