import torch
import torch.nn as nn


class CompactSpectralCNN(nn.Module):
    """
    A lightweight 3D CNN classifier designed to detect structured spatiotemporal
    frequency artifacts in Face-FFT frequency volumes.

    This architecture explicitly avoids excessive depth or large pretrained
    transformer blocks to emphasize interpretability and lightweight artifact detection
    as requested for the HPC environment.
    """

    def __init__(
        self, in_channels: int = 3, base_channels: int = 16, num_classes: int = 1
    ):
        """
        Args:
            in_channels: Number of channels in the input volume (usually 3 for RGB spectral magnitudes).
            base_channels: Number of base feature planes. Kept small for lightweight architecture.
            num_classes: Number of output dimension. 1 for simple binary BCE loss.
        """
        super().__init__()

        # Lightweight 3D Volumetric Feature Extractor
        self.features = nn.Sequential(
            self._conv_block(in_channels, base_channels),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            self._conv_block(base_channels, base_channels * 2),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            self._conv_block(base_channels * 2, base_channels * 4),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            self._conv_block(base_channels * 4, base_channels * 8),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(base_channels * 8, base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(base_channels * 4, num_classes),
        )

    def _conv_block(self, in_c, out_c):
        """Standard 3D Convolution -> BatchNorm -> ReLU bottleneck"""
        return nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Output from SpatiotemporalFFT, shape (B, C, T, H, W).

        Returns:
            torch.Tensor: Logit predictions, shape (B, num_classes)
        """
        feats = self.features(x)
        logits = self.classifier(feats)
        return logits
