import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights


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

class SpectralResNet3D(nn.Module):
    """
    A deeper 3D ResNet classifier designed to detect faint, 
    multi-scale upsampling artifacts in the 3D-FFT domain.
    """
    def __init__(
        self, 
        in_channels: int = 3, 
        num_classes: int = 1, 
        pretrained: bool = False
    ):
        super().__init__()
        
        # load the base 3D ResNet-18 architecture
        weights = R3D_18_Weights.DEFAULT if pretrained else None
        self.backbone = r3d_18(weights=weights)
        
        # modify the input layer if we aren't using 3 channels
        if in_channels != 3:
            original_conv = self.backbone.stem[0]
            self.backbone.stem[0] = nn.Conv3d(
                in_channels, 
                original_conv.out_channels, 
                kernel_size=original_conv.kernel_size, 
                stride=original_conv.stride, 
                padding=original_conv.padding, 
                bias=False
            )
            
        # replace the final fully connected layer for binary classification
        # add heavy dropout (0.5) because 3D ResNets have 33M+ parameters 
        # and are prone to overfitting on smaller datasets
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Output from SpatiotemporalFFT, shape (B, C, T, H, W)
        Returns:
            Logit predictions, shape (B, 1)
        """
        return self.backbone(x)