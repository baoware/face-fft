import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, mc3_18, r2plus1d_18

class LearnableSpectralMask(nn.Module):
    """
    A dynamic, learnable frequency filter. 
    It acts as a self-attention mechanism over the 3D-FFT spectrum, 
    learning to dynamically suppress the DC center (natural physics) 
    and highlight the generative grid harmonics.
    """
    def __init__(self, T=8, H=256, W=256):
        super().__init__()
        # Initialize a tensor of ones (letting all frequencies pass initially)
        # Shape: (1, 1, T, H, W) so it broadcasts across batches and RGB channels
        self.mask = nn.Parameter(torch.ones(1, 1, T, H, W))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use a sigmoid to ensure the mask acts as a strict filter (values 0.0 to 1.0)
        # x shape: (B, C, T, H, W)
        filtered_spectrum = x * torch.sigmoid(self.mask)
        return filtered_spectrum

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

        # self.spectral_filter = LearnableSpectralMask(T=8, H=256, W=256)

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
        # x_filtered = self.spectral_filter(x)

        # feats = self.features(x_filtered)
        feats = self.features(x)
        logits = self.classifier(feats)
        return logits
        
class SpectralVideoCNN(nn.Module):
    """
    A unified wrapper for torchvision 3D models adapted for 3D-FFT Spectral classification.
    Supports 'r3d_18', 'mc3_18', and 'r2plus1d_18'.
    """
    def __init__(
        self, 
        model_name: str = "r3d_18",
        in_channels: int = 3, 
        num_classes: int = 1
    ):
        super().__init__()

        self.spectral_filter = LearnableSpectralMask(T=8, H=256, W=256)
        
        # Load the requested backbone (untrained, since FFT != RGB)
        if model_name == "r3d_18":
            self.backbone = r3d_18(weights=None)
        elif model_name == "mc3_18":
            self.backbone = mc3_18(weights=None)
        elif model_name == "r2plus1d_18":
            self.backbone = r2plus1d_18(weights=None)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
            
        # Patch the input channel depth if not 3
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
            
        # Patch the classification head for Binary Classification + Dropout
        # Use aggressive dropout to combat overfitting on small datasets
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x_filtered = self.spectral_filter(x)
        # return self.backbone(x_filtered)

        return self.backbone(x)