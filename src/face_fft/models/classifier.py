import torch
import torch.nn as nn
from torchvision.models.video import (r3d_18, mc3_18, r2plus1d_18, R3D_18_Weights,
                                      MC3_18_Weights, R2Plus1D_18_Weights)

# Kinetics-400 input statistics the pretrained torchvision video models expect
KINETICS_MEAN = (0.43216, 0.394666, 0.37645)
KINETICS_STD = (0.22803, 0.22145, 0.216989)


class LearnableSpectralMask(nn.Module):
    """
    A learnable multiplicative gate over the 3D-FFT spectrum (same spatial shape as the input volume).
    """

    def __init__(self, T: int, H: int, W: int):
        super().__init__()
        self.mask = nn.Parameter(torch.ones(1, 1, T, H, W))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, T, H, W); mask broadcasts over B and C
        return x * torch.sigmoid(self.mask)


class CompactSpectralCNN(nn.Module):
    """
    A lightweight 3D CNN classifier designed to detect structured spatiotemporal
    frequency artifacts in Face-FFT frequency volumes (or raw video for ablations).

    This architecture explicitly avoids excessive depth or large pretrained
    transformer blocks to emphasize interpretability and lightweight artifact detection
    as requested for the HPC environment.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 16,
        num_classes: int = 1,
        *,
        use_learnable_mask: bool = True,
        temporal_frames: int = 8,
        spatial_size: tuple[int, int] = (256, 256),
    ):
        """
        Args:
            in_channels: Channels in the input volume (typically 3 for RGB).
            base_channels: Base feature width for the 3D CNN.
            num_classes: Output dimension (1 for binary BCE).
            use_learnable_mask: If True, apply LearnableSpectralMask with shape
                (temporal_frames, spatial_size[0], spatial_size[1]). If False, identity.
            temporal_frames: T dimension for the learnable mask (must match input T when mask is used).
            spatial_size: (H, W) for the learnable mask (must match input when mask is used).
        """
        super().__init__()

        h, w = spatial_size
        self.spectral_filter: nn.Module
        if use_learnable_mask:
            self.spectral_filter = LearnableSpectralMask(T=temporal_frames, H=h, W=w)
        else:
            self.spectral_filter = nn.Identity()

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

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(base_channels * 8, base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(base_channels * 4, num_classes),
        )

    def _conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_filtered = self.spectral_filter(x)
        feats = self.features(x_filtered)
        return self.classifier(feats)


class SpectralVideoCNN(nn.Module):
    """
    Torchvision 3D video backbones adapted for spectral volumes or raw video (same tensor shape).
    Supports 'r3d_18', 'mc3_18', and 'r2plus1d_18'.
    """

    def __init__(
        self,
        model_name: str = "r3d_18",
        in_channels: int = 3,
        num_classes: int = 1,
        *,
        use_learnable_mask: bool = True,
        temporal_frames: int = 8,
        spatial_size: tuple[int, int] = (256, 256),
        pretrained: bool = False,
        kinetics_norm: bool = False,
    ):
        """
        pretrained: load Kinetics-400 weights (downloads once) and normalise inputs
            with Kinetics statistics. Meaningful for raw-pixel input only.
        kinetics_norm: apply that normalisation without downloading weights -- used
            when rebuilding a pretrained-then-finetuned model to load its checkpoint.
        """
        super().__init__()

        h, w = spatial_size
        if use_learnable_mask:
            self.spectral_filter = LearnableSpectralMask(T=temporal_frames, H=h, W=w)
        else:
            self.spectral_filter = nn.Identity()

        builders = {"r3d_18": (r3d_18, R3D_18_Weights), "mc3_18": (mc3_18, MC3_18_Weights),
                    "r2plus1d_18": (r2plus1d_18, R2Plus1D_18_Weights)}
        if model_name not in builders:
            raise ValueError(f"Unsupported model: {model_name}")
        build, weights = builders[model_name]
        self.backbone = build(weights=weights.KINETICS400_V1 if pretrained else None)

        self.kinetics_norm = pretrained or kinetics_norm
        self.register_buffer("in_mean", torch.tensor(KINETICS_MEAN).view(1, 3, 1, 1, 1), persistent=False)
        self.register_buffer("in_std", torch.tensor(KINETICS_STD).view(1, 3, 1, 1, 1), persistent=False)

        if in_channels != 3:
            original_conv = self.backbone.stem[0]
            self.backbone.stem[0] = nn.Conv3d(
                in_channels,
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=False,
            )

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_filtered = self.spectral_filter(x)
        if self.kinetics_norm:
            x_filtered = (x_filtered - self.in_mean) / self.in_std
        return self.backbone(x_filtered)
