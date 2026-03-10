import torch.nn as nn

from face_fft.features.spectral import SpatiotemporalFFT
from face_fft.models.classifier import CompactSpectralCNN


class FaceFFTPipeline(nn.Module):
    """
    End-to-end Face-FFT detection pipeline.

    Composes SpatiotemporalFFT (feature extraction) and CompactSpectralCNN
    (binary classifier) into a single nn.Module for training and inference.

    SpatiotemporalFFT has no learnable parameters, so state_dict() contains
    only the classifier weights — save/load behavior is identical to saving
    the classifier alone.
    """

    def __init__(
        self,
        log_scale: bool = True,
        in_channels: int = 3,
        base_channels: int = 16,
        num_classes: int = 1,
    ):
        super().__init__()
        self.fft = SpatiotemporalFFT(log_scale=log_scale)
        self.classifier = CompactSpectralCNN(
            in_channels=in_channels,
            base_channels=base_channels,
            num_classes=num_classes,
        )

    def forward(self, x):
        return self.classifier(self.fft(x))
