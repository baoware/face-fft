import torch.nn as nn

from face_fft.features.spectral import SpatiotemporalFFT
from face_fft.models.classifier import CompactSpectralCNN, SpectralVideoCNN
from face_fft.models.pipeline_mode import PipelineMode, parse_pipeline_mode


class FaceFFTPipeline(nn.Module):
    """
    End-to-end Face-FFT detection pipeline.

    Composes optional SpatiotemporalFFT (feature extraction) and a 3D classifier.
    Use ``mode`` to switch pixel baseline, FFT without learnable mask, or FFT with
    learnable mask (see ``PipelineMode``).
    """

    def __init__(
        self,
        log_scale: bool = True,
        in_channels: int = 3,
        base_channels: int = 16,
        num_classes: int = 1,
        model_type: str = "compact",
        mode: str | PipelineMode = PipelineMode.FFT_LEARNABLE_MASK,
        temporal_frames: int = 8,
        spatial_size: tuple[int, int] = (256, 256),
    ):
        super().__init__()
        self.mode = parse_pipeline_mode(mode)
        self.use_fft = self.mode != PipelineMode.PIXEL_BASELINE
        use_learnable_mask = self.mode == PipelineMode.FFT_LEARNABLE_MASK

        self.fft = SpatiotemporalFFT(log_scale=log_scale) if self.use_fft else None

        clf_kw = dict(
            in_channels=in_channels,
            num_classes=num_classes,
            use_learnable_mask=use_learnable_mask,
            temporal_frames=temporal_frames,
            spatial_size=spatial_size,
        )

        if model_type == "compact":
            self.classifier = CompactSpectralCNN(
                base_channels=base_channels,
                **clf_kw,
            )
        else:
            self.classifier = SpectralVideoCNN(
                model_name=model_type,
                **clf_kw,
            )

    def forward(self, x):
        if self.use_fft:
            assert self.fft is not None
            x = self.fft(x)
        return self.classifier(x)
