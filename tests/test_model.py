import torch
from face_fft.features.spectral import SpatiotemporalFFT
from face_fft.models.classifier import CompactSpectralCNN
from face_fft.models.pipeline import FaceFFTPipeline
from face_fft.models.pipeline_mode import PipelineMode


def test_classifier_forward_pass():
    # Batch size 2, 3 channels, 16 frames, 64x64 — no learnable mask so T/H/W are unconstrained
    x = torch.randn(2, 3, 16, 64, 64)
    model = CompactSpectralCNN(
        in_channels=3,
        base_channels=8,
        num_classes=1,
        use_learnable_mask=False,
    )

    logits = model(x)
    assert logits.shape == (2, 1)


def test_learnable_mask_matches_volume_shape():
    x = torch.randn(2, 3, 16, 64, 64)
    model = CompactSpectralCNN(
        in_channels=3,
        base_channels=8,
        num_classes=1,
        use_learnable_mask=True,
        temporal_frames=16,
        spatial_size=(64, 64),
    )
    logits = model(x)
    assert logits.shape == (2, 1)


def test_integration_fft_and_classifier():
    x_vid = torch.randn(4, 3, 16, 128, 128)

    fft_transform = SpatiotemporalFFT(log_scale=True)

    model = CompactSpectralCNN(
        in_channels=3,
        base_channels=8,
        num_classes=1,
        use_learnable_mask=False,
    )

    with torch.no_grad():
        x_freq = fft_transform(x_vid)
        logits = model(x_freq)

    assert x_freq.shape == x_vid.shape
    assert not x_freq.is_complex()
    assert logits.shape == (4, 1)


def test_face_fft_pipeline_modes():
    x = torch.randn(2, 3, 8, 32, 32)
    for mode in (
        PipelineMode.PIXEL_BASELINE,
        PipelineMode.FFT_NO_MASK,
        PipelineMode.FFT_LEARNABLE_MASK,
    ):
        m = FaceFFTPipeline(
            mode=mode,
            temporal_frames=8,
            spatial_size=(32, 32),
            model_type="compact",
            base_channels=8,
        )
        y = m(x)
        assert y.shape == (2, 1)
