import torch
from face_fft.features.spectral import SpatiotemporalFFT
from face_fft.models.classifier import CompactSpectralCNN


def test_classifier_forward_pass():
    # Batch size 2, 3 channels, 16 frames, 64x64
    x = torch.randn(2, 3, 16, 64, 64)
    model = CompactSpectralCNN(in_channels=3, base_channels=8, num_classes=1)

    # Model should accept 5D tensor and output (B, 1) logits
    logits = model(x)
    assert logits.shape == (2, 1)


def test_integration_fft_and_classifier():
    # Input video tensor
    x_vid = torch.randn(4, 3, 16, 128, 128)

    # Preprocessing Module
    fft_transform = SpatiotemporalFFT(log_scale=True)

    # Classification Module
    model = CompactSpectralCNN(in_channels=3, base_channels=8, num_classes=1)

    # Simulation Pipeline
    with torch.no_grad():
        x_freq = fft_transform(x_vid)
        logits = model(x_freq)

    assert x_freq.shape == x_vid.shape
    assert not x_freq.is_complex()
    assert logits.shape == (4, 1)
