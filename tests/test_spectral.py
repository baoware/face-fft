import pytest
import torch
from face_fft.features.spectral import SpatiotemporalFFT


def test_spatiotemporal_fft_shape():
    # Basic shape test (B, C, T, H, W)
    x = torch.randn(2, 3, 16, 64, 64)

    transform = SpatiotemporalFFT(log_scale=True)
    out = transform(x)

    assert out.shape == x.shape
    assert not out.is_complex()


def test_spatiotemporal_fft_4d_shape():
    # 4D tensor (C, T, H, W)
    x = torch.randn(3, 16, 64, 64)

    transform = SpatiotemporalFFT(log_scale=True)
    out = transform(x)

    assert out.shape == x.shape


def test_spatiotemporal_fft_log_scale():
    x = torch.ones(1, 1, 8, 32, 32)

    transform_log = SpatiotemporalFFT(log_scale=True)
    transform_linear = SpatiotemporalFFT(log_scale=False)

    out_log = transform_log(x)
    out_linear = transform_linear(x)

    assert torch.all(torch.isfinite(out_log))
    assert torch.all(torch.isfinite(out_linear))


def test_spatiotemporal_fft_dims_error():
    x_wrong = torch.randn(3, 3)  # Only 2D
    transform = SpatiotemporalFFT()

    with pytest.raises(ValueError):
        transform(x_wrong)
