"""Training / inference modes for FaceFFTPipeline (pixel vs FFT, optional learnable mask)."""

from __future__ import annotations

from enum import StrEnum


class PipelineMode(StrEnum):
    """How the pipeline preprocesses input and whether the classifier uses a learnable spectral mask."""

    PIXEL_BASELINE = "pixel_baseline"
    """Raw video tensors → CNN (no 3D-FFT, no spectral mask)."""

    FFT_NO_MASK = "fft_no_mask"
    """3D-FFT magnitude volume → CNN (identity spectral gate; matches arbitrary T,H,W at inference)."""

    FFT_LEARNABLE_MASK = "fft_learnable_mask"
    """3D-FFT → learnable per-voxel mask (parameter shape fixed by temporal_frames × spatial_size)."""

    FFT_1D_TEMPORAL = "fft_1d_temporal"
    """1D FFT over time only, per pixel -> same (C, T, H, W) shape. Ablation: is the
    temporal axis alone enough?"""

    FFT_2D_SPATIAL = "fft_2d_spatial"
    """2D FFT over (H, W) per frame -> same shape. Ablation: per-frame spatial spectrum
    with no temporal transform (the image-detector baseline)."""

    FFT_WHITENED = "fft_whitened"
    """3D-FFT log-magnitude with a fixed temporal whitening: each spatial frequency's
    temporal profile minus its running median. Removes the smooth fall-off that frame
    rate and motion set, keeps narrow periodic peaks. No learned parameters."""


def parse_pipeline_mode(value: str | PipelineMode) -> PipelineMode:
    if isinstance(value, PipelineMode):
        return value
    return PipelineMode(value)
