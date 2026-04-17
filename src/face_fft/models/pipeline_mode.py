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


def parse_pipeline_mode(value: str | PipelineMode) -> PipelineMode:
    if isinstance(value, PipelineMode):
        return value
    return PipelineMode(value)
