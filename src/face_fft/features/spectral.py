import torch
import torch.nn as nn


class SpatiotemporalFFT(nn.Module):
    """
    Applies a 3D Fast Fourier Transform over the spatiotemporal dimensions (T, H, W)
    of a video tensor to extract structured frequency representations.

    This expects the video tensor to retain its natural structure of Time, Height, Width,
    so that compression artifacts such as temporal downsampling or spatial patch tokenization
    create coherent and detectable periodic frequency signatures.
    """

    def __init__(self, log_scale: bool = True, epsilon: float = 1e-8, temporal_whiten: bool = False):
        """
        Args:
            log_scale: Whether to apply logarithmic scaling to emphasize weaker harmonics.
            epsilon: Small constant to avoid log(0).
            temporal_whiten: Subtract, at every spatial frequency, a running median of the
                log-magnitude along the temporal-frequency axis. Frame rate and motion set
                the smooth fall-off of that profile (a shortcut between sources); a latent
                temporal stride puts a narrow peak on it. A median of 5 (3 when T < 16)
                bins passes a single-bin peak almost unchanged and cancels smooth trends.
                Requires log_scale.
        """
        super().__init__()
        if temporal_whiten and not log_scale:
            raise ValueError("temporal_whiten operates on the log-magnitude; set log_scale=True")
        self.log_scale = log_scale
        self.epsilon = epsilon
        self.temporal_whiten = temporal_whiten

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transforms video tensor input to frequency domain.

        Args:
            x (torch.Tensor): Input video tensor of shape (B, C, T, H, W) or (C, T, H, W)

        Returns:
            torch.Tensor: Magnitude spectrum of shape corresponding to input,
                          zero-frequency centered.
        """
        if x.dim() not in (4, 5):
            raise ValueError(f"Expected input to be 4D or 5D, got {x.dim()}D")

        # The variables corresponding to T, H, W are the last 3 dimensions
        dims = (-3, -2, -1)

        # Apply 3D FFT over the spatiotemporal cube
        # We use standard fftn instead of rfftn to ensure full symmetric extraction
        # after fftshift, allowing interpretation of directional frequencies.
        freq_complex = torch.fft.fftn(x, dim=dims)

        # Shift zero frequency (DC component) to the center of the spectrum
        freq_shifted = torch.fft.fftshift(freq_complex, dim=dims)

        # Extract magnitude (phase is discarded for this pipeline formulation)
        magnitude = torch.abs(freq_shifted)

        # Apply optional log scaling
        if self.log_scale:
            magnitude = torch.log(magnitude + self.epsilon)

        if self.temporal_whiten:
            magnitude = magnitude - self._temporal_running_median(magnitude)

        return magnitude

    @staticmethod
    def _temporal_running_median(x: torch.Tensor) -> torch.Tensor:
        """Running median along dim -3 (temporal frequency), circular because the
        spectrum is periodic in frequency."""
        T = x.shape[-3]
        k = 5 if T >= 16 else 3
        p = k // 2
        xt = x.movedim(-3, -1)                                   # (..., H, W, T)
        padded = torch.cat([xt[..., -p:], xt, xt[..., :p]], dim=-1)
        med = padded.unfold(-1, k, 1).median(dim=-1).values      # (..., H, W, T)
        return med.movedim(-1, -3)
