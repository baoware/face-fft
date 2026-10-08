"""Measurements for the VAE round-trip test (bin/vae_roundtrip.py).

A round-trip is a real clip pushed through a generator's VAE: encoder, then decoder.
Content, motion and source are identical in the original and the copy; the only change
is that the decoder touched the copy. Every measurement here is therefore reported as
*copy minus original on the same clip*, so that content cancels out.

Nothing in this file assumes a particular period. Curves are indexed by FFT bin and
the caller reads off whichever bin matches a VAE's stride:

    temporal bin k of a T-frame clip    <-> a repeat every T / k frames
    spatial  bin k of an N-pixel frame  <-> a repeat every N / k pixels

All functions take luma volumes of shape (T, H, W) in uint8 range (see `luma`). The
FFT is taken over the whole volume without a window: the clip length is chosen so that
each VAE's stride falls exactly on one bin, and a window would smear that bin into the
neighbours the prominence baseline is taken from.
"""

import warnings

import numpy as np


def luma(frames: np.ndarray) -> np.ndarray:
    """(T, H, W, 3) uint8 RGB -> (T, H, W) float32. Channel mean, as in lossless_test."""
    return frames.astype(np.float32).mean(axis=-1)


def psnr_db(orig: np.ndarray, copy: np.ndarray) -> float:
    mse = float(np.mean((copy.astype(np.float64) - orig.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else float(10.0 * np.log10(255.0**2 / mse))


# ------------------------------------------------------------------------ prominence
def prominence_curve_db(P: np.ndarray, half_window: int = 5) -> np.ndarray:
    """Prominence of every bin over a local-median baseline, in dB.

    Same rule as codec_control.peak_prominence_db (the lossless test's measure), applied
    at every bin instead of one chosen period: the baseline is the median of the bins
    within +-half_window, excluding the bin itself and its two shoulders. A local
    baseline is needed because video spectra fall off as ~1/f.

    Args:
        P: full power spectrum along one axis, length n (bin k = k cycles per n samples).

    Returns:
        (n // 2 + 1,) array; NaN where the rule is undefined (DC, bin 1, Nyquist).
    """
    n = len(P)
    out = np.full(n // 2 + 1, np.nan)
    for k0 in range(2, n // 2):
        lo, hi = max(1, k0 - half_window), min(n // 2, k0 + half_window + 1)
        neighbourhood = [k for k in range(lo, hi) if abs(k - k0) > 1]
        if len(neighbourhood) < 3:
            continue
        baseline = np.median(P[neighbourhood])
        if baseline > 0 and P[k0] > 0:
            out[k0] = 10.0 * np.log10(P[k0] / baseline)
    return out


def neighbour_median(curve: np.ndarray, k0: int, half_window: int = 5) -> float:
    """Median of the bins around `k0` in a one-sided curve: what bin k0 should be read against.

    Same neighbourhood as `prominence_curve_db` (within +-half_window, excluding k0 and
    its two shoulders), applied to a curve that has already been averaged over clips.

    Reading a bin against 0 is not enough. Anything tied to position in the clip -- a
    worse last frame, a slow drift -- has the same phase in every clip and lifts EVERY
    bin, and a decoder that merely smooths in time bends the spectrum and shifts every
    prominence. A k-frame repeat is the one thing that lifts bin T/k and not the bins
    beside it, so a mark is a value that stands out from this median.
    """
    n_half = len(curve) - 1
    lo, hi = max(1, k0 - half_window), min(n_half, k0 + half_window + 1)
    values = np.asarray(
        [curve[k] for k in range(lo, hi) if abs(k - k0) > 1], dtype=np.float64
    )
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) >= 3 else float("nan")


# -------------------------------------------------------------- 1: per-frame error
def frame_error(orig: np.ndarray, copy: np.ndarray) -> np.ndarray:
    """Mean squared error of each frame of the copy against the original. (T,)"""
    d = copy.astype(np.float64) - orig.astype(np.float64)
    return (d**2).mean(axis=(1, 2))


def cycle_coefficients(e: np.ndarray) -> np.ndarray:
    """Complex amplitude of each repeat spacing in a per-frame curve, relative to its mean.

    |coef[k]| = 0.10 at bin T/4 means the curve swings +-10% around its mean every 4
    frames. Being relative, it compares across VAEs with very different overall error.

    The phase is kept on purpose. A decoder that works in groups of k frames puts its
    cycle at the SAME phase in every clip, while content-driven wobble (a burst of
    motion) lands at a random phase. Averaging these complex values over clips therefore
    keeps the decoder's cycle and cancels the content, and the average is 0 under the
    null -- unlike an average of magnitudes, which is positive for any noisy curve.

    Returns:
        (T // 2 + 1,) complex array; coef[0] is 0 by definition.
    """
    T = len(e)
    mean = float(np.mean(e))
    if mean == 0:
        return np.zeros(T // 2 + 1, dtype=np.complex128)
    coef = np.fft.rfft(e - mean) / T
    coef[1:] *= 2.0  # one-sided spectrum: fold in the mirrored negative frequency
    if T % 2 == 0:
        coef[-1] /= 2.0  # Nyquist has no mirror
    return coef / mean


def modulation_depth(e: np.ndarray) -> np.ndarray:
    """Magnitude of `cycle_coefficients`: cycle strength of ONE clip at each spacing."""
    return np.abs(cycle_coefficients(e))


# ----------------------------------------------------------------- 3D FFT marginals
def power_volume(x: np.ndarray) -> np.ndarray:
    """3D FFT power of a (T, H, W) volume, mean removed. Unshifted (DC at [0, 0, 0])."""
    x = x.astype(np.float32)
    F = np.fft.fftn(x - x.mean())
    return F.real**2 + F.imag**2


def temporal_marginal(P: np.ndarray) -> np.ndarray:
    """Power summed over spatial frequency: one value per temporal bin. (T,)

    Identical to lossless_test.spectral_stat, so the two tests compare directly.
    """
    return P.sum(axis=(1, 2))


def spatial_marginals(P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Power per vertical bin (H,) and per horizontal bin (W,), summed over the rest."""
    return P.sum(axis=(0, 2)), P.sum(axis=(0, 1))


def time_radius_map(P: np.ndarray) -> np.ndarray:
    """Power per (temporal bin, spatial radius). (T // 2 + 1, min(H, W) // 2 + 1)

    Spatial frequencies are pooled into integer-radius rings so the map stays small
    enough to keep per clip; frequencies beyond the Nyquist ring (the corners) are
    dropped. Temporal bins k and T - k are folded together.
    """
    T, H, W = P.shape
    n_r = min(H, W) // 2 + 1
    ky = np.fft.fftfreq(H) * H
    kx = np.fft.fftfreq(W) * W
    r = np.rint(np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)).astype(int).ravel()
    keep = r < n_r
    rings = np.stack(
        [
            np.bincount(r[keep], weights=P[t].ravel()[keep], minlength=n_r)
            for t in range(T)
        ]
    )
    out = rings[: T // 2 + 1].copy()
    out[1 : (T + 1) // 2] += rings[:0:-1][: (T + 1) // 2 - 1]  # fold k with T - k
    return out


def space_map(P: np.ndarray) -> np.ndarray:
    """Power per spatial frequency, summed over time. (H, W), fftshifted (DC centred)."""
    return np.fft.fftshift(P.sum(axis=0))


def _ratio_db(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """10 log10(num / den), with 0 where both are 0 (nothing there before or after)."""
    out = np.zeros(num.shape, dtype=np.float64)
    ok = (num > 0) & (den > 0)
    out[ok] = 10.0 * np.log10(num[ok] / den[ok])
    out[(num > 0) & (den <= 0)] = np.inf
    out[(num <= 0) & (den > 0)] = -np.inf
    return out


# ------------------------------------------------------------------------ per clip
def clip_metrics(orig: np.ndarray, copy: np.ndarray) -> dict:
    """Every step-1 measurement for one (original, round-trip) pair of luma volumes.

    Both are (T, H, W), with the VAE's lone first frame already dropped by the caller.

    Returns a dict of:
        psnr_db            overall closeness; a sanity check that the VAE ran correctly
        frame_error        (T,) MSE of each frame                              [metric 1]
        error_coef         (T/2+1,) complex cycle coefficients of frame_error  [metric 1]
        time_spike_db      (T/2+1,) temporal prominence, copy minus original   [metric 2]
        space_spike_db     (N/2+1,) spatial prominence, copy minus original,
                           averaged over the vertical and horizontal axes      [metric 3]
        resid_time_spike_db / resid_space_spike_db
                           the same prominences measured on the residual
                           (copy - original) alone. More sensitive, because the
                           content is subtracted before the FFT rather than after,
                           but not zero under the null: read against neighbouring bins.
        orig_time_spike_db / orig_space_spike_db
                           the ORIGINAL's own prominences, kept so a cycle that was
                           already in the source (its codec's frame pattern) is visible.

    Only time_spike_db and space_spike_db are properties of the copy that a generated
    video could also carry. frame_error, error_coef and the residual spikes need the
    original, and reflect what the encoder lost as well as what the decoder added.
        time_map_db        (T/2+1, N/2+1) dB change per temporal bin x radius  [metric 4]
        space_map_db       (H, W) dB change per spatial frequency              [metric 4]
    """
    if orig.shape != copy.shape:
        raise ValueError(f"shape mismatch: original {orig.shape}, copy {copy.shape}")
    T, H, W = orig.shape
    if H != W:
        raise ValueError(f"square frames expected, got {H}x{W}")

    e = frame_error(orig, copy)
    Po, Pc = power_volume(orig), power_volume(copy)

    def spikes(P):
        py, px = spatial_marginals(P)
        return (
            prominence_curve_db(temporal_marginal(P)),
            (prominence_curve_db(py) + prominence_curve_db(px)) / 2,
        )

    to, so = spikes(Po)
    tc, sc = spikes(Pc)
    if e.any():
        tr, sr = spikes(power_volume(copy.astype(np.float32) - orig.astype(np.float32)))
    else:  # identical copy: there is no residual to measure
        tr, sr = np.full_like(to, np.nan), np.full_like(so, np.nan)

    return dict(
        psnr_db=psnr_db(orig, copy),
        frame_error=e,
        error_coef=cycle_coefficients(e),
        time_spike_db=tc - to,
        space_spike_db=sc - so,
        resid_time_spike_db=tr,
        resid_space_spike_db=sr,
        orig_time_spike_db=to,
        orig_space_spike_db=so,
        time_map_db=_ratio_db(time_radius_map(Pc), time_radius_map(Po)).astype(
            np.float32
        ),
        space_map_db=_ratio_db(space_map(Pc), space_map(Po)).astype(np.float32),
    )


# --------------------------------------------------------------------- aggregation
def clustered_mean_sem(
    values: np.ndarray, groups
) -> tuple[np.ndarray, np.ndarray, int]:
    """Mean and standard error over SOURCE VIDEOS, not over clips.

    Several windows cut from one video are not independent samples, so they are averaged
    within their video first and the standard error is taken across videos.

    Args:
        values: (n_clips, ...) one row per clip.
        groups: n_clips labels; clips sharing a label came from the same video.

    Returns:
        (mean, sem, n_groups). NaN entries are ignored per bin.
    """
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups)
    with warnings.catch_warnings():
        # bins that are NaN in every clip (DC, Nyquist) are expected; keep them NaN quietly
        warnings.simplefilter("ignore", category=RuntimeWarning)
        per_group = np.stack(
            [np.nanmean(values[groups == g], axis=0) for g in np.unique(groups)]
        )
        n = np.sum(~np.isnan(per_group), axis=0)
        mean = np.nanmean(per_group, axis=0)
        sem = np.where(
            n > 1, np.nanstd(per_group, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1)), 0.0
        )
    return mean, np.where(np.isnan(mean), np.nan, sem), len(per_group)
