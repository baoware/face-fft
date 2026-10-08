"""Tests for the VAE round-trip measurements (NEXT_STEPS step 1).

Every test plants a known artifact in synthetic video and checks that the measurement
reports it at the right frequency bin and nowhere else.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from face_fft.eval import roundtrip_metrics as rm

T, H, W = 48, 64, 64


def noise_video(seed=0, t=T, h=H, w=W):
    """White-noise luma volume (T, H, W), float32, roughly in uint8 range."""
    rng = np.random.default_rng(seed)
    return rng.normal(128.0, 20.0, size=(t, h, w)).astype(np.float32)


# ------------------------------------------------------------------------ prominence
def test_prominence_curve_matches_lossless_test_measure():
    """Same number as bin/codec_control.peak_prominence_db, so results line up."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "bin"))
    codec_control = pytest.importorskip("codec_control")
    P = np.random.default_rng(1).exponential(size=T)
    curve = rm.prominence_curve_db(P)
    for period in (3, 4, 6, 8, 12):
        assert curve[T // period] == pytest.approx(
            codec_control.peak_prominence_db(P, period)
        )


def test_prominence_curve_flags_only_the_spiked_bin():
    P = np.ones(T)
    P[12] = 100.0
    curve = rm.prominence_curve_db(P)
    assert curve[12] == pytest.approx(20.0)
    others = np.delete(curve, 12)
    assert np.nanmax(np.abs(others)) == pytest.approx(0.0)


def test_prominence_curve_has_one_value_per_bin_up_to_nyquist():
    curve = rm.prominence_curve_db(np.ones(T))
    assert curve.shape == (T // 2 + 1,)
    assert np.isnan(curve[0])  # DC has no meaningful neighbourhood


# ------------------------------------------------------------- 1: per-frame error
def test_frame_error_is_mse_per_frame():
    orig = np.zeros((4, 8, 8), np.float32)
    copy = orig.copy()
    copy[2] += 3.0
    assert rm.frame_error(orig, copy) == pytest.approx([0.0, 0.0, 9.0, 0.0])


def test_error_cycle_reports_period_of_the_error_pattern():
    """Error that is larger on every 4th frame -> modulation at bin T/4 only."""
    e = np.ones(T)
    e[::4] = 2.0
    depth = rm.modulation_depth(e)
    assert depth.shape == (T // 2 + 1,)
    assert depth[T // 4] > 0.3
    # a 1-in-4 pulse train also has its 2nd harmonic at period 2; nothing else
    quiet = np.delete(depth, [0, T // 4, T // 2])
    assert np.max(quiet) < 1e-6


def test_error_cycle_amplitude_is_relative_to_mean_error():
    """A sinusoidal +-10% swing of the error reads as 0.10, whatever the error scale."""
    t = np.arange(T)
    for scale in (1.0, 50.0):
        e = scale * (1.0 + 0.10 * np.cos(2 * np.pi * t / 8))
        assert rm.modulation_depth(e)[T // 8] == pytest.approx(0.10, abs=1e-6)


def test_modulation_depth_of_zero_error_is_zero():
    assert np.all(rm.modulation_depth(np.zeros(T)) == 0.0)


# ------------------------------------------------------------- 2: temporal spike
def test_temporal_marginal_bin_k_is_k_cycles_per_clip():
    t = np.arange(T, dtype=np.float32)[:, None, None]
    x = np.cos(2 * np.pi * t / 4) * np.ones((T, H, W), np.float32)
    P = rm.temporal_marginal(rm.power_volume(x))
    assert P.shape == (T,)
    assert int(np.argmax(P[: T // 2 + 1])) == T // 4


def test_temporal_spike_appears_at_planted_period_only():
    orig = noise_video()
    t = np.arange(T, dtype=np.float32)[:, None, None]
    # decoder that brightens every 4th frame
    copy = orig + 20.0 * (t % 4 == 0)
    m = rm.clip_metrics(orig, copy)
    d = np.array(m["time_spike_db"])
    assert d[T // 4] > 3.0
    quiet = np.delete(d, [0, T // 4, T // 2])
    assert np.nanmax(np.abs(quiet)) < 1.0


def test_residual_spike_sees_an_artifact_too_weak_for_the_paired_spike():
    """Subtracting content before the FFT is why the residual variant is reported too."""
    orig = noise_video()
    t = np.arange(T, dtype=np.float32)[:, None, None]
    decoder_noise = np.random.default_rng(3).normal(0.0, 0.5, orig.shape)
    copy = orig + 2.0 * (t % 4 == 0) + decoder_noise
    m = rm.clip_metrics(orig, copy)
    assert m["time_spike_db"][T // 4] < 1.0
    assert m["resid_time_spike_db"][T // 4] > 3.0


# -------------------------------------------------------------- 3: spatial spike
def test_spatial_spike_appears_at_planted_grid_spacing():
    orig = noise_video()
    grid = np.zeros((H, W), np.float32)
    grid[::8, :] = 30.0
    grid[:, ::8] = 30.0
    m = rm.clip_metrics(orig, orig + grid)
    d = np.array(m["space_spike_db"])
    assert d.shape == (H // 2 + 1,)
    assert d[H // 8] > 3.0  # 8-pixel spacing
    assert abs(d[H // 16 + 1]) < 1.0  # a neighbouring, non-harmonic spacing


# ------------------------------------------------------------ 4: difference maps
def test_difference_maps_show_blur_as_high_frequency_loss():
    orig = noise_video()
    # 2x2 box blur inside each frame: removes high spatial frequencies, keeps low ones
    copy = (
        orig
        + np.roll(orig, 1, axis=1)
        + np.roll(orig, 1, axis=2)
        + np.roll(orig, 1, axis=(1, 2))
    ) / 4
    m = rm.clip_metrics(orig, copy)
    time_map, space_map = m["time_map_db"], m["space_map_db"]
    assert time_map.shape == (T // 2 + 1, H // 2 + 1)  # temporal bin x spatial radius
    assert space_map.shape == (H, W)  # fftshifted: DC in the centre
    assert time_map[:, -4:].mean() < -6.0  # high spatial radius lost
    assert abs(time_map[1:, 1:4].mean()) < 1.0  # low spatial radius kept
    assert space_map[0, 0] < -6.0  # corner = highest spatial frequency
    assert abs(space_map[H // 2, W // 2 + 2]) < 1.0  # near DC


# ----------------------------------------------------------------- identity/PSNR
def test_identical_copy_gives_zero_everywhere():
    orig = noise_video()
    m = rm.clip_metrics(orig, orig.copy())
    assert m["psnr_db"] == float("inf")
    assert np.all(np.array(m["frame_error"]) == 0.0)
    assert np.nanmax(np.abs(m["time_spike_db"])) == 0.0
    assert np.nanmax(np.abs(m["space_spike_db"])) == 0.0
    assert np.all(m["time_map_db"] == 0.0)
    assert np.all(m["space_map_db"] == 0.0)


def test_psnr_uses_255_peak():
    orig = np.zeros((2, 4, 4), np.float32)
    assert rm.psnr_db(orig, orig + 255.0) == pytest.approx(0.0)
    assert rm.psnr_db(orig, orig + 25.5) == pytest.approx(20.0)


def test_content_cancels_in_paired_spike():
    """A period-4 pattern already in the ORIGINAL is not blamed on the decoder."""
    t = np.arange(T, dtype=np.float32)[:, None, None]
    orig = noise_video() + 5.0 * (t % 4 == 0)
    copy = orig + noise_video(seed=7) * 0.01
    m = rm.clip_metrics(orig, copy)
    assert abs(m["time_spike_db"][T // 4]) < 0.5


def test_luma_averages_channels():
    rgb = np.zeros((2, 4, 4, 3), np.uint8)
    rgb[..., 0] = 30
    out = rm.luma(rgb)
    assert out.shape == (2, 4, 4) and out.dtype == np.float32
    assert np.all(out == 10.0)


# -------------------------------------------------------------------- aggregation
def test_clustered_sem_treats_windows_of_one_video_as_one_sample():
    # two source videos, two windows each; windows of a video agree exactly
    values = np.array([[1.0], [1.0], [3.0], [3.0]])
    mean, sem, n = rm.clustered_mean_sem(values, ["a", "a", "b", "b"])
    assert n == 2
    assert mean == pytest.approx([2.0])
    assert sem == pytest.approx([1.0])  # std of {1, 3} (ddof=1) / sqrt(2)


def test_clustered_sem_ignores_nan_bins():
    values = np.array([[np.nan, 1.0], [np.nan, 3.0]])
    mean, sem, _ = rm.clustered_mean_sem(values, ["a", "b"])
    assert np.isnan(mean[0]) and mean[1] == pytest.approx(2.0)


def test_cycle_coefficients_keep_phase_so_clips_can_be_averaged_coherently():
    """A decoder's cycle has the same phase in every clip; content-driven wobble does not."""
    t = np.arange(T)
    same = 1.0 + 0.2 * np.cos(2 * np.pi * t / 4)
    opposite = 1.0 - 0.2 * np.cos(2 * np.pi * t / 4)
    k = T // 4
    assert abs(rm.cycle_coefficients(same)[k]) == pytest.approx(0.2, abs=1e-6)
    assert abs(
        (rm.cycle_coefficients(same)[k] + rm.cycle_coefficients(same)[k]) / 2
    ) == pytest.approx(0.2, abs=1e-6)
    assert (
        abs((rm.cycle_coefficients(same)[k] + rm.cycle_coefficients(opposite)[k]) / 2)
        < 1e-6
    )
    assert np.allclose(np.abs(rm.cycle_coefficients(same)), rm.modulation_depth(same))


# ------------------------------------------------------- reading a bin against its neighbours
def test_neighbour_median_ignores_the_bin_itself_and_its_shoulders():
    curve = np.ones(T // 2 + 1)
    curve[11:14] = 50.0  # the bin under test and both shoulders
    assert rm.neighbour_median(curve, 12) == pytest.approx(1.0)


def test_neighbour_median_skips_undefined_bins():
    curve = np.full(T // 2 + 1, 2.0)
    curve[:2] = np.nan  # DC and bin 1 are undefined for prominence curves
    assert rm.neighbour_median(curve, 4) == pytest.approx(2.0)


def test_edge_transient_is_broadband_so_it_does_not_stand_out_from_neighbours():
    """A worse last frame is locked to clip position like a decoder cycle, but it has
    no period: it lifts every bin equally. Only the neighbour comparison tells them apart."""
    e = np.ones(T)
    e[-1] = 1.5
    depth = rm.modulation_depth(e)
    assert depth[T // 4] > 0.01  # read against 0, it looks like a cycle ...
    assert depth[T // 4] == pytest.approx(
        rm.neighbour_median(depth, T // 4), rel=0.05
    )  # ... but it is not


def test_original_spike_is_reported_so_a_source_cycle_is_visible():
    t = np.arange(T, dtype=np.float32)[:, None, None]
    orig = noise_video() + 20.0 * (t % 4 == 0)
    m = rm.clip_metrics(
        orig, orig + np.random.default_rng(5).normal(0, 0.5, orig.shape)
    )
    assert m["orig_time_spike_db"][T // 4] > 3.0
    assert m["orig_space_spike_db"].shape == (H // 2 + 1,)
