"""Tests for bin/vae_roundtrip.py: clip selection, framing, and the analyze stage
end to end on synthetic "round-trips" with planted artifacts."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "bin"))
import vae_roundtrip as vr  # noqa: E402


# --------------------------------------------------------------------- clip plan
def test_clip_plan_takes_first_window_of_every_video_before_any_second_window():
    videos = [("1", Path("1/a.mp4"), 200), ("2", Path("2/b.mp4"), 200)]
    plan = vr.clip_plan(videos, n=3, frames=49)
    assert [(c["pexels_id"], c["window"]) for c in plan] == [
        ("1", 0),
        ("2", 0),
        ("1", 1),
    ]


def test_clip_plan_staggers_start_frames_across_videos():
    """Every file starts on a keyframe; identical start frames would lock the source
    codec's frame-type cadence to the same phase in every clip, like a decoder cycle."""
    videos = [(str(i), Path(f"{i}/a.mp4"), 500) for i in range(10)]
    starts = [c["start"] for c in vr.clip_plan(videos, n=10, frames=49)]
    assert starts == [0, 1, 2, 3, 4, 5, 6, 7, 0, 1]


def test_clip_plan_windows_do_not_overlap_and_fit_inside_the_video():
    videos = [("1", Path("1/a.mp4"), 120), ("2", Path("2/a.mp4"), 98)]
    plan = vr.clip_plan(videos, n=10, frames=49)
    assert [(c["pexels_id"], c["start"]) for c in plan] == [
        ("1", 0),
        ("2", 1),
        ("1", 49),
    ]
    # video 2 starts at frame 1, so its second window would need 99 frames


def test_clip_plan_ids_are_unique_and_stable():
    videos = [("1", Path("1/a.mp4"), 200), ("1", Path("1/b.mp4"), 200)]
    ids = [c["clip"] for c in vr.clip_plan(videos, n=4, frames=49)]
    assert ids == ["1_a_w0", "1_b_w0", "1_a_w1", "1_b_w1"]


def test_frame_count_must_be_one_plus_a_multiple_of_the_stride():
    vr.check_frames(49, k=4)
    vr.check_frames(49, k=8)
    vr.check_frames(50, k=None)  # frame-wise control: any length
    with pytest.raises(ValueError):
        vr.check_frames(48, k=4)
    with pytest.raises(ValueError):
        vr.check_frames(45, k=8)


def test_variant_name_marks_non_default_decoder_chunking():
    assert vr.variant_name("cogvideox", None) == "cogvideox"
    assert vr.variant_name("cogvideox", 13) == "cogvideox-dec13"
    with pytest.raises(ValueError):
        vr.variant_name("wan", 2)


# ----------------------------------------------------------------------- framing
def test_resize_view_centre_crops_to_square_then_resizes():
    frame = np.zeros((100, 300, 3), np.uint8)
    frame[:, 100:200] = 200  # the central square is bright, the sides dark
    out = vr.to_view(frame, 32, "resize")
    assert out.shape == (32, 32, 3) and out.dtype == np.uint8
    assert out.min() == 200


def test_crop_view_keeps_native_pixels():
    frame = (
        np.arange(100 * 300 * 3, dtype=np.uint32).reshape(100, 300, 3).astype(np.uint8)
    )
    out = vr.to_view(frame, 32, "crop")
    assert np.array_equal(out, frame[34:66, 134:166])


def test_crop_view_rejects_frames_smaller_than_the_crop():
    with pytest.raises(ValueError):
        vr.to_view(np.zeros((20, 300, 3), np.uint8), 32, "crop")


def test_save_atomic_leaves_only_the_final_file(tmp_path):
    vr.save_atomic(tmp_path / "a.npy", np.arange(4))
    assert [f.name for f in tmp_path.iterdir()] == ["a.npy"]
    assert np.array_equal(np.load(tmp_path / "a.npy"), np.arange(4))


# ------------------------------------------------------------- tensor round trip
def test_uint8_survives_conversion_to_model_range_and_back():
    torch = pytest.importorskip("torch")
    x = np.random.default_rng(0).integers(0, 256, size=(5, 8, 8, 3), dtype=np.uint8)
    t = vr.to_model(x)
    assert t.shape == (1, 3, 5, 8, 8) and t.dtype == torch.float32
    assert float(t.min()) >= -1.0 and float(t.max()) <= 1.0
    assert np.array_equal(vr.from_model(t), x)


# --------------------------------------------------------------- analyze, end to end
def _write_fake_run(root: Path, n_clips=6, T=48, size=32, extra=()):
    """Originals plus synthetic "round-trips":
        cogvideox        every 4th frame degraded             (a real 4-frame cycle)
        sd               uniform noise                        (clean control)
        ltx              last frame degraded                  (edge transient, NOT a cycle)
    `extra` names more directories that get the cogvideox treatment."""
    rng = np.random.default_rng(0)
    names = ("real", "cogvideox", "sd", "ltx") + tuple(extra)
    for name in names:
        (root / "resize" / name).mkdir(parents=True)
    idx = np.arange(T + 1)
    scale = {
        "sd": np.full(T + 1, 2.0),
        "cogvideox": np.where((idx - 1) % 4 == 0, 6.0, 2.0),
        "ltx": np.where(idx == T, 3.0, 2.0),
    }
    for i in range(n_clips):
        orig = rng.integers(40, 216, size=(T + 1, size, size, 3)).astype(np.uint8)
        clip = f"{i}_v_w0"
        np.save(root / "resize" / "real" / f"{clip}.npy", orig)
        for name in names[1:]:
            s = scale.get(name, scale["cogvideox"])[:, None, None, None]
            copy = np.clip(orig + rng.normal(0, 1, orig.shape) * s, 0, 255).astype(
                np.uint8
            )
            np.save(root / "resize" / name / f"{clip}.npy", copy)


def _run(tmp_path, **kw):
    frames, results = tmp_path / "frames", tmp_path / "results"
    _write_fake_run(frames, **kw)
    return frames, results, vr.analyze(frames, results, view="resize", workers=0)


def test_analyze_finds_planted_cycle_in_test_vae_and_not_in_control(tmp_path):
    _, results, table = _run(tmp_path)
    cog, sd = table[("cogvideox", 4)], table[("sd", 4)]
    assert cog["error_cycle"] > 5 * cog["error_cycle_sem"]
    assert (
        cog["error_cycle"] > 10 * cog["error_cycle_nbr"]
    )  # stands out from its own neighbours
    assert sd["error_cycle"] < 0.1
    assert cog["n_videos"] == 6
    assert 25 < cog["psnr_db"] < 50

    out = results / "resize"
    for name in (
        "roundtrip_summary.txt",
        "roundtrip_curves.csv",
        "roundtrip_maps.npz",
        "error_over_time.png",
        "time_difference_map.png",
        "space_difference_map.png",
    ):
        assert (out / name).stat().st_size > 0, name
    assert "cogvideox" in (out / "roundtrip_summary.txt").read_text()


def test_edge_transient_does_not_stand_out_from_its_neighbours(tmp_path):
    """The false positive the neighbour column exists for: clear of 0, but not a peak."""
    _, _, table = _run(tmp_path, n_clips=8, size=16)
    ltx = table[("ltx", 8)]
    assert ltx["error_cycle"] > 2 * ltx["error_cycle_sem"]  # "clear of 0" ...
    assert (
        ltx["error_cycle"] < 1.5 * ltx["error_cycle_nbr"]
    )  # ... yet no higher than the bins around it


def test_analyze_drops_the_first_frame_not_the_last(tmp_path):
    """The VAEs encode frame 0 alone; ruin it in the copy and the analysis must not notice."""
    frames, results = tmp_path / "frames", tmp_path / "results"
    _write_fake_run(frames, n_clips=2)
    for f in (frames / "resize" / "sd").glob("*.npy"):
        copy = np.load(f)
        copy[0] = 0
        np.save(f, copy)
    vr.analyze(frames, results, view="resize", workers=0)
    with np.load(next((results / "resize" / "clips").glob("sd__*.npz"))) as clip:
        assert clip["frame_error"].shape == (48,)
        assert (
            clip["frame_error"].max() < 20
        )  # frame 0's error would be in the thousands


def test_analyze_is_resumable(tmp_path):
    frames, results, _ = _run(tmp_path, n_clips=2)
    done = sorted((results / "resize" / "clips").glob("*.npz"))
    stamps = [f.stat().st_mtime_ns for f in done]
    vr.analyze(frames, results, view="resize", workers=0)
    assert [f.stat().st_mtime_ns for f in done] == stamps


def test_analyze_redoes_a_clip_whose_frames_are_newer_than_its_measurements(tmp_path):
    frames, results, _ = _run(tmp_path, n_clips=2)
    copy = sorted((frames / "resize" / "sd").glob("*.npy"))[0]
    done = results / "resize" / "clips" / f"sd__{copy.stem}.npz"
    earlier = done.stat().st_mtime - 100
    os.utime(
        done, (earlier, earlier)
    )  # as if the round-trip was regenerated after it was measured
    before = done.stat().st_mtime_ns
    vr.analyze(frames, results, view="resize", workers=0)
    assert done.stat().st_mtime_ns != before


def test_summary_uses_only_clips_every_vae_has(tmp_path):
    """A half-finished VAE must not be compared with the control on different clips."""
    frames, results = tmp_path / "frames", tmp_path / "results"
    _write_fake_run(frames, n_clips=4)
    sorted((frames / "resize" / "sd").glob("*.npy"))[0].unlink()
    table = vr.analyze(frames, results, view="resize", workers=0)
    assert table[("sd", 4)]["n_clips"] == 3
    assert table[("cogvideox", 4)]["n_clips"] == 3


def test_ablation_directories_are_analysed_with_their_vaes_stride(tmp_path):
    _, _, table = _run(tmp_path, n_clips=2, extra=("cogvideox-dec13",))
    assert table[("cogvideox-dec13", 4)]["k"] == 4


def test_leftover_temp_files_are_not_read_as_results(tmp_path):
    frames, results, _ = _run(tmp_path, n_clips=2)
    (results / "resize" / "clips" / ".sd__0_v_w0.host.123.tmp").write_bytes(
        b"half a file"
    )
    (frames / "resize" / "sd" / ".9_v_w0.host.123.tmp").write_bytes(b"half a file")
    vr.analyze(frames, results, view="resize", workers=0)


def test_cogvideox_decoder_chunk_sizes_that_would_drop_frames_are_rejected():
    """CogVideoX only upsamples a chunk in time correctly if the first chunk has an odd
    number of latent frames and the rest an even number; 49 frames = 13 latent frames."""
    for ok in (2, 4, 6, 12, 13):
        vr.check_decode_latents(49, ok)
    for bad in (1, 3, 5, 7, 14):
        with pytest.raises(ValueError):
            vr.check_decode_latents(49, bad)
