import torch
import torchvision.io as io

from face_fft.data.dataset import PairedVideoDataset
from face_fft.data.generate import preprocess_video_tensor


# =============================================================================
# Helpers
# =============================================================================


def _make_video_tensor(T: int, H: int = 64, W: int = 64) -> torch.Tensor:
    """Returns a random uint8 video tensor of shape (T, H, W, 3)."""
    return torch.randint(0, 256, (T, H, W, 3), dtype=torch.uint8)


def _write_fake_mp4(path, T: int = 8, H: int = 64, W: int = 64, fps: float = 8.0):
    """Writes a minimal fake .mp4 to path using torchvision write_video."""
    frames = _make_video_tensor(T, H, W)  # (T, H, W, 3)
    io.write_video(str(path), frames, fps=fps)


# =============================================================================
# Unit tests — preprocess_video_tensor behaviour
# =============================================================================


def test_preprocess_produces_correct_shape():
    video = _make_video_tensor(T=16, H=128, W=128)
    out = preprocess_video_tensor(video, target_size=(256, 256), num_frames=16)
    assert out.shape == (3, 16, 256, 256)
    assert out.dtype == torch.float32
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_preprocess_pads_short_video():
    video = _make_video_tensor(T=4, H=64, W=64)
    out = preprocess_video_tensor(video, target_size=(256, 256), num_frames=16)
    assert out.shape == (3, 16, 256, 256)
    # After resize/crop the spatial dims change, so we just check temporal count
    assert out.shape[1] == 16


def test_preprocess_truncates_long_video():
    video = _make_video_tensor(T=32, H=64, W=64)
    out = preprocess_video_tensor(video, target_size=(256, 256), num_frames=16)
    assert out.shape == (3, 16, 256, 256)


# =============================================================================
# Integration tests — full ingestion pipeline
# =============================================================================


def _run_ingestion(
    input_dir,
    real_out_dir,
    synth_out_dir,
    dataset_name="testset",
    num_frames=16,
    target_size=256,
):
    """Run ingestion logic directly (without subprocess) for test isolation."""
    import os
    from pathlib import Path

    from face_fft.data.generate import preprocess_video_tensor

    ALLOWED_EXTS = {".mp4", ".avi", ".mov"}

    real_in_dir = Path(input_dir) / "real"
    synth_in_dir = Path(input_dir) / "synthetic"

    os.makedirs(real_out_dir, exist_ok=True)
    os.makedirs(synth_out_dir, exist_ok=True)

    real_videos = [f for f in real_in_dir.glob("*") if f.suffix.lower() in ALLOWED_EXTS]
    synth_index = {
        f.stem: f for f in synth_in_dir.glob("*") if f.suffix.lower() in ALLOWED_EXTS
    }

    processed = 0
    skipped_unmatched = 0
    skipped_existing = 0

    for real_path in sorted(real_videos):
        stem = real_path.stem
        out_filename = f"{stem}_{dataset_name}.pt"
        real_out_path = Path(real_out_dir) / out_filename
        synth_out_path = Path(synth_out_dir) / out_filename

        if stem not in synth_index:
            skipped_unmatched += 1
            continue

        if real_out_path.exists() and synth_out_path.exists():
            skipped_existing += 1
            continue

        real_raw, _, _ = io.read_video(str(real_path), pts_unit="sec")
        synth_raw, _, _ = io.read_video(str(synth_index[stem]), pts_unit="sec")

        ts = (target_size, target_size)
        real_tensor = preprocess_video_tensor(
            real_raw, target_size=ts, num_frames=num_frames
        )
        synth_tensor = preprocess_video_tensor(
            synth_raw, target_size=ts, num_frames=num_frames
        )

        torch.save(real_tensor, real_out_path)
        torch.save(synth_tensor, synth_out_path)
        processed += 1

    return processed, skipped_unmatched, skipped_existing


def test_ingest_paired_videos_end_to_end(tmp_path):
    real_in = tmp_path / "input" / "real"
    synth_in = tmp_path / "input" / "synthetic"
    real_in.mkdir(parents=True)
    synth_in.mkdir(parents=True)

    _write_fake_mp4(real_in / "clip01.mp4")
    _write_fake_mp4(synth_in / "clip01.mp4")

    real_out = tmp_path / "real_pt"
    synth_out = tmp_path / "synth_pt"

    processed, skipped_unmatched, skipped_existing = _run_ingestion(
        input_dir=tmp_path / "input",
        real_out_dir=real_out,
        synth_out_dir=synth_out,
        dataset_name="testset",
    )

    assert processed == 1
    assert skipped_unmatched == 0

    real_pt = real_out / "clip01_testset.pt"
    synth_pt = synth_out / "clip01_testset.pt"
    assert real_pt.exists()
    assert synth_pt.exists()

    real_tensor = torch.load(real_pt, weights_only=True)
    synth_tensor = torch.load(synth_pt, weights_only=True)

    assert real_tensor.shape == (3, 16, 256, 256)
    assert synth_tensor.shape == (3, 16, 256, 256)
    assert real_tensor.dtype == torch.float32
    assert synth_tensor.dtype == torch.float32
    assert real_tensor.min() >= 0.0
    assert real_tensor.max() <= 1.0


def test_ingest_skips_unmatched_synthetic(tmp_path):
    real_in = tmp_path / "input" / "real"
    synth_in = tmp_path / "input" / "synthetic"
    real_in.mkdir(parents=True)
    synth_in.mkdir(parents=True)

    # Real video with no matching synthetic
    _write_fake_mp4(real_in / "orphan.mp4")

    real_out = tmp_path / "real_pt"
    synth_out = tmp_path / "synth_pt"

    processed, skipped_unmatched, _ = _run_ingestion(
        input_dir=tmp_path / "input",
        real_out_dir=real_out,
        synth_out_dir=synth_out,
    )

    assert processed == 0
    assert skipped_unmatched == 1
    assert not any(real_out.glob("*.pt")) if real_out.exists() else True


def test_ingest_skips_existing_output(tmp_path):
    real_in = tmp_path / "input" / "real"
    synth_in = tmp_path / "input" / "synthetic"
    real_in.mkdir(parents=True)
    synth_in.mkdir(parents=True)

    _write_fake_mp4(real_in / "vid.mp4")
    _write_fake_mp4(synth_in / "vid.mp4")

    real_out = tmp_path / "real_pt"
    synth_out = tmp_path / "synth_pt"

    # First run
    p1, _, _ = _run_ingestion(
        input_dir=tmp_path / "input",
        real_out_dir=real_out,
        synth_out_dir=synth_out,
        dataset_name="ds",
    )
    assert p1 == 1

    # Second run — should skip, not reprocess
    p2, _, skipped_existing = _run_ingestion(
        input_dir=tmp_path / "input",
        real_out_dir=real_out,
        synth_out_dir=synth_out,
        dataset_name="ds",
    )
    assert p2 == 0
    assert skipped_existing == 1


def test_ingest_dataset_outputs_are_compatible_with_paired_dataset(tmp_path):
    real_in = tmp_path / "input" / "real"
    synth_in = tmp_path / "input" / "synthetic"
    real_in.mkdir(parents=True)
    synth_in.mkdir(parents=True)

    for name in ["a", "b", "c"]:
        _write_fake_mp4(real_in / f"{name}.mp4")
        _write_fake_mp4(synth_in / f"{name}.mp4")

    real_out = tmp_path / "real_pt"
    synth_out = tmp_path / "synth_pt"

    processed, _, _ = _run_ingestion(
        input_dir=tmp_path / "input",
        real_out_dir=real_out,
        synth_out_dir=synth_out,
        dataset_name="compat",
    )
    assert processed == 3

    dataset = PairedVideoDataset.from_directories(real_out, synth_out)
    assert len(dataset) == 6  # 3 pairs * 2 (real + synth)

    video, label = dataset[0]
    assert video.shape == (3, 16, 256, 256)
    assert label in (0, 1)
