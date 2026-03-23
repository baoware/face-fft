from pathlib import Path

import torch
import torchvision.io as io

from face_fft.data.deepaction import (
    DeepActionDataset,
    DeepActionSample,
    discover_deepaction_samples,
    split_deepaction_samples,
)


def _make_video_tensor(T: int, H: int = 32, W: int = 32) -> torch.Tensor:
    """Returns a random uint8 video tensor of shape (T, H, W, 3)."""
    return torch.randint(0, 256, (T, H, W, 3), dtype=torch.uint8)


def _write_fake_mp4(path: Path, T: int = 4, H: int = 32, W: int = 32, fps: float = 8.0):
    frames = _make_video_tensor(T=T, H=H, W=W)
    path.parent.mkdir(parents=True, exist_ok=True)
    io.write_video(str(path), frames, fps=fps)


def test_discover_deepaction_samples_labels_by_top_folder(tmp_path: Path):
    _write_fake_mp4(tmp_path / "Pexels" / "classA" / "real1.mp4")
    _write_fake_mp4(tmp_path / "AnimateDiff" / "classB" / "synth1.mp4")
    _write_fake_mp4(tmp_path / "CogVideoX5B" / "classB" / "synth2.mp4")

    samples = discover_deepaction_samples(
        tmp_path, real_folder_name="Pexels", synth_folder_names=None
    )
    assert len(samples) == 3

    labels = {Path(s.video_path).name: s.label for s in samples}
    assert labels["real1.mp4"] == 0
    assert labels["synth1.mp4"] == 1
    assert labels["synth2.mp4"] == 1


def test_split_deepaction_samples_is_deterministic(tmp_path: Path):
    samples = [
        DeepActionSample(video_path=str(tmp_path / "r1.mp4"), label=0),
        DeepActionSample(video_path=str(tmp_path / "r2.mp4"), label=0),
        DeepActionSample(video_path=str(tmp_path / "s1.mp4"), label=1),
        DeepActionSample(video_path=str(tmp_path / "s2.mp4"), label=1),
    ]
    train1, val1, test1 = split_deepaction_samples(
        samples, train_ratio=0.5, val_ratio=0.25, seed=123
    )
    train2, val2, test2 = split_deepaction_samples(
        samples, train_ratio=0.5, val_ratio=0.25, seed=123
    )

    assert [s.video_path for s in train1] == [s.video_path for s in train2]
    assert [s.video_path for s in val1] == [s.video_path for s in val2]
    assert [s.video_path for s in test1] == [s.video_path for s in test2]


def test_deepaction_dataset_returns_expected_tensor_shape(tmp_path: Path):
    _write_fake_mp4(tmp_path / "Pexels" / "classA" / "real1.mp4", T=4, H=32, W=32)
    _write_fake_mp4(
        tmp_path / "AnimateDiff" / "classB" / "synth1.mp4", T=4, H=32, W=32
    )

    samples = discover_deepaction_samples(tmp_path)
    dataset = DeepActionDataset(
        samples,
        target_size=(64, 64),
        num_frames=4,
        max_duration_sec=10.0,
        skip_errors=False,
    )

    tensor, label = dataset[0]
    assert tensor.shape == (3, 4, 64, 64)
    assert tensor.dtype == torch.float32
    assert tensor.min().item() >= 0.0
    assert tensor.max().item() <= 1.0
    assert label.item() in (0, 1)


def test_deepaction_dataset_skip_errors_returns_zeros(tmp_path: Path):
    bad_sample = DeepActionSample(video_path=str(tmp_path / "missing.mp4"), label=1)
    dataset = DeepActionDataset(
        [bad_sample],
        target_size=(32, 32),
        num_frames=4,
        max_duration_sec=1.0,
        skip_errors=True,
    )

    tensor, label = dataset[0]
    assert tensor.shape == (3, 4, 32, 32)
    assert torch.all(tensor == 0)
    assert label.item() == 1

