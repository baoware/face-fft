import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset
import torchvision.io as io

from face_fft.data.generate import preprocess_video_tensor


@dataclass(frozen=True)
class DeepActionSample:
    video_path: str
    label: int  # 0 = Real, 1 = Synthetic


def _iter_video_paths(
    root_dir: Path, *, allowed_exts: Sequence[str]
) -> Iterable[Path]:
    allowed = {ext.lower() for ext in allowed_exts}
    for p in root_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in allowed:
            yield p


def discover_deepaction_samples(
    root_dir: str | Path,
    *,
    real_folder_name: str = "Pexels",
    synth_folder_names: Optional[Sequence[str]] = None,
    allowed_exts: Sequence[str] = (".mp4", ".avi", ".mov"),
    limit: Optional[int] = None,
) -> List[DeepActionSample]:
    """
    Discover DeepAction v1 videos from an on-disk dataset snapshot.

    Expected directory layout:
      <root>/Pexels/<class_name>/*.mp4                -> label 0 (real)
      <root>/<SyntheticModel>/<class_name>/*.mp4    -> label 1 (synthetic)
    """
    root = Path(root_dir)
    if not root.exists():
        raise ValueError(f"DeepAction root does not exist: {root}")

    real_folder_lc = real_folder_name.lower()
    synth_filter = None
    if synth_folder_names is not None:
        synth_filter = {n.lower() for n in synth_folder_names}

    samples: List[DeepActionSample] = []
    for video_path in _iter_video_paths(root, allowed_exts=allowed_exts):
        rel = video_path.relative_to(root)
        if not rel.parts:
            continue
        top_folder = rel.parts[0].lower()

        if top_folder == real_folder_lc:
            label = 0
        else:
            if synth_filter is not None and top_folder not in synth_filter:
                continue
            label = 1

        samples.append(DeepActionSample(video_path=str(video_path), label=label))
        if limit is not None and len(samples) >= limit:
            break

    if not samples:
        raise ValueError(
            f"No videos discovered under {root} "
            f"(real_folder={real_folder_name}, synth_filter={synth_folder_names})."
        )

    return samples


def split_deepaction_samples(
    samples: List[DeepActionSample],
    *,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[DeepActionSample], List[DeepActionSample], List[DeepActionSample]]:
    """
    Split unpaired samples while roughly preserving the Real/Synth label ratio.
    """
    real = [s for s in samples if s.label == 0]
    synth = [s for s in samples if s.label == 1]

    rng = random.Random(seed)
    rng.shuffle(real)
    rng.shuffle(synth)

    def _split_one(lst: List[DeepActionSample]) -> Tuple[List[DeepActionSample], List[DeepActionSample], List[DeepActionSample]]:
        n = len(lst)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train = lst[:n_train]
        val = lst[n_train : n_train + n_val]
        test = lst[n_train + n_val :]
        return train, val, test

    real_train, real_val, real_test = _split_one(real)
    synth_train, synth_val, synth_test = _split_one(synth)

    train = real_train + synth_train
    val = real_val + synth_val
    test = real_test + synth_test

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return train, val, test


class DeepActionDataset(Dataset):
    """
    On-the-fly loader converting DeepAction videos into Face-FFT tensor format:
      (C, T, H, W) with C=3 and H=W=target_size.
    """

    def __init__(
        self,
        samples: List[DeepActionSample],
        *,
        target_size: Tuple[int, int] = (256, 256),
        num_frames: int = 16,
        max_duration_sec: float = 2.0,
        skip_errors: bool = False,
    ):
        self.samples = samples
        self.target_size = target_size
        self.num_frames = num_frames
        self.max_duration_sec = max_duration_sec
        self.skip_errors = skip_errors

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        try:
            # Read only an initial time window for efficiency (HPC-friendly).
            # We keep a robust fallback to a full read if the trimmed decode yields 0 frames.
            initial_end = self.max_duration_sec if self.max_duration_sec and self.max_duration_sec > 0 else None
            video, _, info = io.read_video(
                sample.video_path, pts_unit="sec", start_pts=0.0, end_pts=initial_end
            )
            if video.numel() == 0:
                # Some decoders can return 0 frames for very small end_pts values;
                # retry once with a full read before declaring failure.
                if initial_end is not None:
                    video, _, info = io.read_video(
                        sample.video_path, pts_unit="sec", start_pts=0.0, end_pts=None
                    )
                if video.numel() == 0:
                    raise ValueError("Empty video stream")

            video_full = video  # Keep a fallback in case the trimmed read is empty.
            fps = None
            if isinstance(info, dict):
                fps = info.get("video_fps", None)

            if fps is not None and fps > 0:
                target_end = min(
                    self.max_duration_sec, float(self.num_frames) / float(fps)
                )
                # If the truncated time window is too small, the trimmed read may
                # legitimately return zero frames. In that case, keep the full
                # video we already decoded above.
                if target_end <= 0:
                    video = video_full
                else:
                    video_trim, _, _ = io.read_video(
                        sample.video_path,
                        pts_unit="sec",
                        start_pts=0.0,
                        end_pts=target_end,
                    )
                    if video_trim.numel() == 0:
                        video = video_full
                    else:
                        video = video_trim

            tensor = preprocess_video_tensor(
                video, target_size=self.target_size, num_frames=self.num_frames
            )
            label = torch.tensor(sample.label, dtype=torch.long)
            return tensor, label
        except Exception:
            if not self.skip_errors:
                raise

            # Fallback: allow training/evaluation to proceed.
            C = 3
            T = self.num_frames
            H, W = self.target_size
            tensor = torch.zeros((C, T, H, W), dtype=torch.float32)
            label = torch.tensor(sample.label, dtype=torch.long)
            return tensor, label

