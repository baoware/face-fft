import os
import random
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class GenVidBenchDataset(Dataset):
    """
    PyTorch Dataset for GenVidBench videos.

    Mirrors the DeepActionDataset interface: yields (video_tensor, label)
    where label=0 is Real and label=1 is Synthetic.

    Uses cv2.CAP_FFMPEG backend to avoid the icvExtractPattern crash
    that occurs with the default OpenCV backend on GenVidBench .mp4 files.
    """

    def __init__(
        self,
        samples: List[Tuple[str, int]],
        target_frames: int = 8,
        target_size: Tuple[int, int] = (256, 256),
        min_file_bytes: int = 1024,
    ):
        self.samples = samples
        self.target_frames = target_frames
        self.target_size = target_size
        self.min_file_bytes = min_file_bytes

    def __len__(self) -> int:
        return len(self.samples)

    def _load_video(self, video_path: str) -> torch.Tensor:
        if os.path.getsize(video_path) < self.min_file_bytes:
            raise ValueError(f"File too small (likely corrupted): {video_path}")

        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            raise ValueError(f"FFMPEG backend cannot open: {video_path}")

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (self.target_size[1], self.target_size[0]))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        if len(frames) < self.target_frames:
            raise ValueError(
                f"Only {len(frames)} frames decoded from {video_path}, "
                f"need at least {self.target_frames}."
            )

        indices = np.linspace(0, len(frames) - 1, self.target_frames).astype(int)
        sampled = [frames[i] for i in indices]
        return torch.from_numpy(np.array(sampled)).permute(3, 0, 1, 2).float() / 255.0

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        video_path, label = self.samples[idx]
        try:
            v = self._load_video(video_path)
        except Exception as e:
            print(f"Warning: failed to load {video_path}: {e}")
            v = torch.zeros(
                (3, self.target_frames, self.target_size[0], self.target_size[1])
            )
        return v, label


def collect_mp4s(directory: str) -> List[str]:
    """Recursively collect sorted .mp4 paths under directory."""
    paths = []
    for dp, _, filenames in os.walk(directory):
        for f in filenames:
            if f.lower().endswith(".mp4"):
                paths.append(os.path.join(dp, f))
    return sorted(paths)


# Keep private alias so existing internal callers continue to work
_collect_mp4s = collect_mp4s


def get_genvidbench_splits(
    root_dir: str,
    synth_generators: List[str] = ["Sora", "Kling", "OpenSora"],
    real_folder: str = "Real",
    real_paths_override: Optional[List[str]] = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    target_frames: int = 8,
    target_size: Tuple[int, int] = (256, 256),
    seed: int = 42,
    max_per_generator: Optional[int] = None,
) -> Tuple[GenVidBenchDataset, GenVidBenchDataset, GenVidBenchDataset]:
    """
    Build balanced train/val/test splits from a local GenVidBench directory.

    Expected layout::

        root_dir/
          Real/       <- real videos  (label=0)
          Sora/       <- Sora fakes   (label=1)
          Kling/      <- Kling fakes  (label=1)
          OpenSora/   <- OpenSora fakes (label=1)

    If ``Real/`` is absent you can supply ``real_paths_override`` — a flat list
    of .mp4 paths from another real-video source (e.g. DeepAction Pexels).
    Classes are balanced to min(|real|, |synth|) at split time.
    """
    root_path = Path(root_dir)
    rng = random.Random(seed)

    # --- Collect real videos ---
    if real_paths_override is not None:
        real_paths = list(real_paths_override)
        print(f"GenVidBench: using {len(real_paths)} real videos from override list.")
    else:
        real_dir = root_path / real_folder
        if not real_dir.exists():
            raise FileNotFoundError(
                f"Real video folder not found at {real_dir}. "
                "Run scripts/download_genvidbench_full.slurm to fetch real videos, "
                "or pass real_paths_override= with paths from another source "
                "(e.g. DeepAction Pexels folder)."
            )
        real_paths = _collect_mp4s(str(real_dir))
        if not real_paths:
            raise ValueError(f"No .mp4 files found under {real_dir}.")
        print(f"GenVidBench: found {len(real_paths)} real videos in {real_dir}.")

    # --- Collect synthetic videos ---
    synth_paths: List[str] = []
    for gen in synth_generators:
        gen_dir = root_path / gen
        if not gen_dir.exists():
            print(f"Warning: generator folder missing: {gen_dir}. Skipping.")
            continue
        vids = _collect_mp4s(str(gen_dir))
        if max_per_generator is not None:
            vids = rng.sample(vids, min(max_per_generator, len(vids)))
        print(f"GenVidBench: found {len(vids)} synthetic videos for {gen}.")
        synth_paths.extend(vids)

    if not synth_paths:
        raise ValueError(
            f"No synthetic videos found under {root_dir} "
            f"for generators {synth_generators}. Check archives were extracted."
        )

    # Balance classes to avoid label imbalance
    n = min(len(real_paths), len(synth_paths))
    real_sample = rng.sample(real_paths, n)
    synth_sample = rng.sample(synth_paths, n)

    # Pair-level shuffle prevents leakage across splits
    pairs = list(zip(real_sample, synth_sample))
    rng.shuffle(pairs)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train : n_train + n_val]
    test_pairs = pairs[n_train + n_val :]

    def _flatten(pair_list: List[Tuple[str, str]]) -> List[Tuple[str, int]]:
        out = []
        for r, s in pair_list:
            out.append((r, 0))
            out.append((s, 1))
        return out

    train_ds = GenVidBenchDataset(_flatten(train_pairs), target_frames, target_size)
    val_ds = GenVidBenchDataset(_flatten(val_pairs), target_frames, target_size)
    test_ds = GenVidBenchDataset(_flatten(test_pairs), target_frames, target_size)

    print(
        f"GenVidBench splits — Train: {len(train_ds)}, "
        f"Val: {len(val_ds)}, Test: {len(test_ds)} "
        f"(balanced {n} real/fake pairs total)"
    )
    return train_ds, val_ds, test_ds
