import os
import random
from pathlib import Path
from typing import List, Tuple, Dict, Union

import torch
from torch.utils.data import Dataset


class PairedVideoDataset(Dataset):
    """
    Dataset for loading paired real and synthetic face videos.
    Assumes videos are stored as PyTorch tensors (.pt files),
    typically with shape (C, T, H, W).
    """

    def __init__(
        self,
        data_pairs: List[Dict[str, str]],
        transform=None,
        yield_pairs: bool = False,
    ):
        """
        Args:
            data_pairs: List of dicts, each containing 'real' and 'synthetic' paths.
            transform: Optional transform to be applied on a video tensor.
            yield_pairs: If True, __getitem__ returns ((real_vid, synth_vid), pair_id).
                         If False, returns (video, label) where label=0 (real) or 1 (synth).
        """
        self.data_pairs = data_pairs
        self.transform = transform
        self.yield_pairs = yield_pairs

    def __len__(self) -> int:
        if self.yield_pairs:
            return len(self.data_pairs)
        return len(self.data_pairs) * 2

    def __getitem__(
        self, idx: int
    ) -> Union[Tuple[Tuple[torch.Tensor, torch.Tensor], int], Tuple[torch.Tensor, int]]:
        if self.yield_pairs:
            pair = self.data_pairs[idx]
            real_vid = torch.load(pair["real"], weights_only=True)
            synth_vid = torch.load(pair["synthetic"], weights_only=True)

            if self.transform:
                real_vid = self.transform(real_vid)
                synth_vid = self.transform(synth_vid)

            return (real_vid, synth_vid), idx
        else:
            pair_idx = idx // 2
            is_synthetic = idx % 2

            pair = self.data_pairs[pair_idx]
            path = pair["synthetic"] if is_synthetic else pair["real"]

            video = torch.load(path, weights_only=True)

            if self.transform:
                video = self.transform(video)

            return video, is_synthetic

    @classmethod
    def from_directories(
        cls, real_dir: Union[str, Path], synth_dir: Union[str, Path], **kwargs
    ):
        """
        Creates dataset from directories assuming paired files have identical names.
        """
        real_dir = Path(real_dir)
        synth_dir = Path(synth_dir)

        real_files = sorted([f for f in os.listdir(real_dir) if f.endswith(".pt")])

        data_pairs = []
        for f in real_files:
            synth_path = synth_dir / f
            if synth_path.exists():
                data_pairs.append(
                    {
                        "real": str(real_dir / f),
                        "synthetic": str(synth_path),
                    }
                )

        if len(data_pairs) == 0:
            raise ValueError(f"No paired .pt files found in {real_dir} and {synth_dir}")

        return cls(data_pairs, **kwargs)


def split_paired_dataset(
    data_pairs: List[Dict[str, str]], train_ratio=0.8, val_ratio=0.1, seed=42
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Randomly splits pairs into train, val, and test sets.
    This ensures that the real and synthetic versions of the same video
    stay in the same split, preventing data leakage.
    """
    random.seed(seed)
    pairs_copy = data_pairs.copy()
    random.shuffle(pairs_copy)

    n = len(pairs_copy)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_pairs = pairs_copy[:n_train]
    val_pairs = pairs_copy[n_train : n_train + n_val]
    test_pairs = pairs_copy[n_train + n_val :]

    return train_pairs, val_pairs, test_pairs
