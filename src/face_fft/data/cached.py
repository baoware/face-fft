"""Dataset over the format-normalised clip cache written by bin/build_cache.py.

Each shard is <pair>__<source>__<split>.npy of shape (N, 16, H, W, 3) uint8 with a
matching .csv index. Rows shorter than T (n_valid < T) are dropped rather than
padded, so clip length can never leak into the input.
"""

import csv
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


class CachedClipDataset(Dataset):
    """Clips as (C, T, H, W) float in [0, 1], label 0 = real, 1 = synthetic.

    Args:
        cache_dir: directory of shards.
        pairs: which pairs to include ("Pair1", "Pair2", "DeepAction", "6.7m").
        split: "train", "val" or "test".
        T: frames per clip (the first T of the cached 16).
        real_strides: which frame-dropping variants of REAL clips to include.
            (1, 2, 3, 4) spans the fakes' frame rates; (1,) turns the augmentation off.
        exclude_sources: sources to drop entirely (leave-one-generator-out).
    """

    def __init__(self, cache_dir: str | Path, pairs: Iterable[str], split: str, T: int,
                 real_strides: Iterable[int] = (1, 2, 3, 4), exclude_sources: Iterable[str] = ()):
        self.cache_dir = Path(cache_dir)
        self.T = T
        pairs, real_strides, exclude = set(pairs), set(int(s) for s in real_strides), set(exclude_sources)
        self.rows = []
        for idx_path in sorted(self.cache_dir.glob("*.csv")):
            pair, source, sp = idx_path.stem.split("__")
            if pair not in pairs or sp != split or source in exclude:
                continue
            for r in csv.DictReader(open(idx_path)):
                label, stride = int(r["label"]), int(r["stride"])
                if int(r["n_valid"]) < T or (label == 0 and stride not in real_strides):
                    continue
                self.rows.append(dict(shard=idx_path.stem, row=int(r["row"]), label=label, pair=pair,
                                      source=source, group=r["group"], stride=stride))
        self._arrays: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def _array(self, shard: str) -> np.ndarray:
        a = self._arrays.get(shard)          # memmap opened lazily, once per worker process
        if a is None:
            a = self._arrays[shard] = np.load(self.cache_dir / f"{shard}.npy", mmap_mode="r")
        return a

    def __getitem__(self, i: int):
        r = self.rows[i]
        clip = np.array(self._array(r["shard"])[r["row"], : self.T])   # copy: memmap slices are read-only
        x = torch.from_numpy(clip).permute(3, 0, 1, 2).float().div_(255.0)
        return x, r["label"]

    def balanced_weights(self) -> list[float]:
        """Sampling weights: half the probability mass to each label, split equally
        across the sources within that label. Without this, fakes outnumber reals
        2.7:1 (Pair1) and 4:1 (Pair2), and the largest generators dominate."""
        per_source = Counter((r["label"], r["source"]) for r in self.rows)
        sources_per_label = Counter(label for label, _ in per_source)
        return [0.5 / sources_per_label[r["label"]] / per_source[(r["label"], r["source"])] for r in self.rows]

    def summary(self) -> str:
        c = Counter((r["label"], r["source"]) for r in self.rows)
        return ", ".join(f"{s}({'R' if l == 0 else 'F'})={n}" for (l, s), n in sorted(c.items()))
