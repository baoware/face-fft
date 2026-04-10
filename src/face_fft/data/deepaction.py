import os
import random
from pathlib import Path
from typing import List, Tuple, Dict, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F_t

class DeepActionDataset(Dataset):
    """
    PyTorch Dataset for loading DeepAction v1 videos on-the-fly.
    Uses OpenCV to memory-efficiently extract only the target frames.
    Outputs: (video_tensor, label) where video_tensor is (C, T, H, W).
    """
    def __init__(
        self,
        samples: List[Tuple[str, int]],
        target_frames: int = 8,
        target_size: Tuple[int, int] = (256, 256)
    ):
        self.samples = samples
        self.target_frames = target_frames
        self.target_size = target_size

    def __len__(self) -> int:
        return len(self.samples)

    def _extract_frames_cv2(self, video_path: str) -> torch.Tensor:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"OpenCV cannot open {video_path}")

        frames =[]
        # sequentially read all frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # resize immediately to save RAM and ensure consistent shapes
            # cv2.resize expects (width, height)
            frame = cv2.resize(frame, (self.target_size[1], self.target_size[0]))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            
        cap.release()

        total_read = len(frames)
        if total_read == 0:
            raise ValueError("No frames could be read from the video.")

        # subsample down to the target 8 frames
        indices = np.linspace(0, total_read - 1, self.target_frames).astype(int)
        sampled_frames = [frames[i] for i in indices]

        # convert to tensor
        v = torch.from_numpy(np.array(sampled_frames))
        
        # permute to (C, T, H, W)
        v = v.permute(3, 0, 1, 2)
        
        # normalize to float[0.0, 1.0]
        v = v.float() / 255.0
        
        return v

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        video_path, label = self.samples[idx]
        
        try:
            v = self._extract_frames_cv2(video_path)
        except Exception as e:
            print(f"Warning: Failed to load {video_path}. Error: {e}")
            # Fallback for corrupted videos
            v = torch.zeros((3, self.target_frames, self.target_size[0], self.target_size[1]))
            
        return v, label


def get_deepaction_splits(
        root_dir: str,
        synth_models: List[str] =["CogVideoX5B"],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        target_frames: int = 8,
        target_size: Tuple[int, int] = (256, 256),
        seed: int = 42
    ) -> Tuple[DeepActionDataset, DeepActionDataset, DeepActionDataset]:

    root_path = Path(root_dir)
    real_dir = root_path / "Pexels"
    
    if not real_dir.exists():
        raise FileNotFoundError(f"Real video directory not found at {real_dir}")

    action_classes =[d.name for d in real_dir.iterdir() if d.is_dir()]
    paired_data =[]
    
    for action in action_classes:
        real_class_dir = real_dir / action
        real_vids = sorted(list(real_class_dir.glob("*.mp4")))
        
        for r_vid in real_vids:
            synth_vids =[]
            for sm in synth_models:
                s_vid = root_path / sm / action / r_vid.name
                if s_vid.exists():
                    synth_vids.append(str(s_vid))
            
            if synth_vids:
                paired_data.append({"real": str(r_vid), "synths": synth_vids})
                
    if not paired_data:
        raise ValueError("No valid paired videos found. Check directory structure.")

    random.seed(seed)
    random.shuffle(paired_data)

    n = len(paired_data)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_pairs = paired_data[:n_train]
    val_pairs = paired_data[n_train:n_train + n_val]
    test_pairs = paired_data[n_train + n_val:]
    
    def flatten_pairs(pairs: List[Dict]) -> List[Tuple[str, int]]:
        samples =[]
        for p in pairs:
            # for every 1 synthetic video, add the real video once
            for s in p["synths"]:
                samples.append((p["real"], 0))  # 0 = Real
                samples.append((s, 1))          # 1 = Synthetic
        return samples

    train_dataset = DeepActionDataset(flatten_pairs(train_pairs), target_frames, target_size)
    val_dataset = DeepActionDataset(flatten_pairs(val_pairs), target_frames, target_size)
    test_dataset = DeepActionDataset(flatten_pairs(test_pairs), target_frames, target_size)

    print(f"Dataset Splits Created: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset