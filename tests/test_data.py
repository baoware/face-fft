import torch

from face_fft.data.dataset import PairedVideoDataset, split_paired_dataset


def test_paired_video_dataset(tmp_path):
    real_dir = tmp_path / "real"
    synth_dir = tmp_path / "synth"
    real_dir.mkdir()
    synth_dir.mkdir()

    # Create dummy data
    for i in range(3):
        fname = f"vid_{i}.pt"
        # Shape: (C, T, H, W) -> 3 channels, 16 frames, 256x256
        dummy_tensor = torch.zeros((3, 16, 256, 256))
        torch.save(dummy_tensor, real_dir / fname)
        # Synthetic might have some noise/difference
        torch.save(dummy_tensor + 1, synth_dir / fname)

    # Unmatched file to ensure pairing ignores it
    torch.save(torch.zeros((3, 16, 256, 256)), real_dir / "unmatched.pt")

    dataset = PairedVideoDataset.from_directories(
        real_dir, synth_dir, yield_pairs=False
    )

    # We should have 6 items (3 pairs * 2)
    assert len(dataset) == 6

    # Check the first item (Real)
    vid, label = dataset[0]
    assert vid.shape == (3, 16, 256, 256)
    assert label == 0  # 0 is real

    # Check the second item (Synthetic of the same pair)
    vid, label = dataset[1]
    assert torch.all(vid == 1)  # synth has +1
    assert label == 1  # 1 is synth


def test_paired_video_dataset_yielding_pairs(tmp_path):
    real_dir = tmp_path / "real"
    synth_dir = tmp_path / "synth"
    real_dir.mkdir()
    synth_dir.mkdir()

    for i in range(2):
        fname = f"vid_{i}.pt"
        torch.save(torch.zeros((3, 16, 256, 256)), real_dir / fname)
        torch.save(torch.ones((3, 16, 256, 256)), synth_dir / fname)

    dataset = PairedVideoDataset.from_directories(real_dir, synth_dir, yield_pairs=True)

    # Should have 2 pairs
    assert len(dataset) == 2

    # Check shape of pairs
    (real_vid, synth_vid), idx = dataset[0]
    assert real_vid.shape == (3, 16, 256, 256)
    assert synth_vid.shape == (3, 16, 256, 256)
    assert torch.all(real_vid == 0)
    assert torch.all(synth_vid == 1)
    assert idx == 0


def test_split_paired_dataset():
    dummy_pairs = [{"real": f"r_{i}", "synthetic": f"s_{i}"} for i in range(100)]

    train, val, test = split_paired_dataset(dummy_pairs, train_ratio=0.7, val_ratio=0.2)

    assert len(train) == 70
    assert len(val) == 20
    assert len(test) == 10

    # Check no leakage: intersection of pairs should be empty
    train_reals = set([p["real"] for p in train])
    val_reals = set([p["real"] for p in val])
    test_reals = set([p["real"] for p in test])

    assert len(train_reals.intersection(val_reals)) == 0
    assert len(train_reals.intersection(test_reals)) == 0
    assert len(val_reals.intersection(test_reals)) == 0
