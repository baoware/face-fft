import argparse
import os
from pathlib import Path

import torch
import torchvision.io as io
from tqdm import tqdm

from face_fft.data.generate import preprocess_video_tensor

ALLOWED_EXTS = {".mp4", ".avi", ".mov"}


def main():
    parser = argparse.ArgumentParser(
        description="Ingest pre-existing real/synthetic video pairs into .pt format."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="src/face_fft/data/raw_video_pairs",
        help=(
            "Root directory containing real/ and synthetic/ subdirectories. "
            "Videos are matched by filename stem."
        ),
    )
    parser.add_argument(
        "--real_out_dir",
        type=str,
        default="src/face_fft/data/real",
        help="Output directory for real .pt tensors.",
    )
    parser.add_argument(
        "--synth_out_dir",
        type=str,
        default=None,
        help=(
            "Output directory for synthetic .pt tensors. "
            "Defaults to src/face_fft/data/synth_{dataset_name}."
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help=(
            "Tag identifying this dataset (e.g. 'faceforensics', 'dfdc'). "
            "Used in output filenames: {stem}_{dataset_name}.pt"
        ),
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=16,
        help="Target number of frames per video.",
    )
    parser.add_argument(
        "--target_size",
        type=int,
        default=256,
        help="Target spatial resolution (square crop).",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    real_in_dir = input_dir / "real"
    synth_in_dir = input_dir / "synthetic"

    real_out_dir = Path(args.real_out_dir)
    synth_out_dir = (
        Path(args.synth_out_dir)
        if args.synth_out_dir
        else Path(f"src/face_fft/data/synth_{args.dataset_name}")
    )

    os.makedirs(real_out_dir, exist_ok=True)
    os.makedirs(synth_out_dir, exist_ok=True)

    real_videos = [f for f in real_in_dir.glob("*") if f.suffix.lower() in ALLOWED_EXTS]

    if not real_videos:
        print(f"No video files found in {real_in_dir}")
        return

    # Build a stem → path index for synthetic videos
    synth_index = {
        f.stem: f for f in synth_in_dir.glob("*") if f.suffix.lower() in ALLOWED_EXTS
    }

    print(
        f"Found {len(real_videos)} real video(s). "
        f"{len(synth_index)} synthetic video(s) indexed."
    )

    processed = 0
    skipped_unmatched = 0
    skipped_existing = 0

    for real_path in tqdm(sorted(real_videos)):
        stem = real_path.stem
        out_filename = f"{stem}_{args.dataset_name}.pt"
        real_out_path = real_out_dir / out_filename
        synth_out_path = synth_out_dir / out_filename

        if stem not in synth_index:
            print(f"WARNING: No matching synthetic video for '{stem}' — skipping.")
            skipped_unmatched += 1
            continue

        if real_out_path.exists() and synth_out_path.exists():
            skipped_existing += 1
            continue

        synth_path = synth_index[stem]

        try:
            real_raw, _, _ = io.read_video(str(real_path), pts_unit="sec")
            synth_raw, _, _ = io.read_video(str(synth_path), pts_unit="sec")

            if real_raw.shape[0] == 0:
                raise ValueError(f"Real video {real_path} yielded 0 frames")
            if synth_raw.shape[0] == 0:
                raise ValueError(f"Synthetic video {synth_path} yielded 0 frames")

            target_size = (args.target_size, args.target_size)
            real_tensor = preprocess_video_tensor(
                real_raw, target_size=target_size, num_frames=args.num_frames
            )
            synth_tensor = preprocess_video_tensor(
                synth_raw, target_size=target_size, num_frames=args.num_frames
            )

            torch.save(real_tensor, real_out_path)
            torch.save(synth_tensor, synth_out_path)
            processed += 1

        except Exception as e:
            import traceback

            print(f"ERROR processing '{stem}': {type(e).__name__}: {e}")
            traceback.print_exc()

    print(
        f"\nIngestion complete: {processed} pair(s) saved, "
        f"{skipped_existing} skipped (already exist), "
        f"{skipped_unmatched} skipped (no matching synthetic)."
    )


if __name__ == "__main__":
    main()
