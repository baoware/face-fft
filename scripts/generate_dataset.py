import argparse
import os
from pathlib import Path
from tqdm import tqdm
import torch
import torchvision.io as io
import numpy as np

from face_fft.data.generate import (
    extract_first_frame,
    preprocess_video_tensor,
    generate_synthetic_video_cogvideox,
    generate_synthetic_video_wan,
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Paired Real/Synthetic Datasets"
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        required=True,
        help="Directory containing real .mp4 videos",
    )
    parser.add_argument(
        "--real_out_dir",
        type=str,
        required=True,
        help="Output dir for real .pt tensors",
    )
    parser.add_argument(
        "--synth_out_dir",
        type=str,
        required=True,
        help="Output dir for synthetic .pt tensors",
    )
    parser.add_argument(
        "--generator",
        type=str,
        choices=["cogvideox", "wan"],
        default="cogvideox",
        help="Which generation model to use",
    )
    parser.add_argument(
        "--model_id", type=str, default=None, help="Hugging Face model ID override"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A highly realistic person's face, talking slightly, photorealistic video.",
        help="Conditioning prompt",
    )
    parser.add_argument("--num_frames", type=int, default=16, help="Target frame count")
    parser.add_argument(
        "--gen_height",
        type=int,
        default=480,
        help="Generation height in pixels. Lower values reduce VRAM and time. "
        "Output is always downsampled to 256x256 after generation.",
    )
    parser.add_argument(
        "--gen_width",
        type=int,
        default=480,
        help="Generation width in pixels. Lower values reduce VRAM and time. "
        "Output is always downsampled to 256x256 after generation.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=20,
        help="Diffusion denoising steps. Lower values are faster; 20 is a good "
        "compute/quality tradeoff for this low-resolution detection task.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help=(
            "Path to local HuggingFace model cache. "
            "Use with --local_files_only to prevent network access during HPC jobs."
        ),
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        default=False,
        help=(
            "Load models from local cache only; raise an error if not found locally. "
            "Recommended for HPC batch jobs to prevent silent downloads."
        ),
    )
    args = parser.parse_args()

    os.makedirs(args.real_out_dir, exist_ok=True)
    os.makedirs(args.synth_out_dir, exist_ok=True)

    allowed_exts = [".mp4", ".avi", ".mov"]
    source_path = Path(args.source_dir)
    videos = [f for f in source_path.iterdir() if f.suffix.lower() in allowed_exts]

    if not videos:
        print(f"No videos found in {args.source_dir}")
        return

    print(f"Found {len(videos)} videos. Generating with {args.generator}...")

    # Define the generation function and default model
    if args.generator == "cogvideox":
        gen_fn = generate_synthetic_video_cogvideox
        default_model = "THUDM/CogVideoX-2b"  # Or THUDM/CogVideoX1.5-5B-I2V
    else:
        gen_fn = generate_synthetic_video_wan
        default_model = "Wan-AI/Wan2.2-I2V-A14B"

    model_id = args.model_id if args.model_id else default_model

    for vid_path in tqdm(videos):
        vid_name = vid_path.stem
        pt_filename = f"{vid_name}_{args.generator}.pt"
        real_out_path = Path(args.real_out_dir) / pt_filename
        synth_out_path = Path(args.synth_out_dir) / pt_filename

        # Skip if already exists
        if real_out_path.exists() and synth_out_path.exists():
            continue

        try:
            # 1. Extract and process real video
            first_frame_img = extract_first_frame(str(vid_path))

            # 2. Generate synthetic frames (list of PIL.Image)
            print(f"\nGenerating synthetic for {vid_name} using {model_id}...")
            synth_frames_pil = gen_fn(
                first_frame_img,
                prompt=args.prompt,
                model_id=model_id,
                height=args.gen_height,
                width=args.gen_width,
                num_inference_steps=args.num_inference_steps,
                cache_dir=args.cache_dir,
                local_files_only=args.local_files_only,
            )

            # Convert synth frames back to Tensor (T, H, W, C) [0, 255]
            synth_frames_np = [
                torch.from_numpy(np.array(img)) for img in synth_frames_pil
            ]
            synth_tensor_raw = torch.stack(synth_frames_np, dim=0)

            # Read real video fully to match length
            real_tensor_raw, _, _ = io.read_video(str(vid_path), pts_unit="sec")

            # 3. Preprocess both to identical target dimensions (C, T, 256, 256)
            real_tensor = preprocess_video_tensor(
                real_tensor_raw, target_size=(256, 256), num_frames=args.num_frames
            )
            synth_tensor = preprocess_video_tensor(
                synth_tensor_raw, target_size=(256, 256), num_frames=args.num_frames
            )

            # 4. Save to paired directories
            torch.save(real_tensor, real_out_path)
            torch.save(synth_tensor, synth_out_path)

        except Exception as e:
            print(f"Failed processing {vid_name}: {e}")

    print("Generation pipeline complete.")


if __name__ == "__main__":
    main()
