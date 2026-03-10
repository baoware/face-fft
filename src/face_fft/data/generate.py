import torch
from diffusers import CogVideoXImageToVideoPipeline, WanImageToVideoPipeline
import torchvision.io as io
import torchvision.transforms.functional as TF
from PIL import Image
import gc


def extract_first_frame(video_path: str) -> Image.Image:
    """
    Extracts the first frame from an mp4/avi video file.
    """
    video, _, _ = io.read_video(video_path, pts_unit="sec", end_pts=0.1)
    if len(video) == 0:
        # Fallback to reading the whole thing if pts fails
        video, _, _ = io.read_video(video_path, pts_unit="sec")
        if len(video) == 0:
            raise ValueError(f"Video {video_path} is empty or unable to read")

    # Video is (T, H, W, C) in [0, 255]
    frame = video[0].numpy()
    return Image.fromarray(frame)


def preprocess_video_tensor(
    video: torch.Tensor, target_size=(256, 256), num_frames: int = 16
) -> torch.Tensor:
    """
    Takes a video tensor and resizes/crops to match the specified dimensions.
    Assumes video shape is (T, H, W, C).
    Outputs (C, T, target_size[0], target_size[1]) in [0, 1].
    """
    # Truncate or pad frames to match num_frames
    T = video.size(0)
    if T > num_frames:
        video = video[:num_frames]
    elif T < num_frames:
        pad_len = num_frames - T
        pad_frames = video[-1:].repeat(pad_len, 1, 1, 1)
        video = torch.cat([video, pad_frames], dim=0)

    # Convert to (T, C, H, W) and normalize
    video = video.permute(0, 3, 1, 2).float() / 255.0

    # Resize and Center Crop
    T, C, H, W = video.shape
    resized_frames = []
    for t in range(T):
        frame = video[t]
        # Resize shorter edge to target size max
        frame = TF.resize(frame, max(target_size), antialias=True)
        # Center crop
        frame = TF.center_crop(frame, target_size)
        resized_frames.append(frame)

    out_video = torch.stack(resized_frames, dim=0)  # (T, C, H, W)
    out_video = out_video.permute(1, 0, 2, 3)  # (C, T, H, W)

    return out_video


def generate_synthetic_video_cogvideox(
    image: Image.Image, prompt: str, model_id: str = "THUDM/CogVideoX-2b"
):
    """
    Generates a synthetic video using CogVideoX 2B.
    """
    pipe = CogVideoXImageToVideoPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")

    frames = pipe(
        image=image, prompt=prompt, num_inference_steps=50, guidance_scale=6.0
    ).frames[0]

    del pipe
    torch.cuda.empty_cache()
    gc.collect()

    return frames


def generate_synthetic_video_wan(
    image: Image.Image, prompt: str, model_id: str = "Wan-AI/Wan2.1-I2V"
):
    """
    Generates a synthetic video using WanImageToVideoPipeline.
    """
    pipe = WanImageToVideoPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    frames = pipe(image=image, prompt=prompt, num_inference_steps=50).frames[0]

    del pipe
    torch.cuda.empty_cache()
    gc.collect()

    return frames
