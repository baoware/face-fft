"""Codec control: is the periodic temporal spectral peak from the generator or from H.264?

The pilot found a period-4 temporal peak in several generators, but H.264 encodes on a
4-frame GOP-like cadence, so the two explanations are confounded. This script separates
them by holding the frames fixed and varying only the codec:

    synth_lossless   generated frames -> tensor            (no codec at all)
    synth_h264       the SAME frames  -> libx264 -> decode (codec only difference)
    real_decoded     source video as it ships              (already H.264)
    real_h264        source frames re-encoded              (adds a second H.264 pass)

Caveat: only synth_lossless is genuinely codec-free. The Pexels sources ship as 4K
H.264, so real_decoded has already been through an encoder and no true lossless real
reference exists in DeepAction. The real_* pair therefore measures what a SECOND H.264
pass adds, not what the first one did.

Decision rule:
  * peak present in synth_lossless            -> generator artifact, hypothesis survives
  * peak only after H.264, absent in lossless -> codec artifact, hypothesis falsified
  * peak also in real_h264 but not real_decoded -> H.264 manufactures it outright

The measurement marginalises the project's existing 3D FFT onto the temporal axis rather
than collapsing to a per-frame scalar, so the spectral framing is preserved.
"""

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
import cv2
import torchvision.io as io
from PIL import Image
from tqdm import tqdm

from face_fft.data.generate import extract_first_frame, preprocess_video_tensor


def temporal_power_spectrum(video: torch.Tensor) -> np.ndarray:
    """Temporal marginal of the 3D FFT power spectrum.

    Args:
        video: (C, T, H, W) float tensor.

    Returns:
        P[k] for k = 0..T-1, where bin k is the temporal frequency k/T cycles/frame.
        Power is summed over spatial frequencies and averaged over channels, so a
        periodicity shared across the frame accumulates instead of being averaged away.
    """
    x = video.to(torch.float32)
    x = x - x.mean()  # drop the DC pedestal so it cannot leak into low bins
    freq = torch.fft.fftn(x, dim=(-3, -2, -1))
    power = (freq.real**2 + freq.imag**2)
    # sum over spatial frequency, mean over channels -> one value per temporal bin
    return power.sum(dim=(-2, -1)).mean(dim=0).cpu().numpy()


def peak_prominence_db(P: np.ndarray, period: int, half_window: int = 5) -> float:
    """Prominence of the period-`period` temporal peak over a LOCAL baseline, in dB.

    The baseline is the median of neighbouring bins within +-half_window, excluding the
    peak and its immediate shoulders. A global-median baseline does not work here: video
    spectra fall off as ~1/f, so any long period scores positive purely from the slope.
    That bug made every generator AND real video look like it had a large peak at p12/p16
    (+11 to +14 dB) while discriminating nothing. Local baselining removes the slope.
    """
    T = len(P)
    k0 = T // period
    if k0 < 2 or k0 >= T // 2:
        return float("nan")

    lo, hi = max(1, k0 - half_window), min(T // 2, k0 + half_window + 1)
    neighbourhood = [k for k in range(lo, hi) if abs(k - k0) > 1]
    if len(neighbourhood) < 3:
        return float("nan")

    baseline = np.median(P[neighbourhood])
    if baseline <= 0:
        return float("nan")
    return float(10.0 * np.log10(P[k0] / baseline))


def read_frames_bounded(path: str, n: int, size: int) -> torch.Tensor:
    """Read at most `n` frames, resizing each during decode. Returns (T,H,W,C) uint8.

    torchvision.io.read_video decodes the WHOLE file into RAM. The Pexels sources are
    4K and hundreds of frames -- 22 GB for a single clip -- which OOM-killed the first
    attempt at 67 GB RSS. Resizing per frame keeps this at a few MB.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"OpenCV cannot open {path}")
    frames = []
    while len(frames) < n:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise ValueError(f"No frames read from {path}")
    return torch.from_numpy(np.array(frames)).to(torch.uint8)


def resize_frames(frames_u8: torch.Tensor, size: int) -> torch.Tensor:
    """Resize (T,H,W,C) uint8 frames to (T,size,size,C) uint8."""
    out = [
        cv2.resize(f.numpy(), (size, size), interpolation=cv2.INTER_AREA)
        for f in frames_u8
    ]
    return torch.from_numpy(np.array(out)).to(torch.uint8)


def h264_roundtrip(frames_uint8: torch.Tensor, fps: int, crf: int) -> torch.Tensor:
    """Encode (T,H,W,C) uint8 to H.264 and decode it back, returning (T,H,W,C) uint8."""
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "clip.mp4")
        io.write_video(
            path,
            frames_uint8,
            fps=fps,
            video_codec="libx264",
            options={"crf": str(crf), "pix_fmt": "yuv420p"},
        )
        decoded, _, _ = io.read_video(path, pts_unit="sec")
    return decoded


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source_dir", required=True, help="Directory of real .mp4 source videos")
    ap.add_argument("--out_dir", required=True, help="Where to write tensors and results")
    ap.add_argument("--num_videos", type=int, default=50)
    ap.add_argument("--num_frames", type=int, default=48, help="Divisible by --period for a clean bin")
    ap.add_argument("--period", type=int, default=4, help="Latent temporal compression stride under test")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--size", type=int, default=256, help="Working resolution; codec is applied here")
    ap.add_argument("--crf", type=int, default=23, help="x264 quality; 23 is the default consumer setting")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt", type=str,
                    default="A highly realistic person's face, talking slightly, photorealistic video.")
    ap.add_argument("--model_id", type=str, default="zai-org/CogVideoX-5b-I2V")
    ap.add_argument("--num_inference_steps", type=int, default=20)
    ap.add_argument("--cache_dir", type=str, default=None)
    ap.add_argument("--local_files_only", action="store_true")
    args = ap.parse_args()

    if args.num_frames % args.period != 0:
        raise ValueError(f"--num_frames {args.num_frames} must be divisible by --period {args.period}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out = Path(args.out_dir)
    (out / "tensors").mkdir(parents=True, exist_ok=True)

    videos = sorted(
        f for f in Path(args.source_dir).rglob("*")
        if f.is_file() and f.suffix.lower() in (".mp4", ".avi", ".mov")
    )[: args.num_videos]
    if not videos:
        raise SystemExit(f"No source videos under {args.source_dir}")
    print(f"Using {len(videos)} source videos from {args.source_dir}", flush=True)

    # Load the diffusion pipeline ONCE. The helper in data/generate.py reloads it per
    # video (~15 GB each time), which is untenable for a sweep of this size.
    from diffusers import CogVideoXImageToVideoPipeline

    print(f"Loading {args.model_id} ...", flush=True)
    pipe = CogVideoXImageToVideoPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    # A 40 GB A100 cannot hold this pipeline plus a 49-frame VAE decode: the first
    # attempt OOMed on every video inside autoencoder_kl_cogvideox. Offload idle
    # components to CPU and decode the VAE in slices/tiles instead of one shot.
    # Do NOT also call .to("cuda") -- that defeats the offload hooks.
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    print("Pipeline ready (cpu offload + vae slicing/tiling).", flush=True)

    conditions = ["synth_lossless", "synth_h264", "real_decoded", "real_h264"]
    rows = []

    for i, vid in enumerate(tqdm(videos, desc="videos")):
        try:
            first = extract_first_frame(str(vid))

            # Per-video seed derived from the run seed: reproducible, and independent
            # across videos so one bad draw cannot correlate the whole sweep.
            gen = torch.Generator(device="cpu").manual_seed(args.seed + i)
            frames = pipe(
                image=first,
                prompt=args.prompt,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=6.0,
                use_dynamic_cfg=True,
                generator=gen,
            ).frames[0]

            synth_u8 = torch.stack([torch.from_numpy(np.array(f)) for f in frames], dim=0)
            if synth_u8.shape[0] < args.num_frames:
                print(f"  {vid.stem}: generator returned {synth_u8.shape[0]} frames "
                      f"(< {args.num_frames}), padding applies", flush=True)
            # Resize to the working resolution BEFORE any encoding, so every condition
            # sees the codec at the same resolution. Otherwise synth (720x480) and real
            # (4K) would be encoded at different scales and the comparison across the
            # real/synth pair would confound resolution with codec.
            synth_u8 = resize_frames(synth_u8[: args.num_frames].to(torch.uint8), args.size)

            real_u8 = read_frames_bounded(str(vid), args.num_frames, args.size)

            variants = {
                "synth_lossless": synth_u8,
                "synth_h264": h264_roundtrip(synth_u8, args.fps, args.crf),
                "real_decoded": real_u8,
                "real_h264": h264_roundtrip(real_u8, args.fps, args.crf),
            }

            # DeepAction names every source a.mp4, so the stem alone collides: the first
            # full run silently overwrote each video's saved tensors with the next one.
            uid = f"{i:03d}_{vid.parent.name}_{vid.stem}"
            rec = {"video": uid, "source_path": str(vid), "spectra": {}}
            for name, u8 in variants.items():
                # identical preprocessing for every condition: the codec is the only variable
                t = preprocess_video_tensor(u8, target_size=(args.size, args.size), num_frames=args.num_frames)
                P = temporal_power_spectrum(t)
                rec[name] = peak_prominence_db(P, args.period)
                # keep the full spectrum so other periods can be checked later: a gap
                # that appears at every period is spectral slope, not a peak
                rec["spectra"][name] = [float(v) for v in P]
                torch.save(
                    {"video": t, "temporal_power": P, "condition": name},
                    out / "tensors" / f"{uid}_{name}.pt",
                )
            rows.append(rec)
            print(f"  {uid}: " + "  ".join(f"{c}={rec[c]:+.2f}dB" for c in conditions), flush=True)

        except Exception as e:
            import traceback
            print(f"FAILED {vid.stem}: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()

    if not rows:
        raise SystemExit("No videos processed successfully.")

    with open(out / "results.json", "w") as f:
        json.dump({"args": vars(args), "rows": rows}, f, indent=2)

    print(f"\nPeriod-{args.period} temporal peak prominence, n={len(rows)} videos")
    print(f"| {'Condition':<16} | {'Mean dB':>8} | {'Median dB':>9} | {'Std':>6} | {'>1dB':>5} |")
    print(f"|{'-'*18}|{'-'*10}|{'-'*11}|{'-'*8}|{'-'*7}|")
    for c in conditions:
        v = np.array([r[c] for r in rows if not np.isnan(r[c])])
        if len(v) == 0:
            continue
        print(f"| {c:<16} | {v.mean():+8.2f} | {np.median(v):+9.2f} | {v.std():6.2f} | "
              f"{100*np.mean(v > 1.0):4.0f}% |")

    lossless = np.array([r["synth_lossless"] for r in rows if not np.isnan(r["synth_lossless"])])
    coded = np.array([r["synth_h264"] for r in rows if not np.isnan(r["synth_h264"])])
    print(f"\nsynth_h264 - synth_lossless = {coded.mean() - lossless.mean():+.2f} dB "
          f"(codec contribution, same frames)")
    print("Hypothesis survives if synth_lossless is clearly > 0 dB on its own.")


if __name__ == "__main__":
    main()
