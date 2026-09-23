"""Week-1 lossless test (SUBMISSION_PLAN.md): is the period-4 temporal peak made by
the generator or by the codec's frame-type pattern?

  generate  (GPU) N clips per model, saved as raw uint8 frames -- no codec ever:
              cogvideox  CogVideoX-5B-I2V   8x8x4 VAE   (4x temporal compression)
              wan        Wan2.1-T2V-1.3B    8x8x4 VAE   (4x temporal compression)
              svd        SVD-img2vid-xt     frame-wise  (control: no temporal compression)
            Content is matched to DeepAction: Pexels video <id> supplies the first
            frame (I2V models) and captions.csv row <id> the prompt.
  analyze   (CPU) every generated clip, plus the real Pexels clips, under
              raw    no re-encode
              intra  all I-frames                 (no inter-frame pattern)
              bf0    I then all P, no B-frames    (no periodic pattern)
              bf2    fixed IBBP...                (period 3)
              bf3    fixed IBBBP...               (period 4 -- the confound)
            at two CRFs, measured on two views (256x256 native-resolution centre crop,
            and square crop resized to 256), with two statistics, at periods 3 and 4.

The trace is real only if (plan, step 4): the period-4 peak appears in RAW output of
the 4x-temporal models, is absent from the frame-wise model, and does not move when the
codec's period changes (bf2 should not pull it to period 3; bf3 should not create it in
real video or SVD).

Statistics:
  spectral   temporal marginal of the 3D FFT power, prominence over a local median
  framediff  spectrum of per-frame mean squared frame difference -- a RECONSTRUCTION of
             the pilot's "frame-difference energy" statistic (scripts/pilot/ was not
             available); the pilot's own numbers should be compared only after running
             its code on these clips.
"""

import argparse
import csv
import io
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import av
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from codec_control import peak_prominence_db  # validated local-median prominence

# Frame-type settings go through x264-params: FFmpeg silently ignores "b-adapt" as a
# plain option, and x264's adaptive B-frame decision then picked IBBP on fast-moving
# generated clips when IBBBP was requested -- the wrong codec period for this test.
CODECS = {
    "intra": "keyint=1:bframes=0",
    "bf0":   "keyint=250:bframes=0:scenecut=0",
    "bf2":   "keyint=250:bframes=2:b-adapt=0:b-pyramid=none:scenecut=0",
    "bf3":   "keyint=250:bframes=3:b-adapt=0:b-pyramid=none:scenecut=0",
}
EXPECTED = {"intra": lambda n: "I" * n, "bf0": lambda n: "I" + "P" * (n - 1),
            "bf2": lambda n: ("I" + "BBP" * n)[:n], "bf3": lambda n: ("I" + "BBBP" * n)[:n]}
PAD = 8   # extra frames encoded then discarded, so x264's short final mini-GOP falls outside the window


# ----------------------------------------------------------------------------- generate
def pexels_items(da_root: Path, n: int):
    caps = {}
    for row in csv.reader(open(da_root / "captions.csv")):
        if row and row[0].isdigit():
            caps[row[0]] = row[1]
    items = []
    for d in sorted((p for p in (da_root / "Pexels").iterdir() if p.is_dir() and p.name.isdigit()),
                    key=lambda p: int(p.name)):   # skip macOS .DS_Store etc.
        vids = sorted(d.glob("*.mp4"))
        if vids and d.name in caps:
            items.append((d.name, vids[0], caps[d.name]))
    return items[:n]


def first_frame(path: Path):
    from PIL import Image
    cap = cv2.VideoCapture(str(path)); ok, f = cap.read(); cap.release()
    if not ok:
        raise ValueError(f"cannot read {path}")
    return Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))


def to_uint8(frames) -> np.ndarray:
    if isinstance(frames, np.ndarray):
        a = frames
    else:
        a = np.stack([np.asarray(f) for f in frames])
    if a.dtype != np.uint8:
        a = np.clip(np.round(a * 255.0), 0, 255).astype(np.uint8)
    return a


def generate(args):
    import torch
    out = Path(args.out) / args.model
    out.mkdir(parents=True, exist_ok=True)
    items = pexels_items(Path(args.da_root), args.n)
    print(f"{args.model}: {len(items)} clips -> {out}", flush=True)

    if args.model == "cogvideox":
        from diffusers import CogVideoXImageToVideoPipeline
        pipe = CogVideoXImageToVideoPipeline.from_pretrained("zai-org/CogVideoX-5b-I2V", torch_dtype=torch.bfloat16)
        pipe.enable_model_cpu_offload(); pipe.vae.enable_tiling()   # spatial tiles only; time untouched
        fps = 8
        run = lambda img, prompt, g: pipe(image=img, prompt=prompt, num_inference_steps=args.steps or 50,
                                         guidance_scale=6.0, use_dynamic_cfg=True, generator=g).frames[0]
    elif args.model == "wan":
        from diffusers import AutoencoderKLWan, WanPipeline
        mid = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
        vae = AutoencoderKLWan.from_pretrained(mid, subfolder="vae", torch_dtype=torch.float32)
        pipe = WanPipeline.from_pretrained(mid, vae=vae, torch_dtype=torch.bfloat16).to("cuda")
        fps = 16
        run = lambda img, prompt, g: pipe(prompt=prompt, height=480, width=832, num_frames=81,
                                         guidance_scale=5.0, num_inference_steps=args.steps or 50,
                                         generator=g).frames[0]
    elif args.model == "svd":
        from diffusers import StableVideoDiffusionPipeline
        pipe = StableVideoDiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt",
                                                            torch_dtype=torch.float16, variant="fp16").to("cuda")
        fps = 7
        # decode all 25 frames in ONE chunk: a chunk size of k would put decoder seams
        # every k frames -- a period-k artifact made by us, not by the generator
        run = lambda img, prompt, g: pipe(img.resize((1024, 576)), num_frames=25, decode_chunk_size=25,
                                         num_inference_steps=args.steps or 25, generator=g).frames[0]
    else:
        raise ValueError(args.model)

    for i, (pid, vid, prompt) in enumerate(items):
        dst = out / f"{pid}.npy"
        if dst.exists():
            continue
        t0 = time.time()
        g = torch.Generator(device="cpu").manual_seed(args.seed + int(pid))
        frames = to_uint8(run(first_frame(vid), prompt, g))
        np.save(dst, frames)
        json.dump(dict(model=args.model, pexels_id=pid, source=str(vid), prompt=prompt, fps=fps,
                       seed=args.seed + int(pid), shape=list(frames.shape)), open(out / f"{pid}.json", "w"))
        print(f"  [{i + 1}/{len(items)}] {pid}: {frames.shape} {time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------------------ analyze
def encode(frames: np.ndarray, x264_params: str, crf: int) -> tuple[np.ndarray, str]:
    """H.264 round trip at the clip's own resolution; returns the first len(frames)
    decoded frames and their frame-type string."""
    n = len(frames)
    h, w = frames.shape[1] // 2 * 2, frames.shape[2] // 2 * 2       # yuv420 needs even dims
    padded = np.concatenate([frames, np.repeat(frames[-1:], PAD, axis=0)])
    buf = io.BytesIO()
    with av.open(buf, "w", format="mp4") as c:
        st = c.add_stream("libx264", rate=8)
        st.width, st.height, st.pix_fmt = w, h, "yuv420p"
        st.options = {"preset": "medium", "crf": str(crf), "x264-params": x264_params}
        for f in padded:
            for p in st.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(f[:h, :w]), format="rgb24")):
                c.mux(p)
        for p in st.encode():
            c.mux(p)
    buf.seek(0)
    out, types = [], []
    with av.open(buf) as c:
        for fr in c.decode(video=0):
            out.append(fr.to_ndarray(format="rgb24"))
            types.append({1: "I", 2: "P", 3: "B"}.get(int(fr.pict_type), "?"))
    return np.stack(out[:n]), "".join(types[:n])


def views(frames: np.ndarray, size: int = 256) -> dict:
    t, h, w, _ = frames.shape
    y0, x0 = (h - size) // 2, (w - size) // 2
    crop = frames[:, max(y0, 0):max(y0, 0) + size, max(x0, 0):max(x0, 0) + size]
    s = min(h, w)
    sq = frames[:, (h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
    resized = np.stack([cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA) for f in sq])
    return {"crop": crop, "resize": resized}


def spectral_stat(v: np.ndarray, T: int) -> np.ndarray:
    x = v[:T].astype(np.float32).mean(axis=-1)                       # luma proxy (T,H,W)
    x -= x.mean()
    F = np.fft.fftn(x)
    return (np.abs(F) ** 2).sum(axis=(1, 2))                         # P[k], k = 0..T-1


def framediff_stat(v: np.ndarray, T: int) -> np.ndarray:
    x = v[:T + 1].astype(np.float32).mean(axis=-1)
    d = ((x[1:] - x[:-1]) ** 2).mean(axis=(1, 2))                   # T values
    d -= d.mean()
    return np.abs(np.fft.fft(d)) ** 2


def analyze_clip(job):
    model, clip_id, path, T, crfs = job
    try:
        if path.endswith(".npy"):
            raw = np.load(path)
        else:
            cap = cv2.VideoCapture(path); fr = []
            while len(fr) < T + 1:
                ok, f = cap.read()
                if not ok:
                    break
                fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            cap.release(); raw = np.stack(fr)
        raw = raw[:T + 1]
        if len(raw) < T + 1:
            return [], f"{model}/{clip_id}: only {len(raw)} frames"
        conds = [("raw", None, raw, "")]
        for name, opts in CODECS.items():
            for crf in crfs:
                dec, types = encode(raw, opts, crf)
                if types != EXPECTED[name](len(raw)):
                    return [], f"{model}/{clip_id}: {name} crf{crf} produced {types}, expected {EXPECTED[name](len(raw))}"
                conds.append((name, crf, dec, types))
        rows = []
        for cname, crf, frames, types in conds:
            for vname, v in views(frames).items():
                rec = dict(model=model, clip=clip_id, codec=cname, crf=crf if crf is not None else "",
                           view=vname, T=T, frame_types=types)
                for sname, fn in (("spectral", spectral_stat), ("framediff", framediff_stat)):
                    P = fn(v, T)
                    for p in (3, 4):
                        rec[f"{sname}_p{p}"] = peak_prominence_db(P, p)
                rows.append(rec)
        return rows, None
    except Exception as e:
        return [], f"{model}/{clip_id}: {type(e).__name__}: {e}"


def analyze(args):
    gen_root = Path(args.out)
    crfs = [int(c) for c in args.crfs.split(",")]
    jobs = []
    for model in ("cogvideox", "wan", "svd"):
        for f in sorted((gen_root / model).glob("*.npy")):
            jobs.append((model, f.stem, str(f), args.T, crfs))
    for pid, vid, _ in pexels_items(Path(args.da_root), args.n):
        jobs.append(("real_pexels", pid, str(vid), args.T, crfs))
    print(f"analyzing {len(jobs)} clips at T={args.T}, CRFs {crfs}", flush=True)

    rows, t0 = [], time.time()
    with Pool(args.workers) as pool:
        for i, (r, err) in enumerate(pool.imap_unordered(analyze_clip, jobs), 1):
            rows.extend(r)
            if err:
                print("  skip", err, flush=True)
            if i % 25 == 0:
                print(f"  {i}/{len(jobs)} clips  {time.time() - t0:.0f}s", flush=True)

    out_csv = Path(args.results) / f"lossless_test_T{args.T}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # frame-type patterns actually produced (verify the codec did what we asked)
    print("\nframe types actually encoded (first clip per setting):")
    seen = set()
    for r in rows:
        k = (r["codec"], r["crf"])
        if r["codec"] != "raw" and k not in seen:
            seen.add(k); print(f"  {r['codec']:6s} crf {r['crf']}: {r['frame_types']}")

    lines = [f"Lossless test, T={args.T}. Mean prominence in dB (+-SEM over clips). "
             f"Period 4 = generator/IBBBP period; period 3 = IBBP period.", ""]
    for view in ("crop", "resize"):
        for stat in ("spectral", "framediff"):
            lines.append(f"== view={view}  statistic={stat}")
            lines.append(f"{'model':<13}{'codec':<7}{'crf':>4}{'n':>5}{'period 4':>16}{'period 3':>16}")
            for model in ("real_pexels", "svd", "cogvideox", "wan"):
                for codec in ["raw"] + list(CODECS):
                    for crf in ([""] if codec == "raw" else crfs):
                        sel = [r for r in rows if r["model"] == model and r["codec"] == codec
                               and str(r["crf"]) == str(crf) and r["view"] == view]
                        if not sel:
                            continue
                        cell = lambda k: (lambda a: f"{a.mean():+7.2f} +-{a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0:4.2f}")(
                            np.array([s[k] for s in sel if s[k] == s[k]]))
                        lines.append(f"{model:<13}{codec:<7}{str(crf):>4}{len(sel):>5}{cell(f'{stat}_p4'):>16}{cell(f'{stat}_p3'):>16}")
            lines.append("")
    text = "\n".join(lines)
    print("\n" + text)
    (Path(args.results) / f"lossless_test_T{args.T}_summary.txt").write_text(text + "\n")
    print(f"wrote {out_csv}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["generate", "analyze"])
    ap.add_argument("--model", choices=["cogvideox", "wan", "svd"])
    ap.add_argument("--da_root", default="src/face_fft/data/deepaction_dataset")
    ap.add_argument("--out", default="/sfs/weka/scratch/rjr6zk/face-fft/lossless_frames")
    ap.add_argument("--results", default="/standard/uva-mira-drive/3d-fft_results/lossless_test")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--steps", type=int, default=None, help="denoising steps; default = model's usual")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--T", type=int, default=24, help="frames analysed; SVD-xt yields 25 = T+1")
    ap.add_argument("--crfs", default="23,35")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    generate(args) if args.stage == "generate" else analyze(args)


if __name__ == "__main__":
    main()
