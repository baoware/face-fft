"""VAE round-trip measurement (NEXT_STEPS step 1): does the DECODER leave a mark?

A fake reaches pixels through a generator's VAE decoder. This test isolates that
decoder: real video -> encoder -> decoder, with no AI step in between. Original and
copy show the same people, motion and scene from the same source file, so anything that
differs between them was put there by the VAE.

  roundtrip (GPU) N real clips through one VAE, saved as raw uint8 frames -- no codec:
              cogvideox  CogVideoX-5B          squeezes 4 frames into 1   test
              wan        Wan 2.1-1.3B          squeezes 4 frames into 1   test
              ltx        LTX-Video             squeezes 8 frames into 1   test (other k)
              sd         SD image VAE ft-mse   one frame at a time        control
            Clips come from DeepAction's Pexels videos, the lossless test's sources.
  analyze   (CPU) per VAE, on the same clip before and after:
              1  per-frame error over time, and its cycle strength at every spacing
              2  temporal spike (dB) at every spacing  -- the lossless test's measure
              3  spatial grid spike (dB) at every pixel spacing
              4  difference maps: dB change at every frequency
              -  PSNR, as a check that the VAE ran correctly
            Nothing is tuned to one period: the curves cover every spacing and the
            summary reads off 4 and 8 frames for every VAE, control included.

Differences from bin/lossless_test.py: that test ran the full generation pipeline, whose
content differs from the real clip, and compared across clips. This one has identical
content, so the comparison is paired.

HOW TO READ A NUMBER. A k-frame repeat lifts the bin at k frames and NOT the bins beside
it. "Clear of 0" alone is not evidence, for three reasons the summary guards against:
  * anything tied to position in the clip (a worse last frame, drift) has the same phase
    in every clip and lifts every bin of the error cycle;
  * a decoder that only smooths in time bends the spectrum, which shifts every spike
    value without making a peak;
  * the error cycle is a magnitude, so it is positive (about 0.9 SEM) with no cycle.
So every value is printed beside the median of its neighbouring bins ("nbrs"), and a
mark is a value that stands out from its own neighbours, in a test VAE, at that VAE's
own stride, that the control does not also show.

WHAT EACH MEASURE CAN CLAIM. Only the paired spikes (2, 3) describe the copy by itself,
so only they are something a generated video could also carry. The error cycle (1) and
the residual spikes need the original; they show what encoder plus decoder lost, and a
cycle there says the VAE works in groups, not yet that fakes are marked.

Choices that affect the numbers:
  * Clip length is 1 + 48 frames. The VAEs encode the first frame alone, so it is
    dropped before analysis; the 48 that remain are a whole number of 4- and 8-frame
    groups, which puts each stride on exactly one FFT bin (12 and 6).
  * Each video's windows start 0-7 frames into the file, by video. Every file opens on
    a keyframe, so identical start frames would put the SOURCE codec's frame-type
    pattern (often period 4) at one phase in every clip -- indistinguishable from a
    decoder cycle. Staggering randomises the source's phase; the VAE's stays locked to
    the clip. The originals' own spike is reported too, to show what the source carries.
  * Latents are the encoder's mode, not a sample, so a round-trip is deterministic.
  * Chunking. CogVideoX normalises over each chunk it processes, so chunk seams are
    not exact. Its DECODER runs at the pipeline's default (2 latent frames = 8 frames a
    time), which is what a real CogVideoX fake goes through -- but those seams recur
    every 8 frames with a harmonic at 4, CogVideoX's own stride. --decode_latents
    reruns it with other chunk sizes (written to cogvideox-dec<N>/): a squeezing mark
    stays at 4 frames, a chunking mark moves. Its ENCODER is never run when generating,
    so it gets the whole clip in one pass rather than adding 8-frame seams of our own.
    Wan decodes one latent frame (4 frames) at a time; chunk and stride coincide there.
  * VAEs run in float32. Tiling is off -- 256x256 is below every tiling threshold.
  * Frames are consecutive at the source's own rate (25-30 fps), as in the training
    cache. That is less motion per frame than the 8-16 fps these VAEs generate at, so a
    mark that depends on motion could be understated.
  * --view picks how frames reach 256x256. NEXT_STEPS says crop; the default here is
    "resize" (centre square, area-resized, as the cache does), because a 256-pixel crop
    of a 4K frame is a small smooth patch unlike anything these VAEs were trained on.
    "crop" keeps native pixels, but then the source codec's 8/16-pixel block grid lines
    up with the bins read by the spatial table.
  * The spatial table has no clean control: SD's VAE also works on 8x8 blocks.
  * Luma is the plain channel mean, as in the lossless test; a mark with opposite sign
    in different channels would cancel.
  * Error bars are over source videos, not clips: windows cut from one video are
    averaged first (roundtrip_metrics.clustered_mean_sem). The summary uses only clips
    that every VAE has, so a half-finished VAE is never compared on different clips.
"""

import argparse
import csv
import json
import os
import socket
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from face_fft.eval import roundtrip_metrics as rm  # noqa: E402

# k = frames squeezed into one latent frame; None = frame-wise (the control)
VAES = {
    "cogvideox": dict(
        k=4,
        cls="AutoencoderKLCogVideoX",
        repo="zai-org/CogVideoX-5b-I2V",
        subfolder="vae",
    ),
    "wan": dict(
        k=4,
        cls="AutoencoderKLWan",
        repo="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        subfolder="vae",
    ),
    "ltx": dict(
        k=8, cls="AutoencoderKLLTXVideo", repo="Lightricks/LTX-Video", subfolder="vae"
    ),
    # not SVD: its decoder mixes neighbouring frames, so it is not a clean frame-wise control
    "sd": dict(
        k=None, cls="AutoencoderKL", repo="stabilityai/sd-vae-ft-mse", subfolder=None
    ),
}
TIME_PERIODS = (4, 8)  # frames; read off for every VAE so the control lines up
SPACE_PERIODS = (
    8,
    16,
    32,
)  # pixels: the VAEs' 8x8 blocks and multiples (LTX's stride is 32)
STAGGER = 8  # window starts are spread over this many frames, by video


def variant_name(vae: str, decode_latents: int | None) -> str:
    """Directory name for a VAE run; non-default decoder chunking gets its own."""
    if decode_latents is None:
        return vae
    if vae != "cogvideox":
        raise ValueError("--decode_latents only applies to cogvideox")
    return f"{vae}-dec{decode_latents}"


# -------------------------------------------------------------------- clip selection
def pexels_videos(da_root: Path) -> list[tuple[str, Path, int]]:
    """(pexels id, path, frame count) for every real DeepAction video, in id order."""
    out = []
    for d in sorted(
        (p for p in (da_root / "Pexels").iterdir() if p.is_dir() and p.name.isdigit()),
        key=lambda p: int(p.name),
    ):  # skip macOS .DS_Store etc.
        for vid in sorted(d.glob("*.mp4")):
            cap = cv2.VideoCapture(str(vid))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            out.append((d.name, vid, n))
    return out


def clip_plan(videos, n: int, frames: int) -> list[dict]:
    """Up to `n` non-overlapping `frames`-long windows, spread over as many videos as possible.

    Window 0 of every video comes first, then window 1 of every video that is long
    enough, and so on. Video i's windows start (i mod STAGGER) frames into the file --
    see the module docstring for why. The plan depends only on the video list, so every
    VAE gets the same clips.
    """
    plan, w = [], 0
    while len(plan) < n:
        fits = [
            (pid, path, i % STAGGER)
            for i, (pid, path, total) in enumerate(videos)
            if i % STAGGER + (w + 1) * frames <= total
        ]
        if not fits:
            break
        for pid, path, offset in fits[: n - len(plan)]:
            plan.append(
                dict(
                    clip=f"{pid}_{Path(path).stem}_w{w}",
                    pexels_id=pid,
                    source=str(path),
                    window=w,
                    start=offset + w * frames,
                )
            )
        w += 1
    return plan


def check_frames(frames: int, k: int | None):
    if k is not None and (frames - 1) % k != 0:
        raise ValueError(
            f"--frames must be 1 + a multiple of {k} for this VAE, got {frames}"
        )


def check_decode_latents(frames: int, n: int):
    """Reject CogVideoX decoder chunk sizes that would return the wrong number of frames.

    Its decoder upsamples a chunk 4x in time, except that a chunk with an odd number of
    latent frames keeps its first one as a single frame. Only the clip's first latent
    frame should get that treatment, so the first chunk (which takes the remainder)
    must be odd and every later chunk even -- or the whole clip is one chunk.
    """
    latents = (frames - 1) // 4 + 1
    if n == latents or (n > 0 and n % 2 == 0 and latents % n == 1):
        return
    raise ValueError(
        f"--decode_latents {n} would drop frames: for {frames} frames ({latents} latent frames) use "
        f"{latents} (one chunk) or an even number that leaves a remainder of 1"
    )


def to_view(frame: np.ndarray, size: int, view: str) -> np.ndarray:
    """One (H, W, 3) frame -> (size, size, 3)."""
    h, w = frame.shape[:2]
    if view == "crop":
        if h < size or w < size:
            raise ValueError(f"{w}x{h} frame is smaller than a {size}x{size} crop")
        y0, x0 = (h - size) // 2, (w - size) // 2
        return frame[y0 : y0 + size, x0 : x0 + size]
    if view == "resize":
        s = min(h, w)
        y0, x0 = (h - s) // 2, (w - s) // 2
        return cv2.resize(
            frame[y0 : y0 + s, x0 : x0 + s], (size, size), interpolation=cv2.INTER_AREA
        )
    raise ValueError(view)


def read_window(path: str, start: int, frames: int, size: int, view: str) -> np.ndarray:
    """`frames` consecutive frames from frame `start`, as (frames, size, size, 3) uint8 RGB.

    Frames before `start` are decoded and discarded rather than seeked past: OpenCV's
    seek lands on the nearest keyframe for some files, which would shift the window.
    """
    cap = cv2.VideoCapture(path)
    try:
        for _ in range(start):
            if not cap.grab():
                raise ValueError(f"{path}: ended before frame {start}")
        out = []
        while len(out) < frames:
            ok, f = cap.read()
            if not ok:
                raise ValueError(
                    f"{path}: only {len(out)} of {frames} frames from frame {start}"
                )
            out.append(to_view(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), size, view))
    finally:
        cap.release()
    return np.stack(out)


def _tmp_path(path: Path) -> Path:
    """A sibling temp name unique per node and process, that no results glob matches."""
    return path.with_name(f".{path.stem}.{socket.gethostname()}.{os.getpid()}.tmp")


def save_atomic(path: Path, array: np.ndarray):
    """Array tasks for different VAEs write the same originals; never expose a half file."""
    tmp = _tmp_path(path)
    with open(tmp, "wb") as f:
        np.save(f, array)
    os.replace(tmp, path)


# ------------------------------------------------------------------------- roundtrip
def to_model(frames: np.ndarray):
    """(T, H, W, 3) uint8 -> (1, 3, T, H, W) float32 in [-1, 1]."""
    import torch

    x = torch.from_numpy(frames).to(torch.float32) / 127.5 - 1.0
    return x.permute(3, 0, 1, 2).unsqueeze(0).contiguous()


def from_model(x) -> np.ndarray:
    """(1, 3, T, H, W) in [-1, 1] -> (T, H, W, 3) uint8."""
    x = ((x[0].float().clamp(-1.0, 1.0) + 1.0) * 127.5).round()
    return x.permute(1, 2, 3, 0).to("cpu").numpy().astype(np.uint8)


def load_vae(name: str, device: str, frames: int, decode_latents: int | None = None):
    import diffusers
    import torch

    spec = VAES[name]
    vae = getattr(diffusers, spec["cls"]).from_pretrained(
        spec["repo"], subfolder=spec["subfolder"], torch_dtype=torch.float32
    )
    if getattr(vae.config, "timestep_conditioning", False):
        # such decoders take a denoising timestep and may inject noise; the round-trip
        # would no longer be the deterministic encode -> decode this test assumes
        raise ValueError(
            f"{spec['repo']}: timestep-conditioned decoder is not supported"
        )
    if name == "cogvideox":
        vae.num_sample_frames_batch_size = (
            frames  # encode the clip in one pass (see docstring)
        )
        if decode_latents is not None:
            vae.num_latent_frames_batch_size = decode_latents
    return vae.to(device).eval()


def run_vae(name: str, vae, frames: np.ndarray, device: str) -> np.ndarray:
    """Encoder then decoder. No sampling, no latent scaling (it would cancel)."""
    import torch

    with torch.no_grad():
        x = to_model(frames).to(device)
        if VAES[name]["k"] is None:
            # image VAE: frames are a batch; chunking is harmless because frames are independent
            per_frame = x[0].permute(1, 0, 2, 3)
            y = torch.cat(
                [
                    vae.decode(vae.encode(c).latent_dist.mode()).sample
                    for c in per_frame.split(16)
                ]
            )
            y = y.permute(1, 0, 2, 3).unsqueeze(0)
        else:
            y = vae.decode(vae.encode(x).latent_dist.mode()).sample
    out = from_model(y)
    if out.shape != frames.shape:
        raise ValueError(
            f"{name}: decoder returned {out.shape} for input {frames.shape}"
        )
    return out


def roundtrip(args):
    import diffusers
    import torch

    spec = VAES[args.vae]
    variant = variant_name(args.vae, args.decode_latents)
    check_frames(args.frames, spec["k"])
    if args.decode_latents is not None:
        check_decode_latents(args.frames, args.decode_latents)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "no CUDA device; pass --device cpu to run on CPU anyway (slow)"
        )
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False

    root = Path(args.out) / args.view
    real_dir, copy_dir = root / "real", root / variant
    real_dir.mkdir(parents=True, exist_ok=True)
    copy_dir.mkdir(parents=True, exist_ok=True)
    plan = clip_plan(pexels_videos(Path(args.da_root)), args.n, args.frames)
    print(
        f"{variant}: {len(plan)} of {args.n} requested clips, from {len({c['source'] for c in plan})} videos "
        f"-> {copy_dir} ({args.device})",
        flush=True,
    )
    if not plan:
        raise SystemExit(f"no usable videos under {Path(args.da_root) / 'Pexels'}")
    vae = load_vae(args.vae, args.device, args.frames, args.decode_latents)
    shape = (args.frames, args.size, args.size, 3)

    done = skipped = 0
    for i, c in enumerate(plan):
        dst = copy_dir / f"{c['clip']}.npy"
        if dst.exists():
            done += 1
            continue
        t0 = time.time()
        orig_path = real_dir / f"{c['clip']}.npy"
        if orig_path.exists():
            orig = np.load(orig_path)
            if orig.shape != shape:
                raise SystemExit(
                    f"{orig_path} is {orig.shape}, this run needs {shape}: it is left over from a run "
                    f"with other --frames/--size. Use a different --out."
                )
        try:
            if not orig_path.exists():
                orig = read_window(
                    c["source"], c["start"], args.frames, args.size, args.view
                )
                save_atomic(orig_path, orig)
            copy = run_vae(args.vae, vae, orig, args.device)
        except (
            ValueError,
            RuntimeError,
            OSError,
            cv2.error,
        ) as e:  # RuntimeError covers CUDA OOM
            skipped += 1
            print(f"  skip {c['clip']}: {type(e).__name__}: {e}", flush=True)
            if args.device == "cuda":
                torch.cuda.empty_cache()
            continue
        save_atomic(dst, copy)
        done += 1
        psnr = rm.psnr_db(
            orig, copy
        )  # RGB, all frames; the summary's PSNR is luma without frame 0
        json.dump(
            dict(
                c,
                vae=args.vae,
                variant=variant,
                repo=spec["repo"],
                k=spec["k"],
                view=args.view,
                size=args.size,
                frames=args.frames,
                decode_latents=args.decode_latents,
                psnr_rgb_db=psnr,
                diffusers=diffusers.__version__,
                torch=torch.__version__,
            ),
            open(copy_dir / f"{c['clip']}.json", "w"),
        )
        print(
            f"  [{i + 1}/{len(plan)}] {c['clip']}: PSNR {psnr:.1f} dB  {time.time() - t0:.1f}s",
            flush=True,
        )

    print(f"{variant}: {done} clips on disk, {skipped} skipped", flush=True)
    if skipped > done:
        raise SystemExit(
            f"{variant}: more clips skipped than written -- treat this run as failed"
        )


# --------------------------------------------------------------------------- analyze
def analyze_clip(job):
    variant, clip, orig_path, copy_path, dst = job
    try:
        orig, copy = np.load(orig_path), np.load(copy_path)
        # Drop frame 0: the causal VAEs encode it alone, and without it the k-frame
        # groups start at index 0, so a k-frame repeat sits on exactly one FFT bin.
        m = rm.clip_metrics(rm.luma(orig[1:]), rm.luma(copy[1:]))
        tmp = _tmp_path(Path(dst))
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **m)
        os.replace(tmp, dst)
        return None
    except Exception as e:
        return f"{variant}/{clip}: {type(e).__name__}: {e}"


def _worker_init():
    cv2.setNumThreads(1)


def _variants(root: Path) -> list[str]:
    """Round-trip directories under `root`: a VAE name, optionally with a -suffix (ablations)."""
    found = (
        [d.name for d in root.iterdir() if d.is_dir() and d.name.split("-")[0] in VAES]
        if root.is_dir()
        else []
    )
    return sorted(found, key=lambda v: (list(VAES).index(v.split("-")[0]), v))


def analyze(
    frames_root,
    results_root,
    view: str = "resize",
    workers: int = 8,
    clip_timeout: int = 600,
) -> dict:
    """Measure every round-trip under <frames_root>/<view>, then summarise.

    Resumable, like lossless_test.analyze: each clip's measurements are written to
    clips/<vae>__<clip>.npz as it finishes, and a clip is skipped only if that file is
    newer than its frames. Returns the summary table, keyed by (vae, period in frames).
    """
    import multiprocessing as mp

    root, res = Path(frames_root) / view, Path(results_root) / view
    clip_dir = res / "clips"
    clip_dir.mkdir(parents=True, exist_ok=True)

    jobs, todo = [], []
    for variant in _variants(root):
        for f in sorted((root / variant).glob("*.npy")):
            orig = root / "real" / f.name
            if not orig.exists():
                continue
            dst = clip_dir / f"{variant}__{f.stem}.npz"
            jobs.append((variant, f.stem, str(orig), str(f), str(dst)))
            if not dst.exists() or dst.stat().st_mtime < max(
                f.stat().st_mtime, orig.stat().st_mtime
            ):
                todo.append(jobs[-1])
    print(
        f"view={view}: {len(jobs)} round-trips, {len(jobs) - len(todo)} already done, {len(todo)} to run",
        flush=True,
    )

    t0 = time.time()
    if workers <= 0:
        errors = [analyze_clip(j) for j in todo]
    else:
        errors = []
        with mp.get_context("spawn").Pool(workers, initializer=_worker_init) as pool:
            pending = [(j, pool.apply_async(analyze_clip, (j,))) for j in todo]
            for i, (j, r) in enumerate(pending, 1):
                try:
                    errors.append(r.get(timeout=clip_timeout))
                except mp.TimeoutError:
                    errors.append(f"{j[0]}/{j[1]}: timed out after {clip_timeout}s")
                if i % 50 == 0 or i == len(pending):
                    print(
                        f"  {i}/{len(pending)} clips  {time.time() - t0:.0f}s",
                        flush=True,
                    )
            pool.terminate()
    for e in filter(None, errors):
        print("  skip", e, flush=True)
    # only clips whose frames are still there: stale measurements of deleted runs are ignored
    return summarize(res, view, {(j[0], j[1]) for j in jobs})


def _load_clips(clip_dir: Path, variant: str, clips: list[str]):
    """Stack the given clips of one VAE: {measurement: (n_clips, ...)} and one video label per clip."""
    loaded = []
    for clip in clips:
        with np.load(clip_dir / f"{variant}__{clip}.npz") as z:
            loaded.append({key: z[key] for key in z.files})
    shapes = {c["frame_error"].shape for c in loaded}
    if len(shapes) > 1:
        raise ValueError(
            f"{variant}: clips of different lengths in {clip_dir} ({shapes}); analyse one --frames at a time"
        )
    data = {key: np.stack([c[key] for c in loaded]) for key in loaded[0]}
    # "<pexels id>_<file stem>_w<window>": windows of one video share everything before _w
    return data, [clip.rsplit("_w", 1)[0] for clip in clips]


def _finite(a: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(a), a, np.nan)


def summarize(res: Path, view: str, valid: set | None = None) -> dict:
    clip_dir = res / "clips"
    have = {}
    for f in sorted(clip_dir.glob("*__*.npz")):
        variant, clip = f.stem.split("__", 1)
        if variant.split("-")[0] in VAES and (
            valid is None or (variant, clip) in valid
        ):
            have.setdefault(variant, set()).add(clip)
    if not have:
        print("no results yet")
        return {}
    have = {
        v: have[v]
        for v in sorted(have, key=lambda v: (list(VAES).index(v.split("-")[0]), v))
    }
    common = sorted(set.intersection(*have.values()))
    notes = [
        f"{v}: {len(c)} clips measured, {len(c) - len(common)} left out (not yet done for every VAE)"
        for v, c in have.items()
        if len(c) != len(common)
    ]
    if not common:
        print(
            "no clip has been measured for every VAE yet:",
            {v: len(c) for v, c in have.items()},
        )
        return {}

    stats, table, curve_rows = {}, {}, []
    for variant in have:
        spec = VAES[variant.split("-")[0]]
        data, videos = _load_clips(clip_dir, variant, common)
        s = dict(k=spec["k"], n_clips=len(videos))
        for key in (
            "time_spike_db",
            "space_spike_db",
            "resid_time_spike_db",
            "resid_space_spike_db",
            "orig_time_spike_db",
            "orig_space_spike_db",
            "time_map_db",
            "space_map_db",
        ):
            s[key], s[key + "_sem"], s["n_videos"] = rm.clustered_mean_sem(
                _finite(data[key]), videos
            )
        psnr = data["psnr_db"]
        s["psnr_db"], s["psnr_db_sem"], _ = rm.clustered_mean_sem(
            _finite(psnr)[:, None], videos
        )
        s["n_lossless"] = int(np.sum(~np.isfinite(psnr)))
        # error over time, each clip scaled by its own mean so that busy clips do not dominate
        e = data["frame_error"]
        mean_e = e.mean(axis=1, keepdims=True)
        rel = np.divide(e, mean_e, out=np.full_like(e, np.nan), where=mean_e > 0)
        s["error_curve"], s["error_curve_sem"], _ = rm.clustered_mean_sem(rel, videos)
        # cycle strength: coherent (phase-preserving) average over videos, so content-driven
        # wobble cancels -- see roundtrip_metrics.cycle_coefficients. The result is a
        # magnitude: with no cycle it is not 0 but about 0.9 of its SEM.
        re, re_sem, _ = rm.clustered_mean_sem(data["error_coef"].real, videos)
        im, im_sem, _ = rm.clustered_mean_sem(data["error_coef"].imag, videos)
        s["error_cycle"], s["error_cycle_sem"] = np.hypot(re, im), np.hypot(
            re_sem, im_sem
        )
        stats[variant] = s

        T, N = e.shape[1], 2 * (len(s["space_spike_db"]) - 1)
        s["T"], s["N"] = T, N
        for metric, n, unit in (
            ("error_cycle", T, "frames"),
            ("time_spike_db", T, "frames"),
            ("resid_time_spike_db", T, "frames"),
            ("orig_time_spike_db", T, "frames"),
            ("space_spike_db", N, "pixels"),
            ("resid_space_spike_db", N, "pixels"),
            ("orig_space_spike_db", N, "pixels"),
        ):
            for b in range(1, len(s[metric])):
                curve_rows.append(
                    dict(
                        vae=variant,
                        metric=metric,
                        bin=b,
                        period=round(n / b, 4),
                        unit=unit,
                        mean=s[metric][b],
                        sem=s[metric + "_sem"][b],
                        n_videos=s["n_videos"],
                    )
                )
        for p in TIME_PERIODS:
            if T % p:
                continue
            b = T // p
            row = dict(
                k=spec["k"],
                n_videos=s["n_videos"],
                n_clips=s["n_clips"],
                psnr_db=float(s["psnr_db"][0]),
                psnr_db_sem=float(s["psnr_db_sem"][0]),
            )
            for metric in (
                "error_cycle",
                "time_spike_db",
                "resid_time_spike_db",
                "orig_time_spike_db",
            ):
                name = metric.replace("_db", "")
                row[name], row[name + "_sem"] = float(s[metric][b]), float(
                    s[metric + "_sem"][b]
                )
                row[name + "_nbr"] = rm.neighbour_median(s[metric], b)
            table[(variant, p)] = row

    with open(res / "roundtrip_curves.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(curve_rows[0].keys()))
        w.writeheader()
        w.writerows(curve_rows)
    np.savez_compressed(
        res / "roundtrip_maps.npz",
        **{
            f"{v}__{key}": s[key]
            for v, s in stats.items()
            for key in ("time_map_db", "space_map_db", "error_curve")
        },
    )
    text = _summary_text(stats, table, view, notes)
    print("\n" + text)
    (res / "roundtrip_summary.txt").write_text(text + "\n")
    _plots(stats, res)
    print(f"wrote {res}")
    return table


def _cell(mean, sem, nbr=None, fmt="+6.2f") -> str:
    if mean != mean:
        return "n/a"
    out = f"{mean:{fmt}} +-{sem:5.2f}"
    return (
        out if nbr is None else out + (f" [{nbr:{fmt}}]" if nbr == nbr else " [   n/a]")
    )


def _summary_text(stats: dict, table: dict, view: str, notes: list[str]) -> str:
    first = next(iter(stats.values()))
    w = 26
    lines = [
        f"VAE round-trip, view={view}: {first['T']} frames analysed (first frame dropped), "
        f"{first['N']}x{first['N']}, {first['n_clips']} clips common to every VAE.",
        "Cells are  mean +- SEM over source videos [nbrs = median of the neighbouring bins of the same curve].",
        "",
        "A DECODER MARK is a value that stands out from its own [nbrs], at a test VAE's own stride ('*'),",
        "and that sd (the frame-wise control) does not show. Clear of 0 is NOT enough: clip-edge effects lift",
        "every bin of the error cycle, and temporal smoothing shifts every spike value (see the script docstring).",
        "",
        "error cycle     swing of per-frame error at that spacing, % of mean error. A magnitude: ~0.9 SEM",
        "                is expected with no cycle. Needs the original, so it describes encoder + decoder.",
        "time spike      temporal-FFT prominence, copy minus original (the lossless test's measure).",
        "                The only temporal column a generated video could also carry.",
        "resid spike     the same prominence on the residual copy - original. Needs the original.",
        "source spike    the ORIGINAL's own prominence: what the source video and its codec already carry.",
        "",
        f"{'vae':<16}{'k':>3}{'videos':>8}{'spacing':>9}{'error cycle %':>{w}}{'time spike dB':>{w}}"
        f"{'resid spike dB':>{w}}{'source spike dB':>{w}}{'PSNR dB':>16}",
    ]
    for (variant, p), r in table.items():
        own = "*" if r["k"] == p else " "
        lines.append(
            f"{variant:<16}{str(r['k'] or '-'):>3}{r['n_videos']:>8}{f'{p} fr{own}':>9}"
            f"{_cell(100 * r['error_cycle'], 100 * r['error_cycle_sem'], 100 * r['error_cycle_nbr'], '6.2f'):>{w}}"
            f"{_cell(r['time_spike'], r['time_spike_sem'], r['time_spike_nbr']):>{w}}"
            f"{_cell(r['resid_time_spike'], r['resid_time_spike_sem'], r['resid_time_spike_nbr']):>{w}}"
            f"{_cell(r['orig_time_spike'], r['orig_time_spike_sem'], r['orig_time_spike_nbr']):>{w}}"
            f"{_cell(r['psnr_db'], r['psnr_db_sem'], fmt='6.2f'):>16}"
        )
    for key, title in (
        ("space_spike_db", "spatial grid spike dB, copy minus original"),
        ("resid_space_spike_db", "spatial grid spike dB on the residual"),
        (
            "orig_space_spike_db",
            "spatial grid spike dB of the ORIGINAL (source video and codec)",
        ),
    ):
        lines += [
            "",
            title
            + (
                ""
                if "orig" in key
                else "   -- no clean control: sd's VAE also works on 8x8 blocks"
            ),
            f"{'vae':<16}" + "".join(f"{f'{p} px':>{w}}" for p in SPACE_PERIODS),
        ]
        for variant, s in stats.items():
            cells = []
            for p in SPACE_PERIODS:
                b = s["N"] // p
                ok = s["N"] % p == 0 and b < len(s[key])
                cells.append(
                    _cell(s[key][b], s[key + "_sem"][b], rm.neighbour_median(s[key], b))
                    if ok
                    else "n/a"
                )
            lines.append(f"{variant:<16}" + "".join(f"{c:>{w}}" for c in cells))
    lossless = {v: s["n_lossless"] for v, s in stats.items() if s["n_lossless"]}
    if lossless:
        lines += ["", f"clips reproduced bit-exactly (excluded from PSNR): {lossless}"]
    if notes:
        lines += ["", "INCOMPLETE RUN -- " + "; ".join(notes)]
    return "\n".join(lines)


def _plots(stats: dict, res: Path):
    """Three figures, one panel per VAE on shared axes so panels compare by eye."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted = "#1f2933", "#7b8794"
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.edgecolor": muted,
            "axes.labelcolor": ink,
            "text.color": ink,
            "xtick.color": muted,
            "ytick.color": muted,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
        }
    )
    names = list(stats)
    n = len(names)

    def title(variant):
        k = stats[variant]["k"]
        return f"{variant}  ({'frame-wise control' if k is None else f'squeezes {k} frames'})"

    # 1. error over time
    fig, axes = plt.subplots(
        n, 1, figsize=(8, 1.9 * n + 0.6), sharex=True, sharey=True, squeeze=False
    )
    for ax, variant in zip(axes[:, 0], names):
        s = stats[variant]
        t = np.arange(1, s["T"] + 1)  # frame index in the clip; frame 0 was dropped
        ax.axhline(1.0, color=muted, lw=0.6)
        for edge in range(1, s["T"] + 1, s["k"] or s["T"] + 1):
            ax.axvline(
                edge - 0.5, color=muted, lw=0.5, alpha=0.35
            )  # start of each k-frame group
        ax.fill_between(
            t,
            s["error_curve"] - s["error_curve_sem"],
            s["error_curve"] + s["error_curve_sem"],
            color=ink,
            alpha=0.15,
            lw=0,
        )
        ax.plot(t, s["error_curve"], color=ink, lw=1.5, marker="o", ms=2.5)
        ax.set_title(title(variant), loc="left")
        ax.set_ylabel("error / clip mean")
    axes[-1, 0].set_xlabel(
        "frame (vertical lines: start of each group of k frames; band: ±1 SEM over videos)"
    )
    fig.suptitle(
        "Per-frame reconstruction error of the round-trip",
        x=0.01,
        ha="left",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(res / "error_over_time.png", dpi=150)
    plt.close(fig)

    # 2 and 3. difference maps: diverging colours around a neutral 0, one scale for all panels
    for key, fname, sup in (
        (
            "time_map_db",
            "time_difference_map.png",
            "Round-trip minus original: time x space",
        ),
        (
            "space_map_db",
            "space_difference_map.png",
            "Round-trip minus original: within-frame",
        ),
    ):
        maps = [stats[v][key] for v in names]
        finite = np.concatenate([np.abs(m[np.isfinite(m)]) for m in maps])
        lim = max(1.0, float(np.percentile(finite, 98))) if finite.size else 1.0
        fig, axes = plt.subplots(
            1, n, figsize=(3.4 * n + 0.8, 3.9), squeeze=False, constrained_layout=True
        )
        for ax, variant, m in zip(axes[0], names, maps):
            s = stats[variant]
            if key == "time_map_db":
                im = ax.imshow(
                    m,
                    origin="lower",
                    aspect="auto",
                    cmap="coolwarm",
                    vmin=-lim,
                    vmax=lim,
                    extent=(
                        -0.5 / s["N"],
                        (m.shape[1] - 0.5) / s["N"],
                        -0.5 / s["T"],
                        (m.shape[0] - 0.5) / s["T"],
                    ),
                )
                ax.set_xlabel("spatial frequency (cycles / pixel)")
                ax.set_ylabel("temporal frequency (cycles / frame)")
                if s["k"]:
                    ax.axhline(1 / s["k"], color=ink, lw=0.6, ls=":")
                    ax.annotate(
                        f"every {s['k']} frames",
                        (ax.get_xlim()[1], 1 / s["k"]),
                        ha="right",
                        va="bottom",
                        fontsize=7,
                        xytext=(-2, 1),
                        textcoords="offset points",
                    )
            else:
                # fftshifted bins run from -N/2 to N/2 - 1, each half a bin wide either side
                im = ax.imshow(
                    m,
                    origin="lower",
                    cmap="coolwarm",
                    vmin=-lim,
                    vmax=lim,
                    extent=(
                        -0.5 - 0.5 / s["N"],
                        0.5 - 0.5 / s["N"],
                        -0.5 - 0.5 / s["N"],
                        0.5 - 0.5 / s["N"],
                    ),
                )
                ax.set_xlabel("horizontal frequency (cycles / pixel)")
                ax.set_ylabel("vertical frequency (cycles / pixel)")
            ax.set_title(title(variant), loc="left", fontsize=9)
        fig.colorbar(
            im,
            ax=axes[0].tolist(),
            shrink=0.85,
            label="power change (dB): blue = lost, red = added",
        )
        fig.suptitle(sup, x=0.01, ha="left", fontweight="bold")
        fig.savefig(res / fname, dpi=150)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("stage", choices=["roundtrip", "analyze"])
    ap.add_argument("--vae", choices=list(VAES), help="roundtrip stage: which VAE")
    ap.add_argument("--da_root", default="src/face_fft/data/deepaction_dataset")
    ap.add_argument(
        "--out",
        default="roundtrip_frames",
        help="raw frames: <out>/<view>/{real,<vae>}/<clip>.npy",
    )
    ap.add_argument(
        "--results", default="/standard/uva-mira-drive/3d-fft_results/vae_roundtrip"
    )
    ap.add_argument("--view", choices=["resize", "crop"], default="resize")
    ap.add_argument(
        "--n",
        type=int,
        default=200,
        help="clips; later windows of the same videos once each has one",
    )
    ap.add_argument(
        "--frames",
        type=int,
        default=49,
        help="1 + a multiple of every stride under test",
    )
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument(
        "--decode_latents",
        type=int,
        default=None,
        help="cogvideox only: latent frames decoded per chunk (pipeline default 2); "
        "the chunking ablation, written to cogvideox-dec<N>/",
    )
    ap.add_argument(
        "--workers", type=int, default=8, help="analyze stage; 0 = run in this process"
    )
    ap.add_argument(
        "--clip_timeout", type=int, default=600, help="analyze stage: seconds per clip"
    )
    args = ap.parse_args()
    if args.stage == "roundtrip":
        if not args.vae:
            ap.error("roundtrip needs --vae")
        roundtrip(args)
    else:
        analyze(args.out, args.results, args.view, args.workers, args.clip_timeout)


if __name__ == "__main__":
    main()
