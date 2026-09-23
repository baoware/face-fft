"""Format-normalise clips from the manifest and cache them as uint8 arrays.

Every clip goes through the same chain, so no file-format property survives
into the model input:
  1. read the first frames at the video's OWN frame rate (consecutive, never
     spread across the clip, never interpolated)
  2. real videos only: also keep every 2nd/3rd/4th frame as extra rows, so reals
     span the fakes' 4-24 fps range. Fakes are never strided -- only strides
     congruent to +-1 mod the latent period keep the artifact in place, and that
     period differs by generator.
  3. centre-crop to square, resize to SIZE (INTER_AREA)
  4. re-encode with ONE fixed H.264 setting and decode back. B-frames are OFF:
     x264's default B-frame pyramid imposes a ~4-frame quality cadence, which would
     inject a period-4 pattern into every clip and mask the artifact we look for.
  5. store (T_MAX, SIZE, SIZE, 3) uint8; clips shorter than T_MAX are zero-padded
     and n_valid records the real length. Slice [:8] for T=8 experiments.

One shard per (pair, source, split): <out>/<pair>__<source>__<split>.npy plus a
matching .csv index. Also records cheap low-level stats per clip for the
shortcut baseline (brightness, contrast, sharpness, motion, encoded size).
"""

import argparse
import csv
import io
import random
import time
from collections import defaultdict
from pathlib import Path

import av
import cv2
import numpy as np

T_MAX = 16
REAL_STRIDES = (1, 2, 3, 4)


def read_frames(path: str, n: int, size: int) -> np.ndarray:
    """First n frames, centre-cropped square and resized. (t, size, size, 3) RGB uint8."""
    cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise ValueError("cannot open")
    out = []
    while len(out) < n:
        ok, f = cap.read()
        if not ok:
            break
        h, w = f.shape[:2]
        s = min(h, w)
        f = f[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
        f = cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA)
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    if not out:
        raise ValueError("no frames decoded")
    return np.stack(out)


def reencode(frames: np.ndarray, crf: int) -> tuple[np.ndarray, int]:
    """Fixed H.264 round trip. Returns decoded frames and encoded byte size."""
    buf = io.BytesIO()
    with av.open(buf, "w", format="mp4") as c:
        st = c.add_stream("libx264", rate=8)
        st.width, st.height = frames.shape[2], frames.shape[1]
        st.pix_fmt = "yuv420p"
        # one keyframe per clip, no B-frames, no scene-cut keyframes: identical GOP
        # structure for every clip regardless of its source encoder
        st.options = {"crf": str(crf), "bf": "0", "g": str(len(frames)),
                      "keyint_min": str(len(frames)), "sc_threshold": "0", "preset": "medium"}
        for f in frames:
            for pkt in st.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
                c.mux(pkt)
        for pkt in st.encode():
            c.mux(pkt)
    nbytes = buf.tell()
    buf.seek(0)
    with av.open(buf, "r") as c:
        dec = np.stack([fr.to_ndarray(format="rgb24") for fr in c.decode(video=0)])
    return dec, nbytes


def lowlevel_stats(x: np.ndarray, nbytes: int) -> dict:
    g = x.mean(axis=-1).astype(np.float32)                    # (t, H, W) luma proxy
    lap = np.mean([cv2.Laplacian(fr, cv2.CV_32F).var() for fr in g])
    motion = float(np.abs(np.diff(g, axis=0)).mean()) if len(g) > 1 else 0.0
    return dict(mean=float(g.mean()), contrast=float(g.std()), sharpness=float(lap),
                motion=motion, bytes_per_frame=nbytes / len(g))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pair", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--caps", default="train=2000,val=300,test=1000",
                    help="max content items per split; 'all' for no cap")
    ap.add_argument("--max_per_group", type=int, default=2, help="clips per content group (vript scenes)")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--crf", type=int, default=23)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    caps = {} if args.caps == "all" else {k: int(v) for k, v in (kv.split("=") for kv in args.caps.split(","))}
    rows = [r for r in csv.DictReader(open(args.manifest))
            if r["pair"] == args.pair and r["source"] == args.source]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    by_split = defaultdict(list)
    for r in rows:
        by_split[r["split"]].append(r)

    for split, items in sorted(by_split.items()):
        rng = random.Random(f"{args.seed}:{args.pair}:{args.source}:{split}")
        rng.shuffle(items)
        per_group, chosen = defaultdict(int), []
        for r in items:
            if per_group[r["group"]] >= args.max_per_group:
                continue
            per_group[r["group"]] += 1
            chosen.append(r)
            if split in caps and len(chosen) >= caps[split]:
                break

        is_real = chosen and chosen[0]["label"] == "0"
        strides = REAL_STRIDES if is_real else (1,)
        stem = f"{args.pair}__{args.source}__{split}"
        arr = np.lib.format.open_memmap(out / f"{stem}.npy", mode="w+", dtype=np.uint8,
                                        shape=(len(chosen) * len(strides), T_MAX, args.size, args.size, 3))
        index, row, t0, failed = [], 0, time.time(), 0
        for i, r in enumerate(chosen):
            try:
                native = read_frames(r["path"], T_MAX * max(strides), args.size)
            except Exception as e:
                failed += 1
                print(f"  FAIL {r['path']}: {e}", flush=True)
                continue
            for s in strides:
                clip = native[::s][:T_MAX]
                if s > 1 and len(clip) < T_MAX:
                    continue                                   # too short for this stride
                dec, nbytes = reencode(clip, args.crf)
                n = len(dec)
                arr[row, :n] = dec
                index.append(dict(row=row, path=r["path"], group=r["group"], label=r["label"],
                                  pair=r["pair"], source=r["source"], split=split, stride=s,
                                  n_valid=n, **lowlevel_stats(dec, nbytes)))
                row += 1
            if (i + 1) % 250 == 0:
                print(f"  {stem}: {i + 1}/{len(chosen)}  {time.time() - t0:.0f}s", flush=True)
        arr.flush(); del arr
        if row < len(chosen) * len(strides):                 # trim unused preallocated rows
            full = np.load(out / f"{stem}.npy", mmap_mode="r")[:row].copy()
            np.save(out / f"{stem}.npy", full)
        with open(out / f"{stem}.csv", "w", newline="") as f:
            if index:
                w = csv.DictWriter(f, fieldnames=list(index[0].keys())); w.writeheader(); w.writerows(index)
        print(f"{stem}: {len(chosen)} clips -> {row} rows, {failed} failed, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
