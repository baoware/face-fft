"""Measure the period-N temporal peak on the DeepAction generators as they ship.

This is the replication check for the pilot: RunwayML and VideoPoet were reported at
+6.3 dB and +4.6 dB. If this instrument does not reproduce roughly those numbers on the
same data, then any null it reports elsewhere is uninterpretable -- the measurement, not
the hypothesis, would be what differs.

Everything here is H.264 as distributed; no generation, no GPU.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent))
from codec_control import read_frames_bounded, temporal_power_spectrum, peak_prominence_db

from face_fft.data.generate import preprocess_video_tensor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--per_gen", type=int, default=25)
    ap.add_argument("--num_frames", type=int, default=48)
    ap.add_argument("--period", type=int, default=4)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.root)
    gens = sorted(d.name for d in root.iterdir() if d.is_dir())
    print(f"Generators: {gens}\n", flush=True)

    results = {}
    for g in gens:
        vids = sorted((root / g).rglob("*.mp4"))[: args.per_gen]
        vals = []
        for v in tqdm(vids, desc=g, leave=False):
            try:
                u8 = read_frames_bounded(str(v), args.num_frames, args.size)
                t = preprocess_video_tensor(u8, target_size=(args.size, args.size),
                                            num_frames=args.num_frames)
                P = temporal_power_spectrum(t)
                d = peak_prominence_db(P, args.period)
                if not np.isnan(d):
                    vals.append(float(d))
            except Exception as e:
                print(f"  skip {v.name}: {type(e).__name__}: {e}", flush=True)
        results[g] = vals
        if vals:
            a = np.array(vals)
            print(f"{g:28s} n={len(a):3d}  mean={a.mean():+6.2f} dB  "
                  f"median={np.median(a):+6.2f}  std={a.std():5.2f}  "
                  f">1dB={100*np.mean(a>1.0):3.0f}%", flush=True)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
