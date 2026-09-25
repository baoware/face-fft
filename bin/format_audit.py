"""Per-source container/codec audit for GenVidBench, and how much each format
field alone separates real from fake.

A spectral detector is maximally exposed to format shortcuts: frame rate, clip
length, resolution and encoder settings all land directly in the 3D FFT. The
write-up already notes clip length alone reaches 0.998 AUC on GenVidBench. This
quantifies every such field per split, so we know what the preprocessing has to
equalise before any result can be attributed to generator artifacts.

Reads container metadata only (PyAV); no frames are decoded except to measure the
keyframe interval, which scans packet flags.
"""

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

import av
import numpy as np
from sklearn.metrics import roc_auc_score

NUMERIC = ["width", "height", "fps", "n_frames", "duration_s", "bit_rate_kbps",
           "bits_per_pixel", "gop"]


def probe(path: Path) -> dict:
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        cc = s.codec_context
        fps = float(s.average_rate) if s.average_rate else float("nan")
        dur = float(c.duration / av.time_base) if c.duration else float("nan")
        n = s.frames or (int(round(fps * dur)) if fps == fps and dur == dur else 0)
        br = (c.bit_rate or 0) / 1000.0
        # keyframe interval from packet flags (container-level, no decode)
        kf, i = [], 0
        for pkt in c.demux(s):
            if pkt.size == 0:
                continue
            if pkt.is_keyframe:
                kf.append(i)
            i += 1
            if i >= 600:
                break
        gop = float(np.median(np.diff(kf))) if len(kf) > 1 else float(i)
        w, h = cc.width, cc.height
        bpp = (br * 1000.0) / (w * h * fps) if w and h and fps == fps and fps > 0 else float("nan")
        return {
            "codec": cc.name, "profile": cc.profile or "", "pix_fmt": cc.format.name if cc.format else "",
            "width": w, "height": h, "fps": round(fps, 3), "n_frames": n,
            "duration_s": round(dur, 3), "bit_rate_kbps": round(br, 1),
            "bits_per_pixel": bpp, "gop": gop,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", help="extracted/ directory (with --labels)")
    ap.add_argument("--labels", nargs="+", help="Pair{1,2}_labels.txt")
    ap.add_argument("--manifest", help="alternative input: a build_manifest.py CSV")
    ap.add_argument("--pairs", default=None, help="with --manifest: comma list of pairs to audit")
    ap.add_argument("--per_source", type=int, default=400)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    by_source = defaultdict(list)  # (pair, source) -> [(path, label)]
    if args.manifest:
        want = set(args.pairs.split(",")) if args.pairs else None
        for r in csv.DictReader(open(args.manifest)):
            if want is None or r["pair"] in want:
                # key by split as well, so train-side and test-side sources stay separate
                by_source[(f"{r['pair']}:{r['split']}", r["source"])].append((Path(r["path"]), int(r["label"])))
    root = Path(args.root) if args.root else None
    for lf in (args.labels or []):
        for line in open(lf):
            line = line.rstrip("\n")
            if not line:
                continue
            rel, lab = line.rsplit(" ", 1)         # filenames contain spaces
            pair, source = rel.split("/")[:2]
            by_source[(pair, source)].append((root / rel, int(lab)))

    rows = []
    for (pair, source), items in sorted(by_source.items()):
        present = [(p, l) for p, l in items if p.exists()]
        random.shuffle(present)
        sample = present[: args.per_source]
        ok = 0
        for p, lab in sample:
            try:
                r = probe(p)
            except Exception as e:
                continue
            r.update(pair=pair, source=source, label=lab, path=str(p))
            rows.append(r); ok += 1
        print(f"{pair}/{source:12s} listed={len(items):6d} on_disk={len(present):6d} probed={ok}", flush=True)

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {len(rows)} rows -> {args.out_csv}\n")

    # per-source summary
    print(f"{'source':<20}{'lbl':>4}{'codec':>7}{'pix_fmt':>10}{'res (mode)':>13}"
          f"{'fps':>8}{'frames':>8}{'dur s':>7}{'kbps':>8}{'bpp':>7}{'gop':>6}")
    for (pair, source) in sorted({(r["pair"], r["source"]) for r in rows}):
        rs = [r for r in rows if r["pair"] == pair and r["source"] == source]
        med = lambda k: np.nanmedian([r[k] for r in rs])
        mode = lambda k: max(set(r[k] for r in rs), key=[r[k] for r in rs].count)
        res = max(set((r["width"], r["height"]) for r in rs), key=[(r["width"], r["height"]) for r in rs].count)
        print(f"{pair+'/'+source:<20}{rs[0]['label']:>4}{mode('codec'):>7}{mode('pix_fmt'):>10}"
              f"{f'{res[0]}x{res[1]}':>13}{med('fps'):>8.2f}{med('n_frames'):>8.0f}"
              f"{med('duration_s'):>7.1f}{med('bit_rate_kbps'):>8.0f}{med('bits_per_pixel'):>7.3f}{med('gop'):>6.0f}")

    # shortcut check: AUC of each single field for real-vs-fake, within each split
    print("\nSingle-field shortcut AUC (real vs fake). 0.5 = uninformative; |AUC-0.5| near 0.5 = the field alone separates the classes.")
    print(f"{'field':<16}" + "".join(f"{p:>10}" for p in sorted({r['pair'] for r in rows})))
    for k in NUMERIC:
        line = f"{k:<16}"
        for pair in sorted({r["pair"] for r in rows}):
            rs = [r for r in rows if r["pair"] == pair and r[k] == r[k]]
            y = [r["label"] for r in rs]; x = [r[k] for r in rs]
            if len(set(y)) < 2:
                line += f"{'n/a':>10}"; continue
            auc = roc_auc_score(y, x)
            line += f"{max(auc, 1 - auc):>10.3f}"
        print(line)


if __name__ == "__main__":
    main()
