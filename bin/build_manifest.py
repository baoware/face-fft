"""Build one manifest of every video we train or evaluate on, with leakage-safe splits.

Splits are assigned per CONTENT GROUP, never per file, because the datasets share
content across files:
  * GenVidBench Pair1: ms/pika/t2vz/vc2 all render the same 13,501 VidProM prompts
    (file stem = prompt uuid). vript's 20,131 real clips are scenes cut from 1,746
    YouTube videos.
  * GenVidBench Pair2: the real HD-VG clip and its 4 fakes share an index (00001___...).
  * DeepAction: Pexels/<id>/ and <generator>/<id>/ belong together.
  * GenVidBench 6.7m (Sora, Kling, OpenSora): fakes only, no reals -> test only.

The split is a hash of (group, seed), so it is deterministic, independent of file
order, and a group lands in the same split in every run and every experiment.
"""

import argparse
import csv
import hashlib
import re
from collections import Counter
from pathlib import Path

GVB_P1_FAKE_PREFIX = {"ms": "ms-", "pika": "pika-", "t2vz": "t2vz-", "vc2": "vc-"}


def split_of(group: str, seed: int, train: int, val: int) -> str:
    h = int(hashlib.sha1(f"{seed}:{group}".encode()).hexdigest(), 16) % 100
    return "train" if h < train else ("val" if h < train + val else "test")


def gvb_rows(gvb_root: Path, labels_dir: Path):
    for pair in ("Pair1", "Pair2"):
        for line in open(labels_dir / f"{pair}_labels.txt"):
            line = line.rstrip("\n")
            if not line:
                continue
            rel, lab = line.rsplit(" ", 1)              # Pair2 filenames contain spaces
            _, source, fname = rel.split("/", 2)
            stem = fname[:-4]
            if pair == "Pair1" and source == "vript":
                group = "vript:" + re.sub(r"-Scene-\d+$", "", stem)
            elif pair == "Pair1":
                pre = GVB_P1_FAKE_PREFIX[source]
                assert stem.startswith(pre), (source, stem)
                group = "vidprom:" + stem[len(pre):]
            else:
                group = "p2:" + stem.split("___", 1)[0]
            yield dict(path=str(gvb_root / "extracted" / rel), dataset="genvidbench",
                       pair=pair, source=source, label=int(lab), group=group, fixed_split="")

    # 6.7m subset: fakes only; keep them out of training entirely
    for source, sub in (("sora", "Sora"), ("kling", "Kling"), ("opensora", "OpenSora")):
        for p in sorted((gvb_root / sub).rglob("*.mp4")):
            if "__MACOSX" in p.parts or p.name.startswith("._"):
                continue                                  # macOS resource-fork junk
            yield dict(path=str(p), dataset="genvidbench_6.7m", pair="6.7m", source=source,
                       label=1, group=f"67m:{source}:{p.stem}", fixed_split="test")


def deepaction_rows(da_root: Path):
    for gen_dir in sorted(d for d in da_root.iterdir() if d.is_dir()):
        if gen_dir.name.startswith(".") or gen_dir.name in ("hub", "xet"):
            continue
        label = 0 if gen_dir.name == "Pexels" else 1
        for p in sorted(gen_dir.rglob("*.mp4")):
            pexels_id = p.parent.name
            yield dict(path=str(p), dataset="deepaction", pair="DeepAction", source=gen_dir.name,
                       label=label, group=f"da:{pexels_id}", fixed_split="")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gvb_root", required=True)
    ap.add_argument("--da_root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_pct", type=int, default=70)
    ap.add_argument("--val_pct", type=int, default=10)
    args = ap.parse_args()

    gvb = Path(args.gvb_root)
    rows = list(gvb_rows(gvb, gvb / "raw" / "GenVidBench")) + list(deepaction_rows(Path(args.da_root)))

    missing = [r for r in rows if not Path(r["path"]).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} manifest paths do not exist, e.g. {missing[0]['path']}")

    for r in rows:
        r["split"] = r.pop("fixed_split") or split_of(r["group"], args.seed, args.train_pct, args.val_pct)

    # leakage check: every group must sit in exactly one split
    g2s = {}
    for r in rows:
        assert g2s.setdefault(r["group"], r["split"]) == r["split"], r["group"]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "dataset", "pair", "source", "label", "group", "split"])
        w.writeheader(); w.writerows(rows)

    print(f"wrote {len(rows)} rows, {len(g2s)} content groups -> {args.out}\n")
    c = Counter((r["pair"], r["source"], r["label"], r["split"]) for r in rows)
    print(f"{'pair/source':<28}{'lbl':>4}{'train':>8}{'val':>7}{'test':>8}{'groups':>8}")
    for key in sorted({(p, s, l) for p, s, l, _ in c}):
        p, s, l = key
        ng = len({r['group'] for r in rows if (r['pair'], r['source']) == (p, s)})
        print(f"{p + '/' + s:<28}{l:>4}{c[key + ('train',)]:>8}{c[key + ('val',)]:>7}{c[key + ('test',)]:>8}{ng:>8}")


if __name__ == "__main__":
    main()
