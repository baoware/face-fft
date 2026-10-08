"""Our detectors under each benchmark paper's own protocol, next to the published
numbers (as extracted from the papers on 2026-09-30; verify by hand before citing).

  1. GenVidBench official   train Pair1 -> test Pair2         (top-1 accuracy; MViTv2-B 83.27 on 143k)
  2. VidAudit-style LOGO     hold out one GenVidBench generator (AUC; WaveRep 0.996, ReStraV 0.931, CLIP 0.852)
  3. DVF (MM-Det)            7 test generators, per-gen AUC     (MM-Det 92.0, CNNDet 78.2)
  4. GenVideo (DeMamba)      GenVideo -> GenVideo-Val, per-gen AP (DeMamba-XCLIP-FT 0.971, VideoMAE-B 0.945)

Differences that remain, by design: our clips are format-normalised (consecutive
native-rate frames, square crop, 256x256, one fixed H.264 encode) while the papers read
raw files; and our training sets are capped at 2,000 clips per source.
"""

import glob
import json
from collections import defaultdict

import numpy as np

C = "/standard/uva-mira-drive/3d-fft_checkpoints/cached/"
fmt = lambda a: f"{np.mean(a):.3f}±{np.std(a, ddof=1):.3f}" if len(a) > 1 else (f"{a[0]:.3f}" if a else "–")


def runs(pattern, eval_file="eval.json"):
    out = defaultdict(list)
    for d in sorted(glob.glob(C + pattern)):
        try:
            cfg = json.load(open(d + "/config.json")); ev = json.load(open(d + "/" + eval_file))
        except FileNotFoundError:
            continue
        out[cfg["arch"] + ("+K400" if cfg.get("pretrained") else "")].append(ev["tests"])
    return out


print("1. GenVidBench official protocol: train Pair1 only, test Pair2 (unseen generators + unseen real source)")
print("   published (143k, raw files): MViTv2-B top-1 83.27 | per class MuseV 76.34, SVD 98.29, CogV 47.50, Mora 96.62, HD(real) 97.58")
print(f"   {'model':<22}{'AUC':>14}{'acc @0.5':>14}{'bal.acc @0.5':>14}{'F1 @0.5':>14}   per-class @val-thr: MuseV SVD CogV Mora HD")
for arch, evs in runs("off_*").items():
    t = [e["Pair2"] for e in evs if "Pair2" in e]
    pc = lambda g: np.mean([x["per_fake_source"][g]["detection_rate"] for x in t]) * 100
    hd = np.mean([1 - x["per_real_source"]["hd_vg_130m"]["false_positive_rate"] for x in t]) * 100
    print(f"   {arch:<22}{fmt([x['auc'] for x in t]):>14}{fmt([x['acc_05'] for x in t]):>14}"
          f"{fmt([x['balanced_acc_05'] for x in t]):>14}{fmt([x['f1_05'] for x in t]):>14}   "
          f"{pc('musev'):.1f} {pc('svd'):.1f} {pc('cogvideo'):.1f} {pc('mora'):.1f} {hd:.1f}")

print("\n2. Leave-one-generator-out on GenVidBench (held-out generator vs its benchmark's reals), pretrained R3D")
print("   published (VidAudit, audited): WaveRep 0.996, ReStraV 0.931, CLIP-ViT-B/32 0.852, D3 0.557")
per_gen = defaultdict(list)
for d in sorted(glob.glob(C + "logo_r3d_*")):
    try:
        cfg = json.load(open(d + "/config.json")); ev = json.load(open(d + "/eval.json"))["tests"]
    except FileNotFoundError:
        continue
    g = cfg["exclude_sources"]
    pair = "Pair1" if g in ("ms", "pika", "t2vz", "vc2") else "Pair2"
    per_gen[g].append(ev[pair]["per_fake_source"][g]["auc_vs_reals"])
for g, a in per_gen.items():
    print(f"   held out {g:<10} AUC {fmt(a)}")
if per_gen:
    seeds = min(len(a) for a in per_gen.values())
    means = [np.mean([per_gen[g][s] for g in per_gen]) for s in range(seeds)]
    print(f"   MEAN over {len(per_gen)} generators: {fmt(means)}  (spread over seeds)")

print("\n3. DVF, MM-Det protocol (7 test generators, no SVD fakes / YouTube reals), mean per-generator AUC")
print("   published: MM-Det 92.0±2.6 (trained on DVF's SVD + YouTube), CNNDet 78.2")
for label, pattern, f in (("trained on DVF split", "dvftrain_*", "eval.json"),
                          ("zero-shot, trained on all 8 GenVidBench gens", "pre_lr1e-4_*", "eval_dvfp.json"),
                          ("zero-shot, COMPACT 3D FFT (GenVidBench)", "gvb_fft_no_mask_*", "eval_dvfp.json")):
    for arch, evs in runs(pattern, f).items():
        t = [e["DVFp"] for e in evs if "DVFp" in e]
        print(f"   {label:<46}{arch:<20}{fmt([x['mean_per_generator_auc'] * 100 for x in t])}")

print("\n4. GenVideo protocol: train GenVideo-100K (capped 2,000/source), test GenVideo-Val, mean per-generator AP")
print("   published (full 2.26M training set, raw files): DeMamba-XCLIP-FT 0.971, VideoMAE-B 0.945, XCLIP-B-FT 0.868")
for arch, evs in runs("gvp_*").items():
    t = [e["GenVideo"] for e in evs if "GenVideo" in e]
    print(f"   {arch:<22} AP {fmt([x['mean_per_generator_ap'] for x in t])}   AUC {fmt([x['mean_per_generator_auc'] for x in t])}")
