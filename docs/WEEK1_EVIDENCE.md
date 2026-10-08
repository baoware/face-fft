# Week 1 evidence: does the paper's main claim hold?

As of 2026-09-28. This answers the week-1 decision in `SUBMISSION_PLAN.md`. The code is on
branch `feat/reproducible-seeding-codec-control`. Results are under
`/standard/uva-mira-drive/3d-fft_results/` and checkpoints under
`/standard/uva-mira-drive/3d-fft_checkpoints/cached/`.

## Bottom line

Measured against the plan's own week-1 rule, **the main claim is not supported**:

- **(a) A generator trace that doesn't depend on the codec: not found.** Lossless CogVideoX and Wan output shows
  no period-4 peak, and codec frame patterns don't create one either.
- **(b) The spectrum beats CLIP under degradation: not tested.** It became moot after (a).
- **(c) 1D vs 3D: 3D isn't supported.** No spectrum wins consistently, and 3D never wins on a dataset it
  wasn't trained on.

The plan's fallback paper ("which spectral traces come from generation, and which from format and codec")
is already mostly measured; see the last section. **One thing could still change this:** the pilot script
(`scripts/pilot/spectral_peaks.py`), which isn't in the repo. None of my statistics reproduce its numbers,
so it should be run on the 200 lossless clips before the team decides.

## 1. Lossless test: no trace, and no codec confound

For each model, 50 clips were stored as raw frames with no codec. The real clips are the Pexels videos the
generated clips started from. I re-encoded every clip with fixed frame patterns: all-keyframe, IPPP, IBBP
(period 3) and IBBBP (period 4). Each encode was checked to confirm it produced the requested pattern.
Table values are period-4 peak strength in dB (mean ± SEM, 24 frames, native-resolution crop). A real peak
would be clearly above 0; the pilot reported +4 to +6 dB.

| Source | Raw | IBBBP, CRF 35 | IBBP, CRF 35 |
|---|---|---|---|
| Real (Pexels) | −1.48 ± 0.12 | −1.42 | −1.61 |
| SVD (frame-by-frame) | −1.07 ± 0.10 | −1.09 | −1.11 |
| CogVideoX-5B (4× in time) | −0.85 ± 0.12 | −0.83 | −0.90 |
| Wan 2.1-1.3B (4× in time) | −0.77 ± 0.20 | −0.81 | −0.91 |

- **No model produces a period-4 peak.** The 4×-in-time models sit only 0.2–0.3 dB above SVD, and the
  ordering follows frame rate (CogVideoX 8, Wan 16, real 25 fps), so it is more likely spectral slope than
  a trace.
- **Codec frame patterns change nothing,** by at most 0.1 dB even at CRF 35. So the codec confound the plan
  worried about doesn't show up here either.
- The same holds with the frame-difference statistic, and on the resized view (`lossless_test/`).
- The one small positive: in an earlier, content-matched CogVideoX run, a period-4-specific excess of
  +0.49 dB over real video (SEM 0.17, n = 50). It is real but tiny, far from the pilot's +4 to +6 dB.

**The pilot doesn't reproduce.** On the pilot's own DeepAction files and native crop, my reconstruction of
its frame-difference statistic gives:

| Source | This reconstruction | Pilot |
|---|---|---|
| RunwayML | −0.37 ± 1.18 | +6.3 |
| VideoPoet | −0.01 ± 1.00 | +4.6 |
| CogVideoX | −2.12 ± 1.13 | +0.9 |

Over 96 frames, with the slow motion trend removed, Runway's frame changes have the same autocorrelation
pattern as real video (lag 4: −0.24 vs −0.31), so frame interpolation doesn't explain it either.

## 2. The benchmarks can be solved from file format alone

**Single file fields** (as distributed, 400 files per source). Each value is the AUC for real vs fake from that
one number:

| Benchmark | Frame count | Frame rate | Height |
|---|---|---|---|
| GenVidBench Pair1 / Pair2 | 0.997 / 0.998 | 0.976 / 1.000 | 0.983 / 1.000 |
| GenVideo-Val (test) | 0.668 | **1.000** | **1.000** |
| DVF | 0.933 | 0.927 | 0.882 |

GenVideo-Val's real videos are MSR-VTT stored at **3 fps**, 298×224. Every fake is 8 fps or more.

**Metadata-only detector.** Container fields plus codec, no pixels, trained on one benchmark and tested on
another (`bin/metadata_classifier.py`):

| Protocol | Metadata only | Best spectral detector |
|---|---|---|
| Pair1 → Pair2 (unseen generators) | **0.994–1.000** | 0.814 |
| GenVideo → Pair2 | **1.000** | 0.845 |
| Pair1 → DeepAction | **0.958–0.988** | about 0.55 |
| Pair1 → DVF | **0.924–0.973** | 0.858 |
| GenVideo train → GenVideo-Val | 0.012–0.380 (inverted) | 0.975 |

Caveat: the metadata model reads the files as distributed, while the spectral detectors see
format-normalized clips. So this shows the benchmarks can be solved from format. It does not show that
our detectors used format.

## 3. Detectors: no consistent winner, and dependence on the real-video source

Setup: COMPACT CNN, 16 frames, 3 seeds, AUC. Clips are normalized: consecutive frames, square crop resized
to 256, one fixed H.264 encode.

| Trained on → tested on | 3D FFT | 2D per frame | 1D in time | Pixels |
|---|---|---|---|---|
| Pair1 → Pair2 | 0.743 | **0.814** | 0.722 | 0.743 |
| Pair1 → GenVideo-Val | 0.837 | 0.728 | **0.904** | 0.849 |
| Pair1 → DVF | 0.852 | **0.858** | 0.690 | 0.800 |
| GenVideo → Pair2 | 0.707 | 0.764 | **0.845** | 0.765 |
| GenVideo → DeepAction | 0.319 | 0.362 | 0.453 | **0.588** |

- **The best representation changes with every test set, and 3D is never the best.**
- **Five simple statistics come close.** Using brightness, contrast, sharpness, motion and encoded size, they
  reach 0.663 on Pair1 → Pair2 and **0.870 on Pair1 → GenVideo-Val, which beats the 3D FFT's 0.837**.
- **Detectors learn the real-video source.** Detectors trained on GenVideo mark **85–99% of clean real
  video** (4K Pexels, DVF YouTube) as fake, but only **5–13%** of MSR-VTT. The 3D and 2D models are
  inverted on DeepAction (0.32–0.36).
- **Temporal models inverted on DeepAction.** A 1D temporal model trained on frame-by-frame generators
  scores DeepAction's CogVideoX and VideoPoet at 0.27–0.30, i.e. confidently real. It appears to have
  learned "flicker means fake", and generators that compress in time don't flicker.

## 4. The fallback paper, from what we already have

1. **Do temporal-compression traces exist in pixels?** No: lossless test plus codec sweep (§1).
2. **How much of benchmark performance is format?** Nearly all of it: format audit, metadata-only
   detector, simple statistics (§2, §3).
3. **What do detectors learn instead?** The real-video source and its quality: per-source false-positive
   rates and cross-dataset inversions (§3).
4. **Contributions:** a format-normalized protocol, leakage-safe splits (e.g. GenVidBench's Pair1 shares
   13,501 prompts across its 4 generators), and the codec-sweep test.

**Still needed for that paper:**
- a format-swap test: re-encode reals in the fakes' format and see whether detectors flip
- CLIP-probe and WaveRep baselines, to show the problem isn't specific to our model
- running the pilot script on the lossless clips

## Known limitations

- The pilot's exact statistic is unknown.
- Clips are resized rather than cropped at native resolution; that choice is still open.
- The ablation uses one architecture (COMPACT).
- DeepAction has only 24 real test videos.
- GIFs are excluded from GenVideo.
