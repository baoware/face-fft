# VAE round-trip results, explained from zero

Results of `bash scripts/vae_roundtrip.sh` (view = resize, 256x256, 200 clips from 100
real videos). Files are in `output/resize/`. This page goes through them one at a time:
what the file is, how to read it, and what it shows.

## Conclusion

The question: does a video generator's VAE decoder, by itself, leave a mark in the video?

> On real video passed through three video VAEs at 256x256, we found **no periodic
> fingerprint common to all of them**. The expected spike at the temporal stride is
> absent for CogVideoX and LTX and small for Wan. The marks that do exist are specific to
> one or two models and sit in fine detail, where video compression is likely to remove
> them. The VAE decoder alone does not look like a sufficient basis for a detector that
> generalises across generators.

**What each VAE leaves in its output** (measurable without the original, so usable by a
detector):

| VAE | Mark at its stride (4 or 8 frames) | 2-frame flicker | Grid within each frame |
| --- | --- | --- | --- |
| CogVideoX (4) | No: −0.09 ± 0.01 dB, neighbours −0.03 | No: −0.03 dB | No: −0.00 dB at 8 px |
| Wan (4) | Small: +0.30 ± 0.08 dB, neighbours −0.11 | Faint: +0.64 ± 0.09 dB | No: −0.06 dB at 8 px |
| LTX (8) | No: −0.13 ± 0.02 dB, neighbours −0.07 | Faint: +1.22 ± 0.11 dB | **Strong: +1.77 dB at 4 px, single frequencies up to +25 dB** |
| SD (control) | No: +0.04 / +0.02 dB | No: +0.06 dB | No: +0.01 dB at 8 px |

**Supporting facts**

- **The run is sound.** 200 clips from 100 videos, PSNR 30 to 37 dB for all four VAEs.
- **The VAEs do work in groups of frames.** Per-frame error cycles by 11.3% (CogVideoX)
  and 8.4% (Wan) every 4 frames and 5.5% (LTX) every 8, against 0.1% for the control.
  This needs the original to measure, so it is a check that the experiment can see a
  stride rhythm, not a usable fingerprint. It is why the "No" entries above can be
  trusted.
- **All three video VAEs blur fine detail.** This is shared, but ordinary compression
  and downscaling do the same, so it does not separate fake from real.

**Not tested here**

- **Compression.** Frames were uncompressed. "Likely removed by compression" is an
  expectation, not a result.
- **Generated video.** These are real clips through the VAE only. The generation step
  could leave marks this experiment cannot see.
- **The 2-frame flicker** sits at the edge of the measurement and needs confirming.
- **CogVideoX chunking** (see the end of this page). It affects only the explanation of
  its error cycle, not the conclusion.

**Suggested next steps**

- **Generated pairs.** Train and test a detector on fakes generated from a real clip's
  first frame, paired with that clip. This tests whether the generation step leaves a
  mark, which this experiment cannot see.
- **Small detector on real vs round-trip.** A cheap double-check for patterns the
  measurements did not look for. A detector can succeed here on blur alone, so only its
  score after re-compression, or on real fakes, says anything about a fingerprint.
- **Compression check.** Re-measure the round-trips after shrinking and H.264
  re-compression, to see whether the LTX grid and the flicker survive.
- **Existing fakes.** Run the same measures on the lossless-test CogVideoX and Wan clips,
  to see whether Wan's small stride mark and the flicker appear in generated video.

The rest of this page explains each result file for a reader with no background.

## Background you need (2 minutes)

**What a VAE is.** A video generator does not draw pixels directly. It works on a small
compressed version of the video, and a component called the VAE turns that back into
pixels. The VAE squeezes several frames into one: 4 frames for CogVideoX and Wan, 8 for
LTX. That number is called its *stride*.

**What we did.** We took real videos, squeezed them with each VAE and immediately
un-squeezed them. No AI generation happened. We call the result the *copy*. Because the
copy and the original show the same scene, anything different between them was put there
by the VAE.

**The four VAEs.**

| Name | Squeezes | Role |
| --- | --- | --- |
| `cogvideox` | 4 frames into 1 | being tested |
| `wan` | 4 frames into 1 | being tested |
| `ltx` | 8 frames into 1 | being tested |
| `sd` | nothing (one frame at a time) | **control** |

`sd` is an image VAE. It handles each frame separately, so it cannot create a pattern
across frames. It is the "nothing happening" baseline: a result only counts if a test
VAE shows it and `sd` does not.

**Three ideas used everywhere below.**

- **"Every N frames".** A pattern that repeats every 4 frames looks like
  good-good-bad-good, good-good-bad-good. We look for repeats at 4 and 8 frames because
  those are the strides.
- **dB (decibels).** A way of writing "how many times bigger". 0 dB is no change.
  +3 dB is about 2 times, +10 dB is 10 times, +20 dB is 100 times. Negative means
  smaller. +0.3 dB is only about 7% more.
- **`±` and "neighbours".** `±` is the uncertainty of a number. "Neighbours" is the value
  at nearby spacings (for 4 frames, that is spacings like 3.4 or 4.8 frames). A real
  4-frame mark makes the 4-frame value stick out from its neighbours. If the neighbours
  are just as high, it is not a mark, it is everything going up together.

---

## File 1: `error_over_time.png`

![error over time](../output/resize/error_over_time.png)

### What it is

One panel per VAE. It shows how badly each frame was reconstructed, frame by frame.

- **Horizontal axis ("frame"):** position in the clip, frame 1 to frame 48. (The clip
  has 49 frames; the very first is left out because VAEs treat it specially.)
- **Vertical axis ("error / clip mean"):** how wrong that frame's copy is, compared with
  the average frame in the same clip. **1.0 means average.** 1.1 means 10% worse than
  average, 0.85 means 15% better. Dividing by the clip's average lets us compare VAEs
  that differ in overall quality.
- **Faint vertical lines:** where each group of frames starts (every 4 frames for
  CogVideoX and Wan, every 8 for LTX).
- Each dot is the average over all 100 videos.

### Why it matters and how to read it

If a VAE handled every frame the same way, the line would be flat at 1.0. If it works in
groups, some positions inside each group will always come out better or worse, and the
line will show **the same shape repeating between every pair of vertical lines**. That
repeating shape is direct proof that the VAE has a rhythm.

- Repeating shape = the VAE works in groups.
- Flat line = no rhythm.
- A jump only at the very end = an edge effect, not a rhythm.

### What we found

| VAE | What the line does | Meaning |
| --- | --- | --- |
| `cogvideox` | Clean zigzag repeating every 4 frames. The 3rd frame of each group is about 14% better than average (0.86); the other three are 1.01 to 1.08. | Clear 4-frame rhythm |
| `wan` | Same zigzag, slightly smaller. The 3rd frame of each group is about 11% better (0.89). | Clear 4-frame rhythm |
| `ltx` | A step every 8 frames: error jumps up by about 10% at frames 8, 16, 24, 32, 40 and drifts down in between. Also a slow rise through the clip, and the last frame is 57% worse than average (1.57). | 8-frame rhythm, but mixed with an end-of-clip effect |
| `sd` | Flat. Every frame is between 0.99 and 1.02. | No rhythm, as a control should be |

**So:** the video VAEs really do work in groups of 4 or 8 frames, and the control does
not. This confirms the mechanism behind our hypothesis.

**The catch:** this plot needs the original video to compute the error. A detector never
has the original of a fake. So this shows the rhythm exists, not that it can be detected.
The next file answers that.

---

## File 2: `roundtrip_summary.txt`

### What it is

The results as numbers. It has one main table about time and three small tables about
space. This is the file to quote from.

### How to read one row

Take this row:

```
vae   k  videos  spacing        error cycle %          time spike dB         resid spike dB        source spike dB      PSNR dB
wan   4     100    4 fr*    8.42 +- 0.73 [ 0.18]   +0.30 +- 0.08 [-0.11]   +0.69 +- 0.13 [-0.47]   -0.38 +- 0.02 [-0.43]   35.42 +- 0.43
```

- **`vae`, `k`:** which VAE and its stride (Wan squeezes 4 frames).
- **`videos`:** 100 source videos. This is the sample size.
- **`spacing`:** which repeat this row is about. `4 fr` = a repeat every 4 frames. The
  `*` means "this is the VAE's own stride, so this is where its mark should be".
- **Every cell is `value +- uncertainty [neighbours]`.** For `8.42 +- 0.73 [0.18]`: the
  value is 8.42, give or take 0.73, and nearby spacings read 0.18. 8.42 is far above
  0.18, so this one sticks out.

The columns, in plain words:

| Column | Plain meaning | Needs the original? |
| --- | --- | --- |
| `error cycle %` | The zigzag from File 1 as one number: how much the frame error swings at that spacing. | Yes |
| `time spike dB` | **The important one.** Does the copy, looked at by itself, have more of a repeat at this spacing than the original had? | No |
| `resid spike dB` | Same idea, measured on (copy minus original). More sensitive. | Yes |
| `source spike dB` | The repeat the original already had before any VAE touched it. A warning light. | (original only) |
| `PSNR dB` | Overall closeness of copy to original. About 30 to 40 is healthy. | Yes |

### The rule for calling something a mark

All of these must be true:

1. It is in a row with a `*`.
2. The value is clearly above the number in `[brackets]`.
3. The gap is several times the `+-` number.
4. `sd` does not show the same thing.

### What we found

**Sanity check: pass.** 200 clips, 100 videos, no "INCOMPLETE RUN" warning. PSNR is 36.9
(CogVideoX), 35.4 (Wan), 33.1 (LTX), 30.4 (SD). All four VAEs worked properly.

**`error cycle %` (rhythm exists, needs the original):**

| VAE at its own stride | Value | Neighbours | Verdict |
| --- | --- | --- | --- |
| `cogvideox`, 4 fr | 11.26 ± 0.90 | 0.26 | Huge. Clear rhythm. |
| `wan`, 4 fr | 8.42 ± 0.73 | 0.18 | Huge. Clear rhythm. |
| `ltx`, 8 fr | 5.48 ± 0.48 | 2.88 | Present, less clean (neighbours are raised by the end-of-clip effect). |
| `sd`, 4 fr / 8 fr | 0.10 / 0.27 | 0.09 / 0.11 | Nothing. |

This is File 1 in numbers, and it agrees.

**`time spike dB` (can you see it in the copy alone?):**

| VAE at its own stride | Value | Neighbours | Verdict |
| --- | --- | --- | --- |
| `cogvideox`, 4 fr | −0.09 ± 0.01 | −0.03 | **No mark.** |
| `wan`, 4 fr | +0.30 ± 0.08 | −0.11 | **Small mark.** About 0.4 dB above neighbours, roughly 10% more energy. |
| `ltx`, 8 fr | −0.13 ± 0.02 | −0.07 | **No mark.** |
| `sd`, 4 fr / 8 fr | +0.04 / +0.02 | +0.04 / +0.03 | Nothing, as expected. |

**So:** CogVideoX has the strongest rhythm in its *error* and yet leaves no visible
4-frame mark in the *video*. Only Wan leaves one, and it is small. This is the main
disappointment for the original hypothesis.

**The spatial tables (patterns within a frame, at 8, 16 and 32 pixels):**

| VAE | 8 px, copy alone | 8 px, on the residual |
| --- | --- | --- |
| `cogvideox` | −0.00 ± 0.01 | +0.26 ± 0.02 |
| `wan` | −0.06 ± 0.01 | +0.14 ± 0.02 |
| `ltx` | **+0.29 ± 0.04** (neighbours −0.04) | **+1.41 ± 0.08** (neighbours −0.06) |
| `sd` | +0.01 ± 0.01 | +0.07 ± 0.02 |

**So:** LTX leaves a grid pattern that repeats every 8 pixels and is visible in the copy
alone. The others do not. Nothing shows at 16 or 32 pixels for any VAE. (File 4 shows
this grid is much stronger at finer spacings than the table's 8 pixels.)

---

## File 3: `time_difference_map.png`

![time difference map](../output/resize/time_difference_map.png)

### What it is

One panel per VAE. A heat map of what the VAE changed, sorted by two things at once.

- **Horizontal axis ("spatial frequency"):** how fine the detail is. Left = large smooth
  shapes. Right = the finest pixel-level texture.
- **Vertical axis ("temporal frequency"):** how fast things change over time. Bottom =
  not changing. Top (0.5) = flipping every 2 frames, the fastest possible.
- **Colour:** red = the copy has **more** of this than the original (something was
  added). Blue = the copy has **less** (something was lost). Grey = unchanged.
- **Dotted line:** the height that corresponds to the VAE's stride ("every 4 frames" is
  at 0.25, "every 8 frames" is at 0.125).

### Why it matters and how to read it

This is a picture of the `time spike` column. A stride mark would be a **thin red
horizontal streak lying on the dotted line**. Large smooth blue areas are just blur
(every compressor loses fine detail) and are not a fingerprint.

### What we found

| VAE | What the panel shows | Meaning |
| --- | --- | --- |
| `cogvideox` | Blue on the right side, at every height (about −2.5 dB on average). Nothing on the dotted line. | Plain blur of fine detail. No time mark. |
| `wan` | Mostly grey. Red band along the **top edge** on the right (about +3.8 dB). Only a faint trace at the dotted line. | Adds fast flicker in fine detail. |
| `ltx` | Strong blue on the right (about −5 dB), then **deep red along the top edge** (about +6 dB). Nothing on the dotted line. | Heavy blur, plus fast flicker in fine detail. |
| `sd` | Faint, even pink in the upper part (about +1.4 dB). No streaks. | A little random noise per frame. No structure. |

**So:** there is no red streak on any dotted line, which matches the table. The
unexpected finding is the red band at the top of the Wan and LTX panels: both make fine
detail flicker from one frame to the next. It is not at the stride, so it is not what we
predicted, but it is in the copy alone.

---

## File 4: `space_difference_map.png`

![space difference map](../output/resize/space_difference_map.png)

### What it is

The same kind of heat map, but for patterns *inside a single frame*.

- **Centre of each panel:** large smooth shapes.
- **Towards the edges and corners:** finer and finer detail.
- **Horizontal axis:** patterns that vary left to right. **Vertical axis:** patterns
  that vary top to bottom.
- **Colour:** same as before. Red = added, blue = lost.
- A position of 0.25 means "repeats every 4 pixels"; 0.125 means every 8 pixels.

### Why it matters and how to read it

A VAE builds each frame out of small blocks. If the blocks do not join perfectly, the
frame carries a faint regular grid, like the weave of a fabric. In this plot a grid shows
up as **isolated bright dots at regular positions**. A smooth blue fade towards the
corners is just blur.

### What we found

| VAE | What the panel shows | Meaning |
| --- | --- | --- |
| `cogvideox` | Smooth blue fade to the corners (about −11 dB there). No dots. | Blur only. |
| `wan` | Light blue at the edges (about −6 dB), two faint pink blobs. No sharp dots. | Mild blur. No grid. |
| `ltx` | **Sharp red dots in a regular lattice**, at multiples of 0.125. The brightest reach +15 to +25 dB, which is 30 to 300 times more energy than the original had there. Deep blue corners. | **A strong regular grid.** |
| `sd` | Flat grey. | Almost nothing changed. |

**So:** LTX stamps a regular grid onto every frame. This is the strongest and cleanest
fingerprint in the whole experiment, it is visible without the original, and the other
three VAEs do not have it.

---

## File 5: `roundtrip_curves.csv`

### What it is

The full data behind the summary table. The summary only reports 4 and 8 frames and 8,
16 and 32 pixels. This file has **every** spacing.

One row per VAE, measurement and spacing. Columns:

| Column | Meaning |
| --- | --- |
| `vae` | Which VAE |
| `metric` | Which measurement (`error_cycle`, `time_spike_db`, `space_spike_db`, and the `resid_` and `orig_` versions) |
| `period`, `unit` | The spacing, in frames or pixels |
| `mean`, `sem` | The value and its uncertainty |
| `n_videos` | Sample size (100) |
| `bin` | Internal index of the spacing; ignore it |

Note `error_cycle` is stored as a fraction here (0.1126 means 11.26%).

### Why it matters and how to read it

Use it to check for marks at spacings we did not think to look at. Open it in Excel,
filter to one `vae` and one `metric`, and sort by `mean`. A real mark is one spacing far
above the rest.

### What we found

- **CogVideoX's rhythm is exactly at 4 frames.** `error_cycle` is 11.3% at 4 frames and
  under 2% at every other spacing except 2 frames (2.7%, an echo of the 4-frame zigzag).
- **The fast flicker is confirmed.** `time_spike_db` at a spacing of about 2 frames:
  LTX **+1.22 ± 0.11 dB**, Wan **+0.64 ± 0.09 dB**, CogVideoX −0.03, SD +0.06. This is
  larger than anything at the strides.
- **LTX's grid is strongest at fine spacings.** `space_spike_db` for LTX: +0.29 dB at
  8 px, **+1.77 dB at 4 px**, +1.85 dB at 2.67 px, +2.29 dB at about 2 px.

## File 6: `roundtrip_maps.npz`

The raw numbers behind Files 1, 3 and 4, for anyone who wants to redraw the plots in
Python (`numpy.load`). Nothing new to read here.

---

## What this means for the project

1. **The mechanism is real.** Video VAEs do work in groups of frames (File 1).
2. **The predicted time fingerprint is weak.** A repeat at the stride is absent in the
   output of CogVideoX and LTX and small in Wan (File 2). A detector that relies only on
   "a spike every 4 or 8 frames" has little to work with, at least at this resolution.
3. **There are fingerprints, in other places, but none is shared.** LTX's spatial grid
   (File 4) and the 2-frame flicker of Wan and LTX (Files 3 and 5) are visible in the
   output alone. CogVideoX has neither, so they do not give a detector something that
   generalises across generators.

## What this does not show yet

- **CogVideoX needs one more run.** Its decoder also cuts the clip into chunks, and the
  joins fall on the same 4/8-frame spacing as its stride. Its zigzag could come from
  either. To settle it:

  ```bash
  sbatch --array=1 --export=ALL,DECODE_LATENTS=4  scripts/vae_roundtrip.slurm
  sbatch --array=1 --export=ALL,DECODE_LATENTS=13 scripts/vae_roundtrip.slurm
  sbatch scripts/vae_roundtrip_analyze.slurm      # after both finish
  ```

- **These are not generated videos.** Real fakes go through a generation step and are
  then compressed as video files. Either could erase the grid or the flicker. Checking
  real LTX and Wan outputs for these two marks is the natural next test.
- **The 2-frame flicker is a lead, not a result.** It sits at the very edge of what the
  measurement can see, where the "neighbours" comparison is one-sided.
- **One setting only.** 256x256 resized, 48 frames, the source videos' own frame rate.
