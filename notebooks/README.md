# Face-FFT Interactive Notebooks

Two Jupyter notebooks for interactive exploration of the face-fft detection pipeline outside the full HPC pipeline.

## Notebook Overview

### 1. `01_generation_assessment.ipynb`

**Purpose:** Verify the CogVideoX generation pipeline works end-to-end.

**Key steps:**
1. Extract first frames from a few real videos
2. Run CogVideoX to generate synthetic video sequences
3. Display input vs. output side-by-side for visual inspection
4. Optionally save preprocessed tensors for Notebook 2

**Use this when:**
- Validating that CogVideoX is installed and working
- Checking temporal coherence and face identity in generated videos
- Building a paired dataset of real/synthetic video tensors

**Outputs:** `.pt` tensor files (if `OUTPUT_DIR` is set)

---

### 2. `02_train_and_evaluate.ipynb`

**Purpose:** Train the spectral CNN classifier and evaluate on test data.

**Key steps:**
1. Load paired real/synthetic `.pt` tensors
2. Visualise frequency-domain differences (FFT magnitude)
3. Train `FaceFFTPipeline` (SpatiotemporalFFT + CompactSpectralCNN) for N epochs
4. Plot training curves
5. Evaluate F1 score and confusion matrix on held-out test set

**Use this when:**
- Prototyping detection with small-scale data
- Understanding the spectral hypothesis visually
- Benchmarking lightweight models before scaling to HPC

**Inputs:** Directories of `.pt` tensor files (real and synthetic, matched by filename)

**Fallback:** If data directories don't exist, random tensors are generated automatically.

---

## Quick Start

### Scenario A: Full Pipeline (Generate → Train)

1. **Notebook 1:**
   - Set `RAW_VIDEO_DIR` to a folder containing `.mp4` / `.avi` / `.mov` files
   - Set `OUTPUT_DIR` to save tensors (e.g., `Path("outputs")`)
   - Run all cells

2. **Notebook 2:**
   - Set `REAL_DIR = Path("../outputs/real")`
   - Set `SYNTH_DIR = Path("../outputs/synth")`
   - Run all cells

### Scenario B: Training Only (Existing Tensors)

- If you already have paired `.pt` tensor files:
  1. Organize them under two directories: `real/` and `synth/`
  2. Open Notebook 2
  3. Update `REAL_DIR` and `SYNTH_DIR`
  4. Run all cells

### Scenario C: Inspection Only (No Data)

- Both notebooks auto-generate dummy random tensors if data directories are missing
- Useful for checking model architecture, loss curves, and evaluation code without real data

---

## Data Format

Both notebooks expect **preprocessed tensors** in PyTorch format (`.pt` files).

**Expected shape:** `(C, T, H, W)` where:
- `C = 3` (RGB)
- `T = 16` (temporal dimension, configurable)
- `H, W = 256, 256` (spatial resolution)
- **Range:** `[0, 1]` (float)

**Pairing:** Files in `real/` and `synth/` directories must have **identical filenames**:
```
real/video_001.pt  ←→  synth/video_001.pt
real/video_002.pt  ←→  synth/video_002.pt
```

Notebook 1's `preprocess_video_tensor` function outputs this format automatically.

---

## Configuration Reference

### Notebook 1: Generation Assessment

| Variable | Default | Notes |
|---|---|---|
| `RAW_VIDEO_DIR` | `../src/face_fft/data/raw_videos` | Source video directory |
| `NUM_VIDEOS` | `3` | How many videos to process |
| `GENERATOR_PROMPT` | face description | Text conditioning for CogVideoX |
| `NUM_INFERENCE_STEPS` | `20` | Denoising iterations (higher = better quality, slower) |
| `CACHE_DIR` | `None` | HuggingFace cache path (set on HPC to avoid re-downloads) |
| `OUTPUT_DIR` | `None` | Where to save `.pt` tensors; if `None`, tensors are kept in-memory |

### Notebook 2: Training & Evaluation

| Variable | Default | Notes |
|---|---|---|
| `REAL_DIR` | `../data/real` | Real tensor directory |
| `SYNTH_DIR` | `../data/synth` | Synthetic tensor directory |
| `NUM_EPOCHS` | `5` | Demo default; increase for real runs (20+) |
| `BATCH_SIZE` | `4` | Samples per gradient step |
| `LR` | `1e-3` | AdamW learning rate |
| `SAVE_PATH` | `checkpoints/demo_model.pt` | Best checkpoint save location |
| `DEVICE` | auto-detect | `"cuda"` or `"cpu"` |

---

## Understanding the Results

### F1 Score Interpretation

| F1 | Interpretation |
|---|---|
| 0.90–1.00 | Strong spectral separation |
| 0.75–0.90 | Reasonable generalization signal |
| 0.60–0.75 | Weak but detectable |
| < 0.60 | Near-random (check data/preprocessing) |

### Spectral Visualization (Notebook 2, Cell 4)

The FFT magnitude plot shows:
- **Left (Real):** Smooth, isotropic spectrum
- **Right (Synthetic):** Periodic grid spikes or harmonic peaks from compression

If both look identical, the spectral hypothesis may not hold for your video type or resolution.

### Loss Curves (Notebook 2, Cell 8)

- **Both decreasing** → Model is learning; train longer
- **Train ↓, Val ↑** → Overfitting; reduce capacity or add regularization
- **Both flat** → Possible learning rate issue or bad initialization

---

## HPC Deployment

These notebooks are meant for **interactive local prototyping**. For production HPC runs:

1. Export the final configuration to the Slurm scripts in `scripts/`
2. Use `bin/generate_dataset.py`, `bin/train.py`, `bin/evaluate.py`
3. Refer to `scripts/README.md` for cluster submission

---

## Troubleshooting

**CogVideoX model fails to load:**
- Increase `NUM_INFERENCE_STEPS` (model loading is slow on first run)
- Set `CACHE_DIR` to a local path to cache model weights

**No video files found:**
- Check `RAW_VIDEO_DIR` points to the correct folder
- Verify files have extensions `.mp4`, `.avi`, or `.mov`

**Out of memory (OOM):**
- Reduce `BATCH_SIZE` in Notebook 2
- Reduce `NUM_VIDEOS` or `NUM_INFERENCE_STEPS` in Notebook 1
- On GPU: reduce `NUM_EPOCHS` or increase inference step intervals

**F1 score near 0.5 (random):**
- Check the spectral visualization (Cell 4, Notebook 2) — is there a visible signal?
- Verify tensor shapes are correct: `(C=3, T, H=256, W=256)`
- Ensure real and synthetic are actually paired (not shuffled differently)

---

## References

- **Main pipeline:** `src/face_fft/`
- **Full docs:** See `CLAUDE.md` for project philosophy and guidelines
- **HPC scripts:** `scripts/` folder
