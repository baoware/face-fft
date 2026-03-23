# Face-FFT: Spatiotemporal Spectral Detection of AI-Generated Face Videos

## Project Goal

Detect AI-generated (synthetic) face videos using 3D Fast Fourier Transform analysis. The core hypothesis is that efficient video generation models introduce structured spectral artifacts in the frequency domain due to latent compression, patch tokenization, and temporal downsampling — and that these artifacts remain detectable even at low spatial resolution (256×256).

---

## Pipeline Overview

The full pipeline consists of five modular stages:

1. **Video Ingestion** — Load real face videos; extract first frame as conditioning input
2. **Synthetic Generation** — Generate paired synthetic videos using image-to-video diffusion models
3. **Preprocessing** — Resize/crop to 256×256, standardize to N temporal frames, export as `.pt` tensors
4. **Spectral Feature Extraction** — Apply 3D FFT (T, H, W dimensions) with log-magnitude scaling and DC centering
5. **Classification** — Lightweight 3D CNN trained with binary cross-entropy (real=0, synthetic=1)

---

## Implementation Status

### Completed

| Component | Location | Description |
|---|---|---|
| Spectral FFT module | `src/face_fft/features/spectral.py` | `SpatiotemporalFFT`: 3D FFT + fftshift + log-magnitude |
| CNN classifier | `src/face_fft/models/classifier.py` | `CompactSpectralCNN`: 4-block 3D CNN, ~300K parameters |
| End-to-end pipeline | `src/face_fft/models/pipeline.py` | `FaceFFTPipeline`: composes FFT + classifier as single `nn.Module` |
| Paired dataset loader | `src/face_fft/data/dataset.py` | `PairedVideoDataset` for `.pt` tensor pairs; leakage-safe splitting |
| Video generation | `src/face_fft/data/generate.py` | CogVideoX-5b-I2V and Wan2.2-I2V-A14B wrappers |
| Trainer | `src/face_fft/training/trainer.py` | AdamW + BCEWithLogitsLoss; best-checkpoint saving |
| Evaluator | `src/face_fft/eval/evaluator.py` | F1 score + confusion matrix reporting |
| DeepAction integration | `src/face_fft/data/deepaction.py` | Discovery + stratified splitting for external dataset |
| Generation notebook | `notebooks/01_generation_assessment.ipynb` | CogVideoX generation and visual QA |
| Train/eval notebook | `notebooks/02_train_and_evaluate.ipynb` | Full training loop + ablation study |

---

## Dataset

### Self-Collected Dataset (current experiments)

- **Real videos:** Selfie-style face recordings sourced locally (`src/face_fft/data/raw_videos/`)
- **Synthetic videos:** Generated with **CogVideoX-5b-I2V** conditioned on the first frame of each real video
  - Prompt: *"A highly realistic person's face, talking slightly, photorealistic video."*
  - 20 diffusion inference steps, guidance scale 6.0, dynamic CFG enabled
- **Total pairs:** 44 real–synthetic pairs (88 videos total)
- **Video format:** `.pt` tensors, shape `(C=3, T=8, H=256, W=256)`
- **Split (80/10/10 on pairs):**
  - Train: 35 pairs → 70 samples
  - Val: 4 pairs → 8 samples
  - Test: 5 pairs → 10 samples

### DeepAction Dataset (integration ready, not yet trained)

- Code in `src/face_fft/data/deepaction.py` supports the DeepAction v1 dataset structure:
  - Real videos under `<root>/Pexels/<class>/`
  - Synthetic videos under `<root>/<ModelName>/<class>/`
- `DeepActionDataset` performs on-the-fly decoding, resizing, and frame standardization
- Stratified 80/10/10 split preserves class balance across real and synthetic

---

## Model Architecture

**FaceFFTPipeline** (`~300,401 parameters`)

```
Input: (B, 3, 8, 256, 256)
  └─ SpatiotemporalFFT (no learnable params)
       - 3D FFT over (T, H, W)
       - fftshift to center DC component
       - log(|magnitude| + ε)
  └─ CompactSpectralCNN
       - Conv3d(3→16) + BN + ReLU → MaxPool3d(2×2×2)
       - Conv3d(16→32) + BN + ReLU → MaxPool3d(2×2×2)
       - Conv3d(32→64) + BN + ReLU → MaxPool3d(2×2×2)
       - Conv3d(64→128) + BN + ReLU → AdaptiveAvgPool3d(1×1×1)
       - Flatten → Dropout(0.3) → Linear(128→64) → ReLU → Dropout(0.3) → Linear(64→1)
Output: (B, 1) logit
```

**Training config:** AdamW, lr=1e-3, weight_decay=1e-4, 5 epochs, batch size 4, seed 42

---

## Results

### Spectral Model (3D-FFT → CNN)

**Training loss curve:**

| Epoch | Train Loss | Val Loss |
|-------|-----------|----------|
| 1 | 0.6974 | 0.6759 |
| 2 | 0.6124 | 0.4839 |
| 3 | 0.4751 | **0.2730** ← best checkpoint |
| 4 | 0.3997 | 0.4627 |
| 5 | 0.3128 | 0.9120 |

**Test set performance (10 samples, 5 pairs):**

| Metric | Value |
|--------|-------|
| F1 Score | **0.8889** |
| True Negatives (Real→Real) | 5 |
| False Positives (Real→Synth) | 0 |
| False Negatives (Synth→Real) | 1 |
| True Positives (Synth→Synth) | 4 |

### Ablation Study: 3D-FFT vs. Pixel Domain Baseline

The pixel-domain baseline uses the identical CNN architecture but receives raw video tensors instead of FFT spectra.

**Pixel baseline training loss curve:**

| Epoch | Train Loss | Val Loss |
|-------|-----------|----------|
| 1 | 0.7092 | 0.6930 |
| 2 | 0.6923 | 0.6873 |
| 3 | 0.7046 | 0.7001 |
| 4 | 0.7139 | 0.6954 |
| 5 | 0.7076 | 0.6913 |

**Comparison:**

| Model | F1 Score |
|-------|----------|
| Pixel domain (no FFT) | 0.3333 |
| **Spectral (3D-FFT)** | **0.8889** |
| Improvement | **+0.5556** |

The pixel baseline fails to learn (loss remains near 0.69 ≈ log(2), consistent with random guessing), while the spectral model converges, supporting the hypothesis that the detection signal is genuinely present in the frequency domain.

---

## Limitations

1. **Extremely small dataset.** 44 pairs is insufficient for statistically robust conclusions. Test set is only 10 samples, so reported F1 has very high variance.
2. **Single generator.** All synthetic videos were generated with CogVideoX-5b-I2V. No cross-generator generalization has been tested.
3. **Overfitting evidence.** Val loss rises sharply after epoch 3 (0.27 → 0.91) despite continued train loss reduction. The model overfits given the small dataset size.
4. **Short training.** Only 5 epochs; performance with longer training + regularization on a larger dataset is unknown.
5. **No cross-model evaluation.** The evaluation philosophy in CLAUDE.md emphasizes cross-model generalization, but this has not yet been performed.
6. **Temporal resolution.** Videos are standardized to only 8 frames, which may limit temporal artifact capture.
7. **Wan2.2 not yet tested.** The Wan2.2-I2V-A14B generation wrapper is implemented but no dataset has been generated with it.
8. **DeepAction not yet trained.** Integration code is complete but no training or evaluation results exist on this dataset.

---

## Next Steps

### High Priority
1. **Scale up dataset** — Use the DeepAction dataset (already integrated) or generate more self-collected pairs. Target 500+ pairs minimum for meaningful evaluation.
2. **Cross-generator experiments** — Train on CogVideoX-generated data, evaluate on Wan2.2-generated data (and vice versa) to assess generalization.
3. **Longer training with proper regularization** — Run 20–50 epochs with early stopping; consider data augmentation to reduce overfitting.

### Medium Priority
4. **Generate Wan2.2 dataset** — Use `generate_synthetic_video_wan()` to create a second synthetic dataset for cross-model ablation.
5. **Frequency visualization** — Plot averaged spectral magnitudes per class to qualitatively confirm the artifact hypothesis and provide paper figures.
6. **HPC deployment** — Package training as a Slurm batch job for Rivanna; document CUDA/environment setup.

### Lower Priority
7. **Increase temporal resolution** — Experiment with T=16 or T=32 frames to assess whether temporal artifacts become more or less detectable.
8. **Quantitative frequency analysis** — Identify and localize specific frequency bands that differ most between real and synthetic spectra (supports the paper's interpretability claims).
