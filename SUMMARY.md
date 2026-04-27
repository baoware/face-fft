# Implementation Summary

## GenVidBench + DeepAction Combined Training

### What was added

| File | Purpose |
|------|---------|
| `src/face_fft/data/genvidbench.py` | `GenVidBenchDataset` class + `get_genvidbench_splits()` |
| `src/face_fft/training/train_combined.py` | Training script for the combined data mixture |
| `src/face_fft/training/run_sweep.py` | Full sweep: trains + evaluates all (mode × arch) combos → CSV |
| `scripts/run_sweep.slurm` | Slurm job for the full sweep |
| `src/face_fft/data/download_genvidbench.py` | Updated to support `--include-real` and `--list-files` |
| `scripts/download_genvidbench_full.slurm` | Slurm job: download full GenVidBench (synth + real) |
| `scripts/train_combined.slurm` | Slurm job: combined training on A100 (or H100) |

---

### Dataset integration design

**GenVidBench real videos**

The GenVidBench HuggingFace repo (`jian-0/GenVidBench`) may or may not include a
real-video archive.  The updated `download_genvidbench.py` attempts to locate it
automatically by scanning the repo file listing; if found it is extracted to
`genvidbench_dataset/Real/`.

If no real-video archive exists in the repo, `train_combined.py` falls back to
using the DeepAction `Pexels/` videos as the real-class source for GenVidBench
fakes.  This is valid for binary real/fake detection — the task does not require
that real and fake videos come from the same source, only that the model learns to
separate the two classes.

**Class balance**

Both `get_deepaction_splits()` and `get_genvidbench_splits()` produce balanced
samples (equal real/fake counts per split).  `ConcatDataset` merges the two
without re-balancing so the mixture is proportional to dataset size.

**Train / val / test split**

Splits are performed at the real–fake pair level with a fixed `seed=42` to prevent
data leakage.  The test split from each dataset is deliberately unused in
`train_combined.py`; use the existing `eval_genvidbench.py` and
`deepaction_evaluator.py` scripts to evaluate on held-out data after training.

---

### How to run

**Step 1 — download GenVidBench (if not already done)**

```bash
sbatch scripts/download_genvidbench_full.slurm
```

Inspect the log to see whether a real-video archive was found.  If you need to
identify the correct filename manually:

```bash
PYTHONPATH=src python -m face_fft.data.download_genvidbench --list-files
```

**Step 2 — train**

```bash
sbatch scripts/train_combined.slurm
```

The best checkpoint is saved to `checkpoints/combined/best_combined_{MODEL_TYPE}_{PIPELINE_MODE}.pt`
(e.g. `best_combined_r3d_18_fft_learnable_mask.pt`).

**Using H100 instead of A100**

Change one line in `scripts/train_combined.slurm`:

```diff
-#SBATCH --gres=gpu:a100:1
+#SBATCH --gres=gpu:h100:1
```

---

### Ablation studies

All ablation knobs live at the top of `src/face_fft/training/train_combined.py`.

#### 1. With / without 3D-FFT

The `PIPELINE_MODE` constant controls whether the spectral transform is applied.

| Goal | Setting |
|------|---------|
| Full pipeline (3D-FFT + learnable filter) | `PipelineMode.FFT_LEARNABLE_MASK` |
| 3D-FFT only, no filter | `PipelineMode.FFT_NO_MASK` |
| **No FFT** (pixel baseline) | `PipelineMode.PIXEL_BASELINE` |

```python
# src/face_fft/training/train_combined.py
PIPELINE_MODE = PipelineMode.PIXEL_BASELINE   # disable FFT
```

Submit three jobs (one per mode) and compare val loss and cross-generator F1.

#### 2. With / without learnable high-pass filter

The learnable spectral mask is a per-voxel sigmoid gate that sits between the
3D-FFT output and the CNN.  It is only active in `FFT_LEARNABLE_MASK` mode.

| Goal | Setting |
|------|---------|
| **With** learnable filter | `PipelineMode.FFT_LEARNABLE_MASK` |
| Without filter (FFT only) | `PipelineMode.FFT_NO_MASK` |

```python
PIPELINE_MODE = PipelineMode.FFT_NO_MASK   # FFT but no learnable gate
```

This isolates the contribution of the gate from the FFT representation itself.

#### 3. Architecture ablation

```python
MODEL_TYPE = "compact"    # lightweight ~0.5M params
MODEL_TYPE = "r3d_18"     # full 3D ResNet-18
MODEL_TYPE = "mc3_18"     # mixed 3D/2D convolutions
MODEL_TYPE = "r2plus1d_18"  # separable (2+1)D convolutions
```

#### Recommended ablation matrix

Run one job per row; compare cross-generator F1 on the GenVidBench held-out set
using `eval_genvidbench.py`.

| Run | PIPELINE_MODE | MODEL_TYPE | Notes |
|-----|--------------|------------|-------|
| A | `FFT_LEARNABLE_MASK` | `r3d_18` | Full pipeline (baseline) |
| B | `FFT_NO_MASK` | `r3d_18` | No learnable filter |
| C | `PIXEL_BASELINE` | `r3d_18` | No FFT at all |
| D | `FFT_LEARNABLE_MASK` | `compact` | Lightweight variant |
| E | `FFT_LEARNABLE_MASK` | `r2plus1d_18` | Separable 3D variant |

Save each checkpoint under a unique name by editing `CHECKPOINT_NAME` in
`train_combined.py` before each submission, then load them in
`eval_genvidbench.py` (edit `weights_path` there).

---

### Automated sweep (recommended)

Instead of running individual ablation jobs, use `run_sweep.py` to train and
evaluate all 12 combinations in a single job:

```bash
sbatch scripts/run_sweep.slurm
```

This produces `results/sweep_results.csv` with one row per combination:

| Column | Description |
|--------|-------------|
| `pipeline_mode` | `pixel_baseline` / `fft_no_mask` / `fft_learnable_mask` |
| `model_type` | `compact` / `r3d_18` / `mc3_18` / `r2plus1d_18` |
| `num_params` | Trainable parameter count |
| `train_time_sec` | Wall-clock training seconds |
| `da_accuracy/precision/recall/f1/auc` | DeepAction test-set metrics |
| `gvb_sora/kling/opensora_det` | GenVidBench fake detection rate per generator |
| `gvb_overall_det` | GenVidBench average fake detection rate |

**Resuming a preempted sweep**: rows already written to the CSV are skipped on
re-submission, so the job is idempotent.

**Eval-only mode** (re-evaluate existing checkpoints without retraining):

```bash
PYTHONPATH=src python -m face_fft.training.run_sweep --eval-only
```

---

### Notes on `train_deepaction.py` vs `train_combined.py`

`train_deepaction.py` trains on DeepAction only and uses hardcoded paths.
`train_combined.py` supersedes it for the joint-dataset experiment but is
otherwise identical in structure — same `FaceFFTPipeline`, same `Trainer`,
same loss function (`BCEWithLogitsLoss`).  Both can co-exist; use
`train_deepaction.py` for DeepAction-only ablations and `train_combined.py`
for the full mixed-data experiments.
