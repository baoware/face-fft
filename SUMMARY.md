# Face-FFT: Combined Training on DeepAction + GenVidBench

This document is the starting point for anyone new to this project.
It covers environment setup, data download, training, and evaluation end-to-end
on UVA Rivanna HPC.

---

## Quick-reference: scripts in order

| Step | Script | What it does |
|------|--------|--------------|
| 1 | *(manual)* | Clone repo and set up Python environment |
| 2 | *(manual)* | Set `HF_TOKEN` in your environment |
| 3 | `sbatch scripts/download_deepaction.slurm` | Download DeepAction v1 dataset |
| 4 | `sbatch scripts/download_genvidbench_full.slurm` | Download GenVidBench dataset |
| 5a | `sbatch scripts/run_sweep.slurm` | **Recommended** — train + evaluate all 12 ablation combos |
| 5b | `sbatch scripts/train_combined.slurm` | Alternative — train one specific (mode, arch) combination |
| 6 | `PYTHONPATH=src python -m face_fft.eval.eval_genvidbench` | Evaluate a checkpoint on GenVidBench |
| 6 | `PYTHONPATH=src python -m face_fft.eval.deepaction_evaluator` | Evaluate a checkpoint on DeepAction |

Steps 3 and 4 can be submitted in parallel — they are independent.
Step 5 depends on both completing successfully.

---

## Step 1 — Clone and set up the environment

```bash
git clone https://github.com/baoware/face-fft.git
cd face-fft

# Install all dependencies (requires uv)
uv sync --dev

# Verify the install
uv run pytest
```

All subsequent commands assume the virtualenv is active:

```bash
source .venv/bin/activate
```

---

## Step 2 — Set your HuggingFace token

Both datasets require a HuggingFace account with access granted.

```bash
export HF_TOKEN=hf_XXXXXXXXXXXXXXXXXXXX
```

Add this to your `~/.bashrc` or pass it inline when submitting Slurm jobs:

```bash
HF_TOKEN=hf_XXXX sbatch scripts/download_deepaction.slurm
```

Alternatively, create a `.env` file at the repo root:

```
HF_TOKEN=hf_XXXXXXXXXXXXXXXXXXXX
```

The download scripts load `.env` automatically via `python-dotenv`.

---

## Step 3 — Download DeepAction v1

**Script:** `scripts/download_deepaction.slurm`
**Partition:** `standard` (no GPU needed)
**Time:** up to 12 hours
**Disk:** ~40–60 GB

```bash
sbatch scripts/download_deepaction.slurm
```

This downloads the full `faridlab/deepaction_v1` snapshot to:

```
/scratch/rjr6zk/face-fft/src/face_fft/data/deepaction_dataset/
  Pexels/          ← real videos (source of truth, label=0)
  CogVideoX5B/     ← AI-generated fakes (label=1)
  Veo/
  RunwayML/
  ... (other generator folders)
```

Check progress:

```bash
tail -f slurm_files/deepaction_download_<jobid>.out
squeue -u $USER
```

---

## Step 4 — Download GenVidBench

**Script:** `scripts/download_genvidbench_full.slurm`
**Partition:** `standard` (no GPU needed)
**Time:** up to 12 hours

```bash
sbatch scripts/download_genvidbench_full.slurm
```

This downloads the synthetic archives (Sora, Kling, OpenSora) and attempts to
find and download a real-video archive from the same HuggingFace repo.

Output directory:

```
/scratch/rjr6zk/face-fft/src/face_fft/data/genvidbench_dataset/
  Sora/       ← OpenAI Sora fakes (label=1)
  Kling/      ← Kling fakes (label=1)
  OpenSora/   ← OpenSora fakes (label=1)
  Real/       ← real videos (label=0) — only if available in the repo
```

**If `Real/` is not downloaded**, training falls back automatically to using
DeepAction `Pexels/` videos as the real-class source. You will see this message
in the training log:

```
GVB Real/ absent — using DeepAction Pexels as real source for GVB.
```

This is expected and valid. Download Step 3 must finish first for the fallback to work.

To inspect what files are available in the GenVidBench repo before downloading:

```bash
PYTHONPATH=src python -m face_fft.data.download_genvidbench --list-files
```

---

## Step 5a — Run the full ablation sweep (recommended)

**Script:** `scripts/run_sweep.slurm`
**GPU:** A100 (change to `gpu:h100:1` for H100)
**Time:** up to 4 days (72 h per run × 12 runs)

```bash
sbatch scripts/run_sweep.slurm
```

This trains and evaluates all **12 combinations** of pipeline mode × model type:

| Pipeline modes | Model types |
|---------------|-------------|
| `pixel_baseline` — raw pixels, no FFT | `compact` |
| `fft_no_mask` — 3D-FFT, no learnable gate | `r3d_18` |
| `fft_learnable_mask` — 3D-FFT + learnable spectral gate | `mc3_18` |
| | `r2plus1d_18` |

Results are written incrementally to:

```
results/sweep_results.csv
```

One row per combination. The job is **idempotent**: if preempted, re-submit and
completed rows are skipped automatically.

To use H100:

```diff
-#SBATCH --gres=gpu:a100:1
+#SBATCH --gres=gpu:h100:1
```

---

## Step 5b — Train a single combination (alternative)

**Script:** `scripts/train_combined.slurm`
**GPU:** A100 (or H100, same diff line as above)
**Time:** up to 72 hours

Edit the two constants at the top of `src/face_fft/training/train_combined.py`:

```python
PIPELINE_MODE = PipelineMode.FFT_LEARNABLE_MASK   # or PIXEL_BASELINE / FFT_NO_MASK
MODEL_TYPE    = "r3d_18"                           # or compact / mc3_18 / r2plus1d_18
```

Then submit:

```bash
sbatch scripts/train_combined.slurm
```

Checkpoint is saved to:

```
checkpoints/combined/best_combined_<MODEL_TYPE>_<PIPELINE_MODE>.pt
```

e.g. `checkpoints/combined/best_combined_r3d_18_fft_learnable_mask.pt`

---

## Step 6 — Evaluate

### GenVidBench detection rate

Edit `src/face_fft/eval/eval_genvidbench.py`:
- Set `DATA_ROOT` to the GenVidBench dataset path
- Set `weights_path` to your checkpoint

Then run:

```bash
PYTHONPATH=src python -m face_fft.eval.eval_genvidbench
```

Outputs per-generator fake detection rate (Sora / Kling / OpenSora / Overall).

### DeepAction held-out test set

Edit `src/face_fft/eval/deepaction_evaluator.py`:
- Set `DATA_ROOT` and `WEIGHTS_PATH`

Then run:

```bash
PYTHONPATH=src python -m face_fft.eval.deepaction_evaluator
```

Outputs accuracy, precision, recall, F1, AUC — overall and per-generator.

---

## Ablation studies

### What the knobs control

| Knob | Effect |
|------|--------|
| `PIPELINE_MODE = PipelineMode.PIXEL_BASELINE` | Disables 3D-FFT entirely — model sees raw pixels |
| `PIPELINE_MODE = PipelineMode.FFT_NO_MASK` | Enables 3D-FFT but no learnable spectral gate |
| `PIPELINE_MODE = PipelineMode.FFT_LEARNABLE_MASK` | Full pipeline: 3D-FFT + per-voxel learnable gate |

Comparing `FFT_LEARNABLE_MASK` vs `PIXEL_BASELINE` isolates the value of the
spectral representation. Comparing `FFT_LEARNABLE_MASK` vs `FFT_NO_MASK` isolates
the contribution of the learnable gate on top of the FFT.

### Using the sweep (easiest)

The sweep runs all combinations automatically and writes `results/sweep_results.csv`.
Open it in Excel / pandas to compare. Key columns:

| Column | Meaning |
|--------|---------|
| `da_f1` | F1 on DeepAction held-out test set |
| `da_auc` | ROC-AUC on DeepAction held-out test set |
| `gvb_overall_det` | Average fake detection rate across Sora/Kling/OpenSora |

### Running individual ablation jobs with `train_combined.slurm`

Set `PIPELINE_MODE` and `MODEL_TYPE` in `train_combined.py`, submit, then evaluate.
The checkpoint name includes both values so runs never overwrite each other.

---

## Files added in this integration

| File | Purpose |
|------|---------|
| `src/face_fft/data/genvidbench.py` | `GenVidBenchDataset` + `get_genvidbench_splits()` |
| `src/face_fft/data/download_genvidbench.py` | Download script with `--include-real` / `--list-files` |
| `src/face_fft/training/train_combined.py` | Train on DeepAction + GenVidBench mixture |
| `src/face_fft/training/run_sweep.py` | Full sweep → `results/sweep_results.csv` |
| `scripts/download_deepaction.slurm` | Slurm: download DeepAction v1 |
| `scripts/download_genvidbench_full.slurm` | Slurm: download GenVidBench (synth + real) |
| `scripts/train_combined.slurm` | Slurm: single combined training run |
| `scripts/run_sweep.slurm` | Slurm: full 12-combination sweep |
