# Scripts

This directory contains the Face-FFT pipeline entry points and deployment helpers.

## Overview

### Pipeline Orchestration

These shell and Slurm scripts coordinate the full workflow.

- **`preprocess.sh`** — Local dataset preprocessing (runs both generator models)
- **`train.slurm`** — Slurm batch job for training + evaluation (in-distribution and cross-generator)
- **`eval.slurm`** — _Optional_ Slurm batch job for evaluation only (use if you need to re-evaluate without retraining)
- **`generate_dataset.slurm`** — Slurm batch job for dataset generation (alternative to `preprocess.sh` for HPC)
- **`download_checkpoints.sh`** — Pre-download HuggingFace model weights (HPC only)

---

## Prerequisites

### 1. Environment Setup

Install dependencies and verify the environment:

```bash
uv sync --dev
uv run pytest
```

### 2. Kaggle Dataset

Download the "Selfies and Video" dataset (4,000 people):

```bash
mkdir -p src/face_fft/data/raw_videos
kaggle datasets download -d tapakah68/selfies-and-video-dataset-4-000-people --unzip -p src/face_fft/data/raw_videos
```

> **Note:** Requires Kaggle CLI configured with credentials (`~/.kaggle/kaggle.json`).
> See: https://www.kaggle.com/settings/account

## Running the Pipeline

### Option A: Local Execution (Recommended for Development)

Run all three pipeline stages sequentially on your local machine.

#### Step 1: Preprocess (Dataset Generation)

Generates paired real/synthetic datasets using both CogVideoX and Wan models:

```bash
bash scripts/preprocess.sh
```

**Outputs:**

- `src/face_fft/data/real/*.pt` — Preprocessed real video tensors
- `src/face_fft/data/synth_cogvideox/*.pt` — CogVideoX-generated videos
- `src/face_fft/data/synth_wan/*.pt` — Wan 2.2-generated videos

**Requirements:** ~24 GB GPU VRAM

**Approx. time:** 2–4 hours per generator (depends on dataset size)

#### Step 2: Train

Train the spectral CNN on CogVideoX synthetic data:

```bash
uv run python bin/train.py \
    --real_dir src/face_fft/data/real \
    --synth_dir src/face_fft/data/synth_cogvideox \
    --epochs 20 \
    --batch_size 16 \
    --lr 1e-3 \
    --save_path checkpoints/best_model_cogvideox.pt
```

**Outputs:**

- `checkpoints/best_model_cogvideox.pt` — Trained model

**Approx. time:** 30 minutes–2 hours (depends on dataset size and GPU)

#### Step 3: Evaluate

Evaluate in-distribution and cross-generator generalization:

```bash
# In-distribution: CogVideoX model on CogVideoX test split
uv run python bin/evaluate.py \
    --real_dir src/face_fft/data/real \
    --synth_dir src/face_fft/data/synth_cogvideox \
    --model_path checkpoints/best_model_cogvideox.pt \
    --batch_size 16

# Cross-generator: CogVideoX model on Wan test split
uv run python bin/evaluate.py \
    --real_dir src/face_fft/data/real \
    --synth_dir src/face_fft/data/synth_wan \
    --model_path checkpoints/best_model_cogvideox.pt \
    --batch_size 16
```

**Outputs:** Metrics in stdout (F1, accuracy, confusion matrix, etc.)

---

### Option B: HPC Execution (Rivanna)

For large-scale production runs on UVA Rivanna.

#### Step 0: Build Apptainer Container

```bash
cd apptainer
bash build.sh
cd ..
```

Upload the image to Rivanna:

```bash
scp apptainer/face_fft.sif hpc.virginia.edu:/home/${USER}/face-fft/apptainer/
```

#### Step 1: Pre-download Checkpoints (Interactive, Login Node)

Pre-download model weights to `/scratch/` **once before submitting any batch jobs**. This prevents compute nodes from downloading weights during GPU-billed time and avoids network access issues on restricted compute nodes.

The script downloads both:
- `THUDM/CogVideoX-2b` (~15 GB)
- `Wan-AI/Wan2.2-I2V-A14B` (~40 GB)

Run on a login node or inside an `ijob` session:

```bash
bash scripts/download_checkpoints.sh
```

The script sets `HF_HOME=/scratch/${USER}/checkpoints` and uses the standard HuggingFace blob-hash cache layout that `from_pretrained(cache_dir=...)` expects.

**Options:**

- Override destination: `CHECKPOINT_DIR=/path/to/cache bash scripts/download_checkpoints.sh`
- Check progress: `du -sh /scratch/${USER}/checkpoints/hub/models--*`
- Re-run if purged: `/scratch` on Rivanna purges files after 90 days of inactivity

#### Step 2: Submit Dataset Generation Job

```bash
# Edit scripts/generate_dataset.slurm to set your allocation (-A flag)
sbatch scripts/generate_dataset.slurm
```

Check status:

```bash
squeue -u ${USER} | grep face-fft
tail -f slurm_files/generate_*.out
```

#### Step 3: Submit Training + Evaluation Job

Once dataset generation completes:

```bash
sbatch scripts/train.slurm
```

This job runs training followed by in-distribution and cross-generator evaluation. Check status:

```bash
squeue -u ${USER} | grep face-fft
tail -f slurm_files/train_*.out
```

**Optional:** If you need to re-evaluate an existing checkpoint without retraining, use `eval.slurm`:

```bash
sbatch scripts/eval.slurm
```

---

## Configuration

### Common Parameters

Edit the script headers to customize execution:

- **`GENERATOR`** — `cogvideox` or `wan` (default: `cogvideox`)
- **`GEN_HEIGHT`, `GEN_WIDTH`** — Generation resolution (default: 480×480)
- **`NUM_INFERENCE_STEPS`** — Diffusion denoising steps (default: 20)
- **`NUM_FRAMES`** — Target video length (default: 16)
- **`EPOCHS`, `BATCH_SIZE`, `LR`** — Training hyperparameters (default: 20, 16, 1e-3)

### HPC-Specific

Edit Slurm directives (lines starting with `#SBATCH`):

- **`-A <allocation-name>`** — Your HPC allocation (required)
- **`-t <hours:minutes:seconds>`** — Wall-clock time limit
- **`--mem=<MB>`** — Memory per node
- **`--gres=gpu:<count>`** — GPU count

---

## Output Structure

```
face-fft/
├── src/face_fft/data/
│   ├── raw_videos/           # Input: Kaggle dataset (user-downloaded)
│   ├── real/                 # Output: real video tensors
│   ├── synth_cogvideox/      # Output: CogVideoX-generated tensors
│   └── synth_wan/            # Output: Wan 2.2-generated tensors
├── checkpoints/
│   ├── best_model_cogvideox.pt
│   └── best_model_wan.pt     # (if trained on Wan data)
└── slurm_files/              # (HPC only)
    ├── generate_*.out
    ├── train_*.out
    └── eval_*.out
```

---

## Troubleshooting

### "No videos found in raw_videos/"

1. Check the Kaggle download completed: `ls src/face_fft/data/raw_videos/files/`
2. Verify dataset format: `.mp4` files should be nested as `files/<number>/<number>.mp4`
3. Ensure the correct dataset ID: `tapakah68/selfies-and-video-dataset-4-000-people`

### "huggingface-cli not found" / "CUDA out of memory"

For local runs:

- Activate the environment: `uv shell` or `source .venv/bin/activate`
- Reduce generation resolution: `--gen_height 320 --gen_width 320`

For HPC runs:

- Verify the Apptainer image was built and uploaded
- Check allocation name (`-A` flag in `.slurm` files)

### Model download errors on Slurm nodes

1. Pre-run `download_checkpoints.sh` on a login node
2. Verify `TRANSFORMERS_OFFLINE=1` is set in the Slurm script
3. Check `/scratch/${USER}/checkpoints/hub/` exists with model weights

---

## References

- **Face-FFT Paper:** [arxiv link]
- **CogVideoX:** https://huggingface.co/THUDM/CogVideoX-2b
- **Wan 2.2:** https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B
- **Rivanna HPC:** https://www.rc.virginia.edu/userinfo/rivanna/
