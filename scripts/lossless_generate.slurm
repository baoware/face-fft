#!/bin/bash
#SBATCH --job-name=lossless_gen
#SBATCH --account=uva-mira
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100_80gb
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --array=1-3
#SBATCH --output=/sfs/weka/scratch/rjr6zk/face-fft/logs/lossless_gen_%A_%a.out

# Lossless test, generation stage: one array task per model.
#   1 = cogvideox (4x temporal VAE), 2 = wan (4x temporal VAE), 3 = svd (frame-wise)
# 80 GB A100: SVD decodes all 25 frames in one chunk (no decoder seams).
set -euo pipefail
REPO=/sfs/weka/scratch/rjr6zk/face-fft
cd "${REPO}"
source .venv/bin/activate
export HF_HOME=/sfs/weka/scratch/rjr6zk/hf-cache
export PYTORCH_ALLOC_CONF=expandable_segments:True
MODEL=$(echo cogvideox wan svd | cut -d' ' -f"${SLURM_ARRAY_TASK_ID}")
echo "task ${SLURM_ARRAY_TASK_ID}: ${MODEL} on $(hostname) at $(date)"
PYTHONPATH=src python bin/lossless_test.py generate --model "${MODEL}" --n 50
echo "done at $(date)"
