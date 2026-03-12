#!/bin/bash
# scripts/train.slurm
#
# Slurm job for model training on Rivanna.
#
# PREREQUISITES
#   1. Run scripts/preprocess.sh BEFORE submitting this job.
#   2. Verify that data/real/ and data/synth_cogvideox/ contain .pt files.
#
# USAGE
#   sbatch scripts/train.slurm

#SBATCH -A uva-mira
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -t 72:00:00
#SBATCH -J face-fft-train
#SBATCH -o slurm_files/train_%j.out
#SBATCH -e slurm_files/train_%j.err
#SBATCH --mem=64000

module purge
module load apptainer

# ---------------------------------------------------------------------------
# Configuration — edit these to match your Rivanna paths
# ---------------------------------------------------------------------------

APPTAINER="/apps/software/standard/core/apptainer/1.3.4/bin/apptainer"
CODEDIR="/home/${USER}/home/face-fft"
IMAGE="${CODEDIR}/apptainer/face_fft.sif"

REAL_DATA_DIR="/mnt/code/data/real"
SYNTH_DATA_DIR="/mnt/code/data/synth_cogvideox"
CHECKPOINT_DIR="/mnt/code/checkpoints"

EPOCHS=20
BATCH_SIZE=16
LR=1e-3

SAVE_PATH="${CHECKPOINT_DIR}/best_model_cogvideox.pt"

# ---------------------------------------------------------------------------

mkdir -p slurm_files
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Training on CogVideoX synthetic data"

${APPTAINER} exec \
    --nv \
    --bind "${CODEDIR}:/mnt/code" \
    --env PYTHONPATH=/mnt/code/src \
    "${IMAGE}" \
    python3 /mnt/code/bin/train.py \
        --real_dir    "${REAL_DATA_DIR}" \
        --synth_dir   "${SYNTH_DATA_DIR}" \
        --epochs      "${EPOCHS}" \
        --batch_size  "${BATCH_SIZE}" \
        --lr          "${LR}" \
        --save_path   "${SAVE_PATH}"

echo "Job finished: $(date)"
