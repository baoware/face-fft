#!/bin/bash
# scripts/train.slurm
#
# Slurm job for model training on Rivanna.
#
# PREREQUISITES
#   1. Run scripts/preprocess.sh BEFORE submitting this job.
#   2. Verify that src/face_fft/data/real/ and src/face_fft/data/synth_cogvideox/ contain .pt files.
#
# USAGE
#   sbatch scripts/train.slurm
#
# This job runs training followed by evaluation on both in-distribution
# (CogVideoX→CogVideoX) and cross-generator (CogVideoX→Wan) splits.
# Use eval.slurm to re-run evaluation only if needed.

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
CODEDIR="/home/${USER}/face-fft"
IMAGE="${CODEDIR}/apptainer/face_fft.sif"

REAL_DATA_DIR="/mnt/code/src/face_fft/data/real"
SYNTH_COGVIDEOX_DIR="/mnt/code/src/face_fft/data/synth_cogvideox"
SYNTH_WAN_DIR="/mnt/code/src/face_fft/data/synth_wan"
CHECKPOINT_DIR="/mnt/code/checkpoints"

EPOCHS=20
BATCH_SIZE=16
LR=1e-3

SAVE_PATH="${CHECKPOINT_DIR}/best_model_cogvideox.pt"

# ---------------------------------------------------------------------------

mkdir -p slurm_files
echo "Job started: $(date)"
echo "Node: $(hostname)"

# =============================================================================
# STEP 1: TRAIN
# =============================================================================

echo ""
echo "==> [1/3] Training on CogVideoX synthetic data..."
echo ""

${APPTAINER} exec \
    --nv \
    --bind "${CODEDIR}:/mnt/code" \
    --env PYTHONPATH=/mnt/code/src \
    "${IMAGE}" \
    python3 /mnt/code/bin/train.py \
        --real_dir    "${REAL_DATA_DIR}" \
        --synth_dir   "${SYNTH_COGVIDEOX_DIR}" \
        --epochs      "${EPOCHS}" \
        --batch_size  "${BATCH_SIZE}" \
        --lr          "${LR}" \
        --save_path   "${SAVE_PATH}"

# =============================================================================
# STEP 2: EVALUATE — in-distribution (CogVideoX → CogVideoX)
# =============================================================================

echo ""
echo "==> [2/3] Evaluating in-distribution (CogVideoX test split)..."
echo ""

${APPTAINER} exec \
    --nv \
    --bind "${CODEDIR}:/mnt/code" \
    --env PYTHONPATH=/mnt/code/src \
    "${IMAGE}" \
    python3 /mnt/code/bin/evaluate.py \
        --real_dir   "${REAL_DATA_DIR}" \
        --synth_dir  "${SYNTH_COGVIDEOX_DIR}" \
        --model_path "${SAVE_PATH}" \
        --batch_size "${BATCH_SIZE}"

# =============================================================================
# STEP 3: EVALUATE — cross-generator generalization (CogVideoX model → Wan data)
# =============================================================================
# This is the core research question: does the spectral detector generalize
# to a generator it has never seen during training?

echo ""
echo "==> [3/3] Evaluating cross-generator generalization (CogVideoX model → Wan data)..."
echo ""

${APPTAINER} exec \
    --nv \
    --bind "${CODEDIR}:/mnt/code" \
    --env PYTHONPATH=/mnt/code/src \
    "${IMAGE}" \
    python3 /mnt/code/bin/evaluate.py \
        --real_dir   "${REAL_DATA_DIR}" \
        --synth_dir  "${SYNTH_WAN_DIR}" \
        --model_path "${SAVE_PATH}" \
        --batch_size "${BATCH_SIZE}"

# =============================================================================

echo ""
echo "Job finished: $(date)"
