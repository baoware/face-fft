#!/bin/bash
# scripts/eval.slurm
#
# Slurm job for model evaluation on Rivanna.
# Evaluates in-distribution (CogVideoX → CogVideoX) and cross-generator
# generalization (CogVideoX model → Wan data).
#
# OPTIONAL evaluation-only job. Re-run evaluation without retraining.
#
# PREREQUISITES
#   1. Run scripts/preprocess.sh BEFORE submitting this job.
#   2. Run scripts/train.slurm (or train.py locally) BEFORE this job.
#   3. Verify that src/face_fft/data/real/, src/face_fft/data/synth_cogvideox/,
#      and src/face_fft/data/synth_wan/ contain .pt files.
#   4. Verify that checkpoints/best_model_cogvideox.pt exists.
#
# USAGE
#   sbatch scripts/eval.slurm
#
# NOTE: train.slurm already includes evaluation. Use this script only if you need
# to re-evaluate an existing model checkpoint without retraining.

#SBATCH -A uva-mira
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -t 12:00:00
#SBATCH -J face-fft-eval
#SBATCH -o slurm_files/eval_%j.out
#SBATCH -e slurm_files/eval_%j.err
#SBATCH --mem=64000

module purge

source .venv/bin/activate

# ---------------------------------------------------------------------------
# Configuration — edit these to match your Rivanna paths
# ---------------------------------------------------------------------------

export PYTHONPATH=/home/$USER/face-fft/src

CODEDIR="/home/${USER}/home/face-fft"
IMAGE="${CODEDIR}/apptainer/face_fft.sif"

REAL_DATA_DIR="/mnt/code/src/face_fft/data/real"
SYNTH_COGVIDEOX_DIR="/mnt/code/src/face_fft/data/synth_cogvideox"
SYNTH_WAN_DIR="/mnt/code/src/face_fft/data/synth_wan"
CHECKPOINT_DIR="/mnt/code/checkpoints"

BATCH_SIZE=16

MODEL_PATH="${CHECKPOINT_DIR}/best_model_cogvideox.pt"

# ---------------------------------------------------------------------------

mkdir -p slurm_files
echo "Job started: $(date)"
echo "Node: $(hostname)"

# =============================================================================
# STEP 1: EVALUATE — in-distribution (CogVideoX → CogVideoX)
# =============================================================================

echo ""
echo "==> [1/2] Evaluating in-distribution (CogVideoX test split)..."
echo ""

python3 bin/evaluate.py \
    --real_dir   "${REAL_DATA_DIR}" \
    --synth_dir  "${SYNTH_COGVIDEOX_DIR}" \
    --model_path "${MODEL_PATH}" \
    --batch_size "${BATCH_SIZE}"

# =============================================================================
# STEP 2: EVALUATE — cross-generator generalization (CogVideoX model → Wan data)
# =============================================================================
# This is the core research question: does the spectral detector generalize
# to a generator it has never seen during training?

echo ""
echo "==> [2/2] Evaluating cross-generator generalization (CogVideoX model → Wan data)..."
echo ""

python3 /mnt/code/bin/evaluate.py \
    --real_dir   "${REAL_DATA_DIR}" \
    --synth_dir  "${SYNTH_WAN_DIR}" \
    --model_path "${MODEL_PATH}" \
    --batch_size "${BATCH_SIZE}"

# =============================================================================

echo ""
echo "Evaluation complete: $(date)"
