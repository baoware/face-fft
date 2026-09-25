#!/bin/bash
# scripts/ingest_video_pairs.slurm
#
# Slurm job for ingesting pre-existing real/synthetic video pairs on Rivanna.
# CPU-only — no GPU allocation required.
#
# PREREQUISITES
#   1. Place real and synthetic videos in:
#        src/face_fft/data/raw_video_pairs/real/
#        src/face_fft/data/raw_video_pairs/synthetic/
#      Videos are matched by filename stem.
#
#   2. Set DATASET_NAME below to identify the source dataset.
#
#   3. Install dependencies: uv sync --dev
#
# USAGE
#   sbatch scripts/ingest_video_pairs.slurm
#
# OUTPUT
#   Real tensors:      src/face_fft/data/real/
#   Synthetic tensors: src/face_fft/data/synth_{DATASET_NAME}/
#
# After ingestion, submit train.slurm to train on the resulting .pt files.

#SBATCH -A uva-mira
#SBATCH -p standard
#SBATCH -c 8
#SBATCH -t 4:00:00
#SBATCH -J face-fft-ingest
#SBATCH -o slurm_files/ingest_%j.out
#SBATCH -e slurm_files/ingest_%j.err
#SBATCH --mem=32000

module purge

source .venv/bin/activate

# ---------------------------------------------------------------------------
# Configuration — edit these to match your Rivanna paths
# ---------------------------------------------------------------------------

export PYTHONPATH=/home/$USER/face-fft/src

# Short identifier for this external dataset (no spaces)
DATASET_NAME="faceforensics"

INPUT_DIR="src/face_fft/data/raw_video_pairs"
REAL_PT_DIR="src/face_fft/data/real"
SYNTH_PT_DIR="src/face_fft/data/synth_${DATASET_NAME}"

NUM_FRAMES=16
TARGET_SIZE=256

# ---------------------------------------------------------------------------

mkdir -p "${REAL_PT_DIR}"
mkdir -p "${SYNTH_PT_DIR}"
mkdir -p slurm_files

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Dataset: ${DATASET_NAME}"
echo ""

python3 bin/ingest_video_pairs.py \
    --input_dir     "${INPUT_DIR}" \
    --real_out_dir  "${REAL_PT_DIR}" \
    --synth_out_dir "${SYNTH_PT_DIR}" \
    --dataset_name  "${DATASET_NAME}" \
    --num_frames    "${NUM_FRAMES}" \
    --target_size   "${TARGET_SIZE}"

echo ""
echo "Job finished: $(date)"
