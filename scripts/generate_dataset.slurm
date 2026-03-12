#!/bin/bash
# scripts/generate_dataset.slurm
#
# Slurm job for paired real/synthetic dataset generation on Rivanna.
#
# PREREQUISITES
#   1. Run scripts/download_checkpoints.sh BEFORE submitting this job.
#   2. Download Kaggle dataset to ${CODEDIR}/src/face_fft/data/raw_videos/:
#      mkdir -p src/face_fft/data/raw_videos
#      kaggle datasets download -d tapakah68/selfies-and-video-dataset-4-000-people --unzip -p src/face_fft/data/raw_videos
#   3. Build and upload the Apptainer image (see apptainer/README.md)
#
# USAGE
#   sbatch scripts/generate_dataset.slurm

#SBATCH -A uva-mira
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -t 72:00:00
#SBATCH -J face-fft-generate
#SBATCH -o slurm_files/generate_%j.out
#SBATCH -e slurm_files/generate_%j.err
#SBATCH --mem=256000

module purge
module load apptainer

# ---------------------------------------------------------------------------
# Configuration — edit these to match your Rivanna paths
# ---------------------------------------------------------------------------

APPTAINER="/apps/software/standard/core/apptainer/1.3.4/bin/apptainer"
CODEDIR="/home/${USER}/face-fft"
IMAGE="${CODEDIR}/apptainer/face_fft.sif"
CHECKPOINT_DIR="/scratch/${USER}/checkpoints"

# Generator: "cogvideox" or "wan"
GENERATOR="cogvideox"

SOURCE_DIR="/mnt/code/src/face_fft/data/raw_videos"
REAL_OUT_DIR="/mnt/code/src/face_fft/data/real"
SYNTH_OUT_DIR="/mnt/code/src/face_fft/data/synth_${GENERATOR}"

PROMPT="A highly realistic person's face, talking slightly, photorealistic video."
NUM_FRAMES=16
GEN_HEIGHT=480
GEN_WIDTH=480
NUM_INFERENCE_STEPS=20

# ---------------------------------------------------------------------------

mkdir -p slurm_files
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Generator: ${GENERATOR}"

${APPTAINER} exec \
    --nv \
    --bind "${CODEDIR}:/mnt/code" \
    --bind "${CHECKPOINT_DIR}:/mnt/checkpoints" \
    --env HF_HOME=/mnt/checkpoints \
    --env TRANSFORMERS_OFFLINE=1 \
    --env HF_DATASETS_OFFLINE=1 \
    --env PYTHONPATH=/mnt/code/src \
    "${IMAGE}" \
    python3 /mnt/code/bin/generate_dataset.py \
        --source_dir      "${SOURCE_DIR}" \
        --real_out_dir    "${REAL_OUT_DIR}" \
        --synth_out_dir   "${SYNTH_OUT_DIR}" \
        --generator       "${GENERATOR}" \
        --prompt          "${PROMPT}" \
        --num_frames      "${NUM_FRAMES}" \
        --gen_height      "${GEN_HEIGHT}" \
        --gen_width       "${GEN_WIDTH}" \
        --num_inference_steps "${NUM_INFERENCE_STEPS}" \
        --cache_dir       /mnt/checkpoints \
        --local_files_only

echo "Job finished: $(date)"
