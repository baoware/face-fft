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
#SBATCH --gres=gpu:a100:1
#SBATCH -c 4
#SBATCH -t 72:00:00
#SBATCH -J face-fft-generate
#SBATCH -o slurm_files/generate_%j.out
#SBATCH -e slurm_files/generate_%j.err
#SBATCH --mem=256000

module purge

source .venv/bin/activate

# ---------------------------------------------------------------------------
# Configuration — edit these to match your Rivanna paths
# ---------------------------------------------------------------------------

export HF_HOME=/scratch/$USER/checkpoints
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONPATH=/home/$USER/home/face-fft/src

CHECKPOINT_DIR="/scratch/${USER}/checkpoints/hub"

# Run for both generators via for loop:
for GENERATOR in "cogvideox" "wan"; do

    SOURCE_DIR="src/face_fft/data/raw_videos"
    REAL_OUT_DIR="src/face_fft/data/real"
    SYNTH_OUT_DIR="src/face_fft/data/synth_${GENERATOR}"

    PROMPT="A highly realistic person's face, talking slightly, photorealistic video."
    NUM_FRAMES=8
    GEN_HEIGHT=512
    GEN_WIDTH=512
    NUM_INFERENCE_STEPS=50

    # ---------------------------------------------------------------------------

    mkdir -p slurm_files
    echo "Job started: $(date)"
    echo "Node: $(hostname)"
    echo "Generator: ${GENERATOR}"

    python3 bin/generate_dataset.py \
        --source_dir      "${SOURCE_DIR}" \
        --real_out_dir    "${REAL_OUT_DIR}" \
        --synth_out_dir   "${SYNTH_OUT_DIR}" \
        --generator       "${GENERATOR}" \
        --prompt          "${PROMPT}" \
        --num_frames      "${NUM_FRAMES}" \
        --gen_height      "${GEN_HEIGHT}" \
        --gen_width       "${GEN_WIDTH}" \
        --num_inference_steps "${NUM_INFERENCE_STEPS}" \
        --cache_dir       "${CHECKPOINT_DIR}" \
        --local_files_only

    echo "Job finished: $(date)"

done
