#!/bin/bash
#
# scripts/download_deepaction.slurm
#
# Download the full DeepAction v1 dataset from HuggingFace
# (faridlab/deepaction_v1 — Pexels real videos + multiple AI generator folders).
#
# Run this once before any training job that uses DeepAction data.
# The download is ~40-60 GB; the standard partition is used (no GPU needed).
#
# PREREQUISITES
#   HF_TOKEN must be set in your environment or in a .env file at the repo root.
#   Request access to faridlab/deepaction_v1 on HuggingFace if prompted.
#
# USAGE
#   export HF_TOKEN=hf_XXXX
#   sbatch scripts/download_deepaction.slurm

#SBATCH -A uva-mira
#SBATCH -p standard
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -J deepaction_download
#SBATCH -o slurm_files/deepaction_download_%j.out
#SBATCH -e slurm_files/deepaction_download_%j.err

mkdir -p slurm_files

echo "Job started on $(hostname) at $(date)"
echo "----------"

source .venv/bin/activate

export HF_TOKEN="${HF_TOKEN}"

echo "Downloading DeepAction v1 from HuggingFace..."
PYTHONPATH=src python src/face_fft/data/download_deepaction.py

echo "Job finished at $(date)"
