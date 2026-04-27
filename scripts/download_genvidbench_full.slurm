#!/bin/bash
#
# scripts/download_genvidbench_full.slurm
#
# Download the complete GenVidBench dataset (synthetic + real video archives)
# from HuggingFace.  Submit this job before running train_combined.slurm.
#
# USAGE
#   sbatch scripts/download_genvidbench_full.slurm
#
# The job downloads:
#   - Synthetic archives: Sora, Kling, OpenSora  (always)
#   - Real-video archive: Real/                   (--include-real flag)
#
# If the HuggingFace repo does not contain a real-video archive the script
# prints what IS available so you can locate the correct filename.
# In that case train_combined.py will fall back to DeepAction Pexels reals.

#SBATCH --job-name=gvb_download
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/gvb_download_%j.out

mkdir -p logs

echo "Job started on $(hostname) at $(date)"
echo "----------"

source .venv/bin/activate

export HF_TOKEN="${HF_TOKEN}"          # set in your environment or .env file

# List all repo files first so you can inspect what is available
echo "==> Listing all GenVidBench repo files ..."
PYTHONPATH=src python -m face_fft.data.download_genvidbench --list-files

echo ""
echo "==> Downloading synthetic archives + real-video archive ..."
PYTHONPATH=src python -m face_fft.data.download_genvidbench \
    --base_dir /scratch/rjr6zk/face-fft/src/face_fft/data \
    --include-real

echo "Job finished at $(date)"
