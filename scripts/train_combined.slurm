#!/bin/bash
#
# scripts/train_combined.slurm
#
# Train on combined DeepAction + GenVidBench data mixture.
#
# PREREQUISITES
#   1. DeepAction dataset extracted under:
#        src/face_fft/data/deepaction_dataset/
#   2. GenVidBench synthetic archives extracted under:
#        src/face_fft/data/genvidbench_dataset/
#      (Run scripts/download_genvidbench_full.slurm first)
#   3. HF_TOKEN set if network access is needed during the job.
#
# ABLATION VARIANTS
#   Edit PIPELINE_MODE and MODEL_TYPE in
#   src/face_fft/training/train_combined.py before submitting:
#
#   Pipeline mode (fft component):
#     pixel_baseline     -- no FFT, raw pixels into CNN
#     fft_no_mask        -- 3D-FFT into CNN, no learnable filter
#     fft_learnable_mask -- 3D-FFT + learnable per-voxel gate [default]
#
#   Model type (architecture):
#     compact | r3d_18 | mc3_18 | r2plus1d_18
#
# USAGE
#   sbatch scripts/train_combined.slurm
#
# To request an H100 instead of A100 change the gres line to:
#   #SBATCH --gres=gpu:h100:1

#SBATCH -A uva-mira
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 4
#SBATCH -t 72:00:00
#SBATCH -J face_fft_combined
#SBATCH -o slurm_files/train_combined_%j.out
#SBATCH -e slurm_files/train_combined_%j.err
#SBATCH --mem=64000

mkdir -p logs slurm_files results

echo "Job started on $(hostname) at $(date)"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"
echo "----------"

source .venv/bin/activate

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

PYTHONPATH=src python -m face_fft.training.train_combined

echo "Job finished at $(date)"
