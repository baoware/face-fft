#!/bin/bash
#
# scripts/run_sweep.slurm
#
# Full ablation sweep: trains all (pipeline_mode × model_type) combinations
# on the combined DeepAction + GenVidBench dataset, then evaluates each on:
#   - DeepAction held-out test set  (accuracy, precision, recall, F1, AUC)
#   - GenVidBench fakes             (per-generator detection rate)
#
# Results are written incrementally to results/sweep_results.csv.
# If the job is preempted, re-submit and already-completed rows are skipped.
#
# PREREQUISITES
#   1. DeepAction and GenVidBench datasets extracted (see download_*.slurm scripts)
#   2. Python venv activated with all dependencies installed
#
# FLAGS
#   --eval-only   Skip training, evaluate existing checkpoints only.
#                 Useful for re-running evaluation after checkpoint is saved.
#
# USAGE
#   sbatch scripts/run_sweep.slurm
#
# To use H100 instead of A100:
#   #SBATCH --gres=gpu:h100:1

#SBATCH -A uva-mira
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH -t 72:00:00
#SBATCH -J face_fft_sweep
#SBATCH -o slurm_files/sweep_%j.out
#SBATCH -e slurm_files/sweep_%j.err
#SBATCH --mem=64000

mkdir -p slurm_files results logs

echo "Job started on $(hostname) at $(date)"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"
echo "----------"

source .venv/bin/activate

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

PYTHONPATH=src python -m face_fft.training.run_sweep \
    --csv results/sweep_results.csv

echo "Job finished at $(date)"
