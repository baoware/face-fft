#!/bin/bash
#SBATCH --job-name=clip_cache
#SBATCH --account=uva-mira
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=1-20
#SBATCH --output=/sfs/weka/scratch/rjr6zk/face-fft/logs/clip_cache_%A_%a.out

# One array task per (pair, source) in scripts/cache_sources.txt.
set -euo pipefail
REPO=/sfs/weka/scratch/rjr6zk/face-fft
cd "${REPO}"
source .venv/bin/activate
read -r PAIR SOURCE CAPS < <(sed -n "${SLURM_ARRAY_TASK_ID}p" scripts/cache_sources.txt)
echo "task ${SLURM_ARRAY_TASK_ID}: ${PAIR}/${SOURCE} caps=${CAPS} on $(hostname) at $(date)"
python bin/build_cache.py \
    --manifest /standard/uva-mira-drive/3d-fft_results/manifests/manifest_v1.csv \
    --out "${REPO}/cache/v1" \
    --pair "${PAIR}" --source "${SOURCE}" --caps "${CAPS}"
echo "done at $(date)"
