#!/bin/bash
#SBATCH --job-name=lossless_analyze
#SBATCH --account=uva-mira
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/sfs/weka/scratch/rjr6zk/face-fft/logs/lossless_analyze_%j.out

# Lossless test, analysis stage: codec sweep x views x statistics, T=24 (all models)
# and T=48 (SVD-xt has only 25 frames, so it is skipped there and logged).
set -euo pipefail
REPO=/sfs/weka/scratch/rjr6zk/face-fft
cd "${REPO}"
source .venv/bin/activate
for T in 24 48; do
    PYTHONPATH=src python bin/lossless_test.py analyze --T "${T}" --workers 16
done
