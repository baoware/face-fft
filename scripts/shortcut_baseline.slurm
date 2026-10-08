#!/bin/bash
#SBATCH --job-name=shortcut_baseline
#SBATCH --account=uva-mira
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/sfs/weka/scratch/rjr6zk/face-fft/logs/shortcut_baseline_%j.out

# Step 3: low-level-statistics baseline on the normalised cache, at T=8 and T=16.
set -euo pipefail
REPO=/sfs/weka/scratch/rjr6zk/face-fft
cd "${REPO}"
source .venv/bin/activate
for T in 8 16; do
    python bin/shortcut_baseline.py --cache "${REPO}/cache/v1" --T "${T}" \
        --out "/standard/uva-mira-drive/3d-fft_results/shortcut_baseline_T${T}.txt"
done
