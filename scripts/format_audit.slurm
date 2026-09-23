#!/bin/bash
#SBATCH --job-name=gvb_format_audit
#SBATCH --account=uva-mira
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=/sfs/weka/scratch/rjr6zk/face-fft/logs/gvb_format_audit_%j.out

# Step 1 of GenVidBench onboarding: per-source container/codec audit plus
# single-field real-vs-fake shortcut AUCs. Metadata only; no GPU.

set -euo pipefail
REPO=/sfs/weka/scratch/rjr6zk/face-fft
GVB="${REPO}/src/face_fft/data/genvidbench_dataset"
cd "${REPO}"
source .venv/bin/activate

echo "Job started on $(hostname) at $(date)"
PYTHONPATH=src python bin/format_audit.py \
    --root "${GVB}/extracted" \
    --labels "${GVB}/raw/GenVidBench/Pair1_labels.txt" "${GVB}/raw/GenVidBench/Pair2_labels.txt" \
    --per_source 400 \
    --out_csv /standard/uva-mira-drive/3d-fft_results/genvidbench_format_audit.csv
echo "Job finished at $(date)"
