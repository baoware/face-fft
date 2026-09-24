#!/bin/bash
#SBATCH --job-name=reeval_all
#SBATCH --account=uva-mira
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=udc-an36-13
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/sfs/weka/scratch/rjr6zk/face-fft/logs/reeval_all_%j.out

# Re-evaluate every cached-training run on all test sets, including GenVideo-Val,
# which was added after those runs were evaluated. Overwrites each run's eval.json.
set -euo pipefail
REPO=/sfs/weka/scratch/rjr6zk/face-fft
cd "${REPO}"
source .venv/bin/activate
for run in /standard/uva-mira-drive/3d-fft_checkpoints/cached/*/; do
    [ -f "${run}config.json" ] && [ -f "${run}best.pt" ] || continue
    echo "=== ${run} ($(date +%T))"
    PYTHONPATH=src python -m face_fft.eval.eval_cached --cache "${REPO}/cache/v1" --run_dir "${run}" \
        --tests Pair1,Pair2,DeepAction,6.7m,GenVideo | grep -E "^\[GenVideo\]|^wrote"
done
python bin/summarize_runs.py > /dev/null && echo "summary -> /standard/uva-mira-drive/3d-fft_results/runs_summary.md"
