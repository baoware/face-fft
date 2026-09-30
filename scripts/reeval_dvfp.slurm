#!/bin/bash
#SBATCH --job-name=reeval_dvfp
#SBATCH --account=uva-mira
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=udc-an36-13
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/sfs/weka/scratch/rjr6zk/face-fft/logs/reeval_dvfp_%j.out

# Score every existing cached run under MM-Det's DVF protocol (DVFp: 7 test generators,
# no SVD fakes, no YouTube reals, mean per-generator AUC). Written to eval_dvfp.json so
# each run's eval.json is left untouched. Runs trained on DVF itself are skipped here.
set -euo pipefail
REPO=/sfs/weka/scratch/rjr6zk/face-fft
cd "${REPO}"
source .venv/bin/activate
for run in /standard/uva-mira-drive/3d-fft_checkpoints/cached/*/; do
    name=$(basename "${run}")
    case "${name}" in dvftrain_*|off_*|logo_*|gvp_*) continue ;; esac   # new runs score DVFp themselves
    [ -f "${run}config.json" ] && [ -f "${run}best.pt" ] && [ ! -f "${run}eval_dvfp.json" ] || continue
    PYTHONPATH=src python -m face_fft.eval.eval_cached --cache "${REPO}/cache/v1" --run_dir "${run}" \
        --tests DVFp --out "${run}eval_dvfp.json" | grep -E "^\[DVFp\] mean" | sed "s|^|${name}: |"
done
