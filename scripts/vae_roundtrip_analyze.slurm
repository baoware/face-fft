#!/bin/bash
#SBATCH --job-name=vae_rt_analyze
#SBATCH --account=uva-mira
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/vae_rt_analyze_%j.out

# VAE round-trip, analysis stage (CPU): per-clip measurements, then the summary table,
# curves CSV and three figures under ${RESULTS}/${VIEW}/. Resumable; safe to re-run
# while the GPU stage is still filling in clips.
# Optional: queue it behind the GPU array (afterany, so one failed VAE does not block the rest):
#   jid=$(sbatch --parsable scripts/vae_roundtrip.slurm)
#   sbatch --dependency=afterany:$jid scripts/vae_roundtrip_analyze.slurm
# Variables: VIEW (resize) OUT (<repo>/roundtrip_frames) RESULTS REPO
set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR}}"
cd "${REPO}"
source .venv/bin/activate
VIEW="${VIEW:-resize}"; OUT="${OUT:-${REPO}/roundtrip_frames}"
RESULTS="${RESULTS:-/standard/uva-mira-drive/3d-fft_results/vae_roundtrip}"
echo "view=${VIEW} frames=${OUT} results=${RESULTS} on $(hostname) at $(date)"
PYTHONPATH=src python bin/vae_roundtrip.py analyze --view "${VIEW}" --out "${OUT}" --results "${RESULTS}" --workers 14
echo "done at $(date)"
