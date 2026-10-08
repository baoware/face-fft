#!/usr/bin/env bash
# =============================================================================
# Face-FFT: VAE round-trip measurement (NEXT_STEPS step 1)
# Submits the GPU array (one task per VAE) and queues the CPU analysis behind it.
# =============================================================================
#
# Run on a Rivanna login node, from anywhere in the repo:
#
#   bash scripts/vae_roundtrip.sh                 # view=resize, 200 clips, 49 frames
#   VIEW=crop bash scripts/vae_roundtrip.sh       # native-resolution centre crop
#   N=50 bash scripts/vae_roundtrip.sh            # quick pass
#
# Variables (all optional, passed through to both jobs):
#   VIEW (resize)  N (200)  FRAMES (49)  OUT (/scratch/$USER/roundtrip_frames)
#   RESULTS (/standard/uva-mira-drive/3d-fft_results/vae_roundtrip)  HF_HOME
#   DA_ROOT (<repo>/src/face_fft/data/deepaction_dataset; must contain Pexels/<id>/*.mp4)
#
# The analysis job waits for every array task to END, not to succeed (afterany):
# if one VAE fails, the others are still analysed. It is resumable, so after
# fixing a failed VAE just run this script again -- finished clips are skipped.
# =============================================================================
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs

# sbatch exports the caller's environment by default, so VIEW, N, ... reach the jobs
gpu_job=$(sbatch --parsable scripts/vae_roundtrip.slurm)
gpu_job="${gpu_job%%;*}"   # federated clusters print "id;cluster"
analyze_job=$(sbatch --parsable --dependency=afterany:"${gpu_job}" scripts/vae_roundtrip_analyze.slurm)
analyze_job="${analyze_job%%;*}"

echo "round-trip array: ${gpu_job}   (logs/vae_roundtrip_${gpu_job}_<1-4>.out)"
echo "analysis:         ${analyze_job}   (logs/vae_rt_analyze_${analyze_job}.out), starts when the array ends"
echo "results:          ${RESULTS:-/standard/uva-mira-drive/3d-fft_results/vae_roundtrip}/${VIEW:-resize}/roundtrip_summary.txt"
echo "watch with:       squeue -u ${USER}"
