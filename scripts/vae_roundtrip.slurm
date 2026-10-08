#!/bin/bash
#SBATCH --job-name=vae_roundtrip
#SBATCH --account=uva-mira
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=1-4
#SBATCH --output=logs/vae_roundtrip_%A_%a.out

# VAE round-trip, GPU stage: one array task per VAE. No training and no AI step --
# each clip goes through the encoder and decoder once, so any GPU type will do (no
# a100 request: it only lengthens the queue). Memory use has not been measured: a clip that
# runs out of GPU memory is skipped and logged, and re-submitting keeps finished clips.
#   1 = cogvideox (4 frames -> 1), 2 = wan (4 -> 1), 3 = ltx (8 -> 1), 4 = sd (frame-wise control)
# Easiest: bash scripts/vae_roundtrip.sh (submits this and queues the analysis behind it).
# By hand, from the repo root (logs/ must exist):
#   mkdir -p logs && sbatch scripts/vae_roundtrip.slurm
#   sbatch --export=ALL,VIEW=crop scripts/vae_roundtrip.slurm     # native-resolution crop instead
# CogVideoX chunking ablation (task 1 only; writes cogvideox-dec<N>/ beside the default run):
#   sbatch --array=1 --export=ALL,DECODE_LATENTS=4 scripts/vae_roundtrip.slurm     # seams every 16 frames
#   sbatch --array=1 --export=ALL,DECODE_LATENTS=13 scripts/vae_roundtrip.slurm    # one chunk, no seams
# Variables: VIEW (resize) N (200) FRAMES (49) OUT (<repo>/roundtrip_frames) REPO HF_HOME
#            DA_ROOT (DeepAction; must contain Pexels/<id>/*.mp4. Default: this checkout's copy,
#                     else SHARED_DA_ROOT below)
#            DECODE_LATENTS (unset = pipeline default, 2 = seams every 8 frames; valid: 2 4 6 12 13)
set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR}}"
cd "${REPO}"
source .venv/bin/activate
# CogVideoX and Wan are already in the lossless test's cache; point HF_HOME there to reuse them
export HF_HOME="${HF_HOME:-/scratch/${USER}/hf-cache}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
VIEW="${VIEW:-resize}"; N="${N:-200}"; FRAMES="${FRAMES:-49}"; OUT="${OUT:-${REPO}/roundtrip_frames}"
SHARED_DA_ROOT=/sfs/weka/scratch/gcs3ja/face-fft/src/face_fft/data/deepaction_dataset   # Pexels/ only, from faridlab/deepaction_v1
if [ -z "${DA_ROOT:-}" ]; then
    DA_ROOT="${REPO}/src/face_fft/data/deepaction_dataset"
    if [ ! -d "${DA_ROOT}/Pexels" ] && [ -r "${SHARED_DA_ROOT}/Pexels" ]; then
        DA_ROOT="${SHARED_DA_ROOT}"
    fi
fi
VAE=$(echo cogvideox wan ltx sd | cut -d' ' -f"${SLURM_ARRAY_TASK_ID}")
if [ ! -d "${DA_ROOT}/Pexels" ]; then
    echo "no readable ${DA_ROOT}/Pexels -- set DA_ROOT to a copy of DeepAction, or download one:" >&2
    echo "  hf download faridlab/deepaction_v1 --repo-type dataset --include 'Pexels/*' --local-dir <dir>" >&2
    exit 1
fi
CHUNK=()
if [ "${VAE}" = cogvideox ] && [ -n "${DECODE_LATENTS:-}" ]; then
    CHUNK=(--decode_latents "${DECODE_LATENTS}")
fi
echo "task ${SLURM_ARRAY_TASK_ID}: ${VAE} view=${VIEW} n=${N} frames=${FRAMES} ${CHUNK[*]:-} on $(hostname) at $(date)"
echo "data=${DA_ROOT} out=${OUT} hf=${HF_HOME}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
PYTHONPATH=src python bin/vae_roundtrip.py roundtrip --vae "${VAE}" --view "${VIEW}" --n "${N}" \
    --frames "${FRAMES}" --out "${OUT}" --da_root "${DA_ROOT}" ${CHUNK[@]+"${CHUNK[@]}"}
echo "done at $(date)"
