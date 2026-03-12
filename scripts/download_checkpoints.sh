#!/usr/bin/env bash
# scripts/download_checkpoints.sh
#
# Pre-downloads HuggingFace model checkpoints to /scratch/<user>/checkpoints/
# before submitting generate_dataset.slurm batch jobs.
#
# USAGE
#   Run interactively on a login node or inside an ijob session:
#     bash scripts/download_checkpoints.sh
#
#   Override the destination directory:
#     CHECKPOINT_DIR=/path/to/custom bash scripts/download_checkpoints.sh
#
# NOTE
#   /scratch on Rivanna purges files after 90 days of inactivity.
#   Re-run this script if your checkpoints have been purged.

set -euo pipefail

CHECKPOINT_DIR="${CHECKPOINT_DIR:-/scratch/${USER}/checkpoints}"
MODEL_COGVIDEOX="THUDM/CogVideoX-2b"
MODEL_WAN="Wan-AI/Wan2.2-I2V-A14B"

echo ""
echo "==> face-fft checkpoint pre-downloader"
echo "    Destination : ${CHECKPOINT_DIR}"
echo "    Models      : ${MODEL_COGVIDEOX}"
echo "                  ${MODEL_WAN}"
echo ""

export HF_HOME="${CHECKPOINT_DIR}"
mkdir -p "${CHECKPOINT_DIR}"

if ! command -v hf &>/dev/null; then
    echo "ERROR: hf not found. Activate your Python environment first."
    exit 1
fi

echo "==> Checking HuggingFace login status..."
hf auth whoami || echo "(Not logged in — ok for public models)"
echo ""

echo "==> Downloading ${MODEL_COGVIDEOX} (~15 GB)..."
hf download "${MODEL_COGVIDEOX}" --repo-type model --quiet
echo "    Done."
echo ""

echo "==> Downloading ${MODEL_WAN} (~40 GB — this will take a while)..."
hf download "${MODEL_WAN}" --repo-type model --quiet
echo "    Done."
echo ""

echo "==> Download complete. Checkpoint directory contents:"
du -sh "${CHECKPOINT_DIR}"/hub/models--* 2>/dev/null || echo "(no models found — check errors above)"
echo ""
echo "==> Submit the generation job with:"
echo "    sbatch scripts/generate_dataset.slurm"
echo ""
