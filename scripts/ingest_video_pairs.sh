#!/usr/bin/env bash
# =============================================================================
# Face-FFT: External Video Pair Ingestion Pipeline
# From pre-existing real/synthetic video pairs → paired .pt tensors
# =============================================================================
#
# PREREQUISITES — READ BEFORE RUNNING
# =============================================================================
#
# 1. VIDEO PAIRS
#    Place your real and synthetic videos into matching subdirectories:
#
#      src/face_fft/data/raw_video_pairs/real/       ← real video files
#      src/face_fft/data/raw_video_pairs/synthetic/  ← synthetic video files
#
#    Videos are matched by filename stem, e.g.:
#      real/video001.mp4  ↔  synthetic/video001.mp4
#
#    Supported formats: .mp4, .avi, .mov
#    Unmatched files are skipped with a warning — they do not cause failure.
#
# 2. DATASET NAME
#    Set DATASET_NAME below to a short identifier (no spaces) for this dataset,
#    e.g. "faceforensics" or "dfdc". Output files will be named:
#      {stem}_{DATASET_NAME}.pt
#    This avoids filename collisions with generation-produced tensors
#    ({stem}_{generator}_{index}.pt).
#
# 3. SOFTWARE
#    Install all Python dependencies before running:
#      uv sync --dev
#    Run tests to confirm the environment is correct:
#      uv run pytest
#
# 4. NO GPU REQUIRED
#    Unlike scripts/preprocess.sh, this script only performs video I/O and
#    tensor preprocessing. It runs entirely on CPU.
#
# =============================================================================

set -euo pipefail

# =============================================================================
# CONFIGURATION — edit these variables to match your environment
# =============================================================================

# Root of the repository (resolved automatically)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Short identifier for this external dataset (required — no spaces)
DATASET_NAME="faceforensics"

# Input directory containing real/ and synthetic/ subdirectories
INPUT_DIR="${REPO_ROOT}/src/face_fft/data/raw_video_pairs"

# Output tensor directories
REAL_PT_DIR="${REPO_ROOT}/src/face_fft/data/real"
SYNTH_PT_DIR="${REPO_ROOT}/src/face_fft/data/synth_${DATASET_NAME}"

# Preprocessing settings
NUM_FRAMES=16
TARGET_SIZE=256

# Python runner. Override via environment variable for container use:
#   PY=python3 ./ingest_video_pairs.sh
PY="${PY:-uv run python}"

# =============================================================================
# HELPERS
# =============================================================================

log() { echo ""; echo "==> $*"; echo ""; }

# =============================================================================
# PREFLIGHT CHECKS
# =============================================================================

log "Checking prerequisites..."

if [ ! -d "${INPUT_DIR}/real" ]; then
    echo "ERROR: ${INPUT_DIR}/real does not exist."
    echo "       Create the directory and place real video files there."
    echo "       See the PREREQUISITES section at the top of this script."
    exit 1
fi

if [ ! -d "${INPUT_DIR}/synthetic" ]; then
    echo "ERROR: ${INPUT_DIR}/synthetic does not exist."
    echo "       Create the directory and place synthetic video files there."
    echo "       See the PREREQUISITES section at the top of this script."
    exit 1
fi

NUM_REAL=$(find "${INPUT_DIR}/real" \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" \) | wc -l)
if [ "${NUM_REAL}" -eq 0 ]; then
    echo "ERROR: No .mp4/.avi/.mov files found in ${INPUT_DIR}/real."
    exit 1
fi

NUM_SYNTH=$(find "${INPUT_DIR}/synthetic" \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" \) | wc -l)
if [ "${NUM_SYNTH}" -eq 0 ]; then
    echo "ERROR: No .mp4/.avi/.mov files found in ${INPUT_DIR}/synthetic."
    exit 1
fi

echo "Found ${NUM_REAL} real video(s) and ${NUM_SYNTH} synthetic video(s)."

# =============================================================================
# DIRECTORY SETUP
# =============================================================================

log "Creating output directories..."
mkdir -p "${REAL_PT_DIR}"
mkdir -p "${SYNTH_PT_DIR}"

# =============================================================================
# INGEST VIDEO PAIRS
# =============================================================================

log "Ingesting video pairs (dataset: ${DATASET_NAME})..."
${PY} "${REPO_ROOT}/bin/ingest_video_pairs.py" \
    --input_dir     "${INPUT_DIR}" \
    --real_out_dir  "${REAL_PT_DIR}" \
    --synth_out_dir "${SYNTH_PT_DIR}" \
    --dataset_name  "${DATASET_NAME}" \
    --num_frames    "${NUM_FRAMES}" \
    --target_size   "${TARGET_SIZE}"

# =============================================================================

log "Ingestion complete."
echo "Real tensors:       ${REAL_PT_DIR}"
echo "Synthetic tensors:  ${SYNTH_PT_DIR}"
echo ""
echo "Next: Submit train.slurm (or run locally) using these directories."
