#!/usr/bin/env bash
# =============================================================================
# Face-FFT: Full End-to-End Pipeline
# From raw real videos → paired synthetic dataset → training → evaluation
# =============================================================================
#
# PREREQUISITES — READ BEFORE RUNNING
# =============================================================================
#
# 1. KAGGLE DATASET
#    Download the "Selfies and Video" dataset from Kaggle:
#      https://www.kaggle.com/datasets/tapakah68/selfies-and-videos
#
#    Place the downloaded .mp4 video files (real face videos) into:
#      data/raw_videos/
#    relative to the repository root. The script expects that directory to exist
#    and contain at least one .mp4 file before it starts.
#
#    Via Kaggle CLI (if configured):
#      mkdir -p data/raw_videos
#      kaggle datasets download -d tapakah68/selfies-and-videos --unzip -p data/raw_videos
#
#    Or download manually from the Kaggle page and unzip into data/raw_videos/.
#
# 2. HARDWARE
#    Data generation requires a CUDA GPU with sufficient VRAM:
#      - CogVideoX-2b : ~24 GB VRAM
#      - Wan 2.2-I2V  : ~20 GB VRAM at 480p generation resolution
#    Training and evaluation fall back to CPU if no GPU is available,
#    but GPU is strongly recommended.
#
# 3. SOFTWARE
#    Install all Python dependencies before running:
#      uv sync --dev
#    Run tests to confirm the environment is correct:
#      uv run pytest
#
# 4. HUGGINGFACE ACCESS
#    Generation models are fetched from HuggingFace Hub on first use and
#    cached locally. Ensure you have internet access and enough disk space
#    (~15 GB for CogVideoX-2b, ~40 GB for Wan 2.2-I2V-A14B).
#    Log in if models are gated:
#      huggingface-cli login
#
# =============================================================================

set -euo pipefail

# =============================================================================
# CONFIGURATION — edit these variables to match your environment
# =============================================================================

# Root of the repository (script resolves it automatically)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Directory containing downloaded real .mp4 videos (Kaggle dataset)
RAW_VIDEO_DIR="${REPO_ROOT}/data/raw_videos"

# Processed tensor output directories
REAL_PT_DIR="${REPO_ROOT}/data/real"
SYNTH_COGVIDEOX_DIR="${REPO_ROOT}/data/synth_cogvideox"
SYNTH_WAN_DIR="${REPO_ROOT}/data/synth_wan"

# Checkpoint output directory
CHECKPOINT_DIR="${REPO_ROOT}/checkpoints"

# Model checkpoint filenames
CKPT_COGVIDEOX="${CHECKPOINT_DIR}/best_model_cogvideox.pt"
CKPT_WAN="${CHECKPOINT_DIR}/best_model_wan.pt"

# HuggingFace model IDs
MODEL_COGVIDEOX="THUDM/CogVideoX-2b"
MODEL_WAN="Wan-AI/Wan2.2-I2V-A14B"

# Generation settings
NUM_FRAMES=16
# Generation resolution — intentionally below native model resolution.
# Both generators downsample to 256x256 after generation, so there is no
# benefit to generating at 720p. 480x480 cuts VRAM usage and generation
# time significantly on limited hardware.
GEN_HEIGHT=480
GEN_WIDTH=480
# Denoising steps — 20 is a good compute/quality tradeoff for this task.
NUM_INFERENCE_STEPS=20
GEN_PROMPT="A highly realistic person's face, talking slightly, photorealistic video."

# Training hyperparameters
EPOCHS=20
BATCH_SIZE=16
LR=1e-3

# Python runner. Override via environment variable for container use:
#   PY=python3 ./run_pipeline.sh
PY="${PY:-uv run python}"

# =============================================================================
# HELPERS
# =============================================================================

log() { echo ""; echo "==> $*"; echo ""; }

# =============================================================================
# PREFLIGHT CHECKS
# =============================================================================

log "Checking prerequisites..."

if [ ! -d "${RAW_VIDEO_DIR}" ]; then
    echo "ERROR: ${RAW_VIDEO_DIR} does not exist."
    echo "       Download the Kaggle dataset and place .mp4 files there."
    echo "       See the PREREQUISITES section at the top of this script."
    exit 1
fi

NUM_VIDEOS=$(find "${RAW_VIDEO_DIR}" \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" \) | wc -l)
if [ "${NUM_VIDEOS}" -eq 0 ]; then
    echo "ERROR: No .mp4/.avi/.mov files found in ${RAW_VIDEO_DIR}."
    exit 1
fi
echo "Found ${NUM_VIDEOS} video(s) in ${RAW_VIDEO_DIR}."

# =============================================================================
# DIRECTORY SETUP
# =============================================================================

log "Creating output directories..."
mkdir -p "${REAL_PT_DIR}"
mkdir -p "${SYNTH_COGVIDEOX_DIR}"
mkdir -p "${SYNTH_WAN_DIR}"
mkdir -p "${CHECKPOINT_DIR}"

# =============================================================================
# STEP 1: GENERATE PAIRED DATASET — CogVideoX
# =============================================================================
# Real videos are preprocessed alongside generation and saved as .pt tensors.
# Both the real and synthetic tensors are stored at shape (C=3, T=16, H=256, W=256).

log "[1/5] Generating paired dataset with CogVideoX (${MODEL_COGVIDEOX})..."
${PY} "${REPO_ROOT}/scripts/generate_dataset.py" \
    --source_dir  "${RAW_VIDEO_DIR}" \
    --real_out_dir  "${REAL_PT_DIR}" \
    --synth_out_dir "${SYNTH_COGVIDEOX_DIR}" \
    --generator cogvideox \
    --model_id  "${MODEL_COGVIDEOX}" \
    --prompt    "${GEN_PROMPT}" \
    --num_frames "${NUM_FRAMES}" \
    --gen_height "${GEN_HEIGHT}" \
    --gen_width  "${GEN_WIDTH}" \
    --num_inference_steps "${NUM_INFERENCE_STEPS}"

# =============================================================================
# STEP 2: GENERATE PAIRED DATASET — Wan
# =============================================================================
# The real tensors directory is reused; only the synthetic side differs.
# This is safe because generate_dataset.py skips files that already exist.

log "[2/5] Generating paired dataset with Wan (${MODEL_WAN})..."
${PY} "${REPO_ROOT}/scripts/generate_dataset.py" \
    --source_dir  "${RAW_VIDEO_DIR}" \
    --real_out_dir  "${REAL_PT_DIR}" \
    --synth_out_dir "${SYNTH_WAN_DIR}" \
    --generator wan \
    --model_id  "${MODEL_WAN}" \
    --prompt    "${GEN_PROMPT}" \
    --num_frames "${NUM_FRAMES}" \
    --gen_height "${GEN_HEIGHT}" \
    --gen_width  "${GEN_WIDTH}" \
    --num_inference_steps "${NUM_INFERENCE_STEPS}"

# =============================================================================
# STEP 3: TRAIN — on CogVideoX synthetic data
# =============================================================================
# The full pipeline (SpatiotemporalFFT → CompactSpectralCNN) is trained end-to-end.
# Best checkpoint is selected by validation loss.

log "[3/5] Training on CogVideoX data..."
${PY} "${REPO_ROOT}/scripts/train.py" \
    --real_dir  "${REAL_PT_DIR}" \
    --synth_dir "${SYNTH_COGVIDEOX_DIR}" \
    --epochs     "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --lr         "${LR}" \
    --save_path  "${CKPT_COGVIDEOX}"

# =============================================================================
# STEP 4: EVALUATE — in-distribution (CogVideoX → CogVideoX)
# =============================================================================

log "[4/5] Evaluating in-distribution (CogVideoX test split)..."
${PY} "${REPO_ROOT}/scripts/evaluate.py" \
    --real_dir   "${REAL_PT_DIR}" \
    --synth_dir  "${SYNTH_COGVIDEOX_DIR}" \
    --model_path "${CKPT_COGVIDEOX}" \
    --batch_size "${BATCH_SIZE}"

# =============================================================================
# STEP 5: EVALUATE — cross-generator generalization (CogVideoX model → Wan data)
# =============================================================================
# This is the core research question: does the spectral detector generalize
# to a generator it has never seen during training?

log "[5/5] Evaluating cross-generator generalization (CogVideoX model → Wan data)..."
${PY} "${REPO_ROOT}/scripts/evaluate.py" \
    --real_dir   "${REAL_PT_DIR}" \
    --synth_dir  "${SYNTH_WAN_DIR}" \
    --model_path "${CKPT_COGVIDEOX}" \
    --batch_size "${BATCH_SIZE}"

# =============================================================================

log "Pipeline complete."
echo "Checkpoints saved to: ${CHECKPOINT_DIR}"
