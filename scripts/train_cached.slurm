#!/bin/bash
#SBATCH --job-name=cached_train
#SBATCH --account=uva-mira
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/sfs/weka/scratch/rjr6zk/face-fft/logs/cached_%x_%j.out

# Train one configuration on the clip cache, then evaluate it on every protocol.
#   sbatch --job-name=b0_T16_s42 --export=ALL,RUN_NAME=b0_T16_s42,T=16,SEED=42 scripts/train_cached.slurm
# Variables (defaults = baseline B0): TRAIN_PAIRS MODE ARCH T SEED REAL_STRIDES EXCLUDE EPOCHS TESTS
set -euo pipefail
REPO=/sfs/weka/scratch/rjr6zk/face-fft
cd "${REPO}"
source .venv/bin/activate

: "${RUN_NAME:?set RUN_NAME}"
TRAIN_PAIRS="${TRAIN_PAIRS:-Pair1}"; MODE="${MODE:-fft_no_mask}"; ARCH="${ARCH:-compact}"
T="${T:-16}"; SEED="${SEED:-42}"; REAL_STRIDES="${REAL_STRIDES:-1,2,3,4}"; EXCLUDE="${EXCLUDE:-}"
EPOCHS="${EPOCHS:-15}"; TESTS="${TESTS:-Pair1,Pair2,DeepAction,6.7m}"
OUT_ROOT=/standard/uva-mira-drive/3d-fft_checkpoints/cached

echo "Job started on $(hostname) at $(date)"
echo "run=${RUN_NAME} pairs=${TRAIN_PAIRS} mode=${MODE} arch=${ARCH} T=${T} seed=${SEED} strides=${REAL_STRIDES} exclude=${EXCLUDE:-none}"
PYTHONPATH=src python -m face_fft.training.train_cached \
    --cache "${REPO}/cache/v1" --train_pairs "${TRAIN_PAIRS}" --exclude_sources "${EXCLUDE}" \
    --mode "${MODE}" --arch "${ARCH}" --T "${T}" --seed "${SEED}" --real_strides "${REAL_STRIDES}" \
    --epochs "${EPOCHS}" --out_root "${OUT_ROOT}" --run_name "${RUN_NAME}"
PYTHONPATH=src python -m face_fft.eval.eval_cached \
    --cache "${REPO}/cache/v1" --run_dir "${OUT_ROOT}/${RUN_NAME}" --tests "${TESTS}"
echo "Job finished at $(date)"
