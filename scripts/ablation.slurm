#!/bin/bash
#SBATCH --job-name=face_fft_ablation
#SBATCH --account=uva-mira
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/sfs/weka/scratch/rjr6zk/face-fft/logs/ablation_%x_%j.out

# Four-architecture ablation for one pipeline arm.
#
#   sbatch --job-name=pixel   --export=ALL,FACE_FFT_MODE=pixel_baseline scripts/ablation.slurm
#   sbatch --job-name=nofilt  --export=ALL,FACE_FFT_MODE=fft_no_mask    scripts/ablation.slurm
#
# FACE_FFT_SEED overrides the base seed (default 42); each architecture uses
# base+index, so a single architecture reruns reproducibly on its own.

set -euo pipefail

REPO=/sfs/weka/scratch/rjr6zk/face-fft
UV=/home/rjr6zk/.local/bin/uv
PYBIN=/sfs/weka/scratch/rjr6zk/uv-python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12
export UV_LINK_MODE=copy

FACE_FFT_MODE="${FACE_FFT_MODE:-pixel_baseline}"
FACE_FFT_SEED="${FACE_FFT_SEED:-42}"
export FACE_FFT_MODE FACE_FFT_SEED

cd "${REPO}"

echo "Job started on $(hostname) at $(date)"
echo "Allocated GPUs: ${CUDA_VISIBLE_DEVICES:-none}"
echo "Mode: ${FACE_FFT_MODE}   Base seed: ${FACE_FFT_SEED}"
echo "----------"

echo "[1/2] checking venv ($(date))"
if ! ./.venv/bin/python -c "import torch, cv2, sklearn" 2>/dev/null; then
    echo "[1/2] venv unusable, rebuilding from uv.lock"
    "${UV}" venv --clear --python "${PYBIN}" .venv
    "${UV}" sync --frozen
fi
source .venv/bin/activate
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

if [ ! -d "src/face_fft/data/deepaction_dataset/Pexels" ]; then
    echo "FATAL: DeepAction dataset missing; run scripts/rebuild_ablation.slurm first." >&2
    exit 1
fi

echo "[2/2] training ($(date))"
PYTHONPATH=src python -m face_fft.training.run_ablation

echo "----------"
echo "checkpoints:"
ls -la checkpoints/ checkpoints/no_filter/ 2>/dev/null || true
echo "Job finished at $(date)"
