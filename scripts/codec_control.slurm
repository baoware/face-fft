#!/bin/bash
#SBATCH --job-name=face_fft_codec_control
#SBATCH --account=uva-mira
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/sfs/weka/scratch/rjr6zk/face-fft/logs/codec_control_%j.out

# Codec control: does the period-4 temporal peak survive when no codec is involved?
# Holds the generated frames fixed and varies only the encoding, so H.264's 4-frame
# cadence is separated from the generator's latent temporal stride.

set -euo pipefail

REPO=/sfs/weka/scratch/rjr6zk/face-fft
SOURCE_DIR="${REPO}/src/face_fft/data/deepaction_dataset/Pexels"
OUT_DIR=/standard/uva-mira-drive/3d-fft_results/codec_control
HF_CACHE=/sfs/weka/scratch/rjr6zk/hf-cache
MODEL=zai-org/CogVideoX-5b-I2V
UV=/home/rjr6zk/.local/bin/uv
PYBIN=/sfs/weka/scratch/rjr6zk/uv-python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12

export UV_LINK_MODE=copy
export HF_HOME="${HF_CACHE}"

cd "${REPO}"
mkdir -p "${OUT_DIR}"

echo "Job started on $(hostname) at $(date)"
echo "Allocated GPUs: ${CUDA_VISIBLE_DEVICES:-none}"
echo "----------"

echo "[preflight] checking huggingface.co reachability..."
if ! curl -sf -o /dev/null --max-time 60 https://huggingface.co; then
    echo "FATAL: no outbound HTTPS from this compute node." >&2
    exit 1
fi
echo "[preflight] ok"

# --- step 1: venv ------------------------------------------------------------
echo "[1/3] checking venv ($(date))"
if ! ./.venv/bin/python -c "import torch, cv2, diffusers" 2>/dev/null; then
    echo "[1/3] venv unusable, rebuilding from uv.lock"
    "${UV}" venv --clear --python "${PYBIN}" .venv
    "${UV}" sync --frozen
fi
source .venv/bin/activate
python -c "import torch, diffusers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('diffusers', diffusers.__version__)"

# --- step 2: generator weights (~15 GB, purged from scratch in July) ----------
echo "[2/3] ensuring ${MODEL} is cached ($(date))"
python - <<PY
from huggingface_hub import snapshot_download
p = snapshot_download(repo_id="${MODEL}", repo_type="model", max_workers=8)
print("model at", p)
PY

# --- step 3: the control -----------------------------------------------------
echo "[3/3] running codec control ($(date))"
PYTHONPATH=src python bin/codec_control.py \
    --source_dir "${SOURCE_DIR}" \
    --out_dir "${OUT_DIR}" \
    --num_videos 50 \
    --num_frames 48 \
    --period 4 \
    --crf 23 \
    --seed 42 \
    --cache_dir "${HF_CACHE}"

echo "----------"
echo "results: ${OUT_DIR}/results.json"
echo "Job finished at $(date)"
