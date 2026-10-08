#!/bin/bash
#SBATCH --job-name=face_fft_rebuild_ablation
#SBATCH --account=uva-mira
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=/sfs/weka/scratch/rjr6zk/face-fft/logs/rebuild_ablation_%j.out

# Rebuilds everything the July scratch purge removed, then retrains the
# 4-architecture DeepAction ablation:
#   1. venv (purged down to 2 packages)
#   2. DeepAction v1 dataset (purged to 0 files)
#   3. ablation training -> checkpoints/ -> /standard/uva-mira-drive/3d-fft_checkpoints

set -euo pipefail

REPO=/sfs/weka/scratch/rjr6zk/face-fft
DATA_ROOT="${REPO}/src/face_fft/data/deepaction_dataset"
UV=/home/rjr6zk/.local/bin/uv

cd "${REPO}"

echo "Job started on $(hostname) at $(date)"
echo "Allocated GPUs: ${CUDA_VISIBLE_DEVICES:-none}"
echo "----------"

# --- preflight: outbound HTTPS is required for steps 1 and 2 -----------------
echo "[preflight] checking huggingface.co reachability..."
if ! curl -sf -o /dev/null --max-time 60 https://huggingface.co; then
    echo "FATAL: no outbound HTTPS from this compute node; run steps 1-2 on a login node." >&2
    exit 1
fi
echo "[preflight] ok"

# --- step 1: ensure the virtualenv is healthy -------------------------------
# The purge removed .venv/pyvenv.cfg, which stops uv from recognising .venv as
# a venv at all: it then installs into the managed interpreter instead, and
# dies on relative-path arithmetic because the cpython-3.12-* symlink resolves
# through /sfs/gpfs/tardis/home (a different path depth than /home). Recreate
# the venv explicitly against the RESOLVED interpreter to avoid that symlink.
export UV_LINK_MODE=copy
PYBIN=/home/rjr6zk/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12

echo "[1/3] checking venv ($(date))"
if ! ./.venv/bin/python -c "import torch, cv2, sklearn" 2>/dev/null; then
    echo "[1/3] venv unusable, rebuilding from uv.lock"
    "${UV}" venv --clear --python "${PYBIN}" .venv
    "${UV}" sync --frozen
fi
source .venv/bin/activate
python -c "import torch, cv2, sklearn; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# --- step 2: re-download DeepAction v1 (public repo, no token needed) --------
if [ -d "${DATA_ROOT}/Pexels" ]; then
    echo "[2/3] DeepAction already present, skipping download"
else
    echo "[2/3] downloading faridlab/deepaction_v1 ($(date))"
    python - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download(
    repo_id="faridlab/deepaction_v1",
    repo_type="dataset",
    local_dir="/sfs/weka/scratch/rjr6zk/face-fft/src/face_fft/data/deepaction_dataset",
    max_workers=8,
)
print("downloaded to", p)
PY
fi
echo "[2/3] video count: $(find "${DATA_ROOT}" -name '*.mp4' | wc -l)"

# --- step 3: retrain the ablation -------------------------------------------
echo "[3/3] training ablation ($(date))"
PYTHONPATH=src python -m face_fft.training.run_ablation

echo "----------"
echo "checkpoints written:"
ls -la checkpoints/
echo "Job finished at $(date)"
