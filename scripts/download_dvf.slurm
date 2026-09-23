#!/bin/bash
#SBATCH --job-name=dvf_dl
#SBATCH --account=uva-mira
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=/sfs/weka/scratch/rjr6zk/face-fft/logs/dvf_dl_%j.out

# DVF (MM-Det, NeurIPS 2024) ORIGINAL videos from the authors' Google Drive folder.
# The Hugging Face copy (sparklexfantasy/DVF, 363 GB) is their preprocessed
# reconstruction data, not the source videos, so it is not used here.
# gdown runs in a throwaway uv environment so the project's dependencies stay
# untouched. Google Drive can rate-limit large public files ("too many users");
# the download resumes, so a failed job can simply be resubmitted.
set -euo pipefail
DEST=/sfs/weka/scratch/rjr6zk/face-fft/src/face_fft/data/dvf/raw
mkdir -p "${DEST}"
cd "${DEST}"
echo "Job started on $(hostname) at $(date)"
/home/rjr6zk/.local/bin/uv tool run --from gdown python - <<'PY'
import gdown
url = "https://drive.google.com/drive/folders/1NxCvJVPSxV2Mib5NaNj5Cf2WnnjrqpMb"
files = gdown.download_folder(url, output=".", quiet=False, resume=True, retries=5)
print(f"downloaded {len(files) if files else 0} files")
PY
echo "----------"; find . -type f -printf "%10s  %p\n" | sort -k2; du -sh .
echo "Job finished at $(date)"
