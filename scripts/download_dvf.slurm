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
# Only files not already on disk are fetched: re-requesting finished files spends
# Google Drive's per-file download quota, which made an earlier retry fail on them.
# gdown runs in a throwaway uv environment so the project's dependencies stay untouched.
set -euo pipefail
DEST=/sfs/weka/scratch/rjr6zk/face-fft/src/face_fft/data/dvf/raw
mkdir -p "${DEST}"
cd "${DEST}"
echo "Job started on $(hostname) at $(date)"
/home/rjr6zk/.local/bin/uv tool run --from gdown python - <<'PY'
import os, gdown
url = "https://drive.google.com/drive/folders/1NxCvJVPSxV2Mib5NaNj5Cf2WnnjrqpMb"
listing = gdown.download_folder(url, skip_download=True, quiet=True)
missing = [f for f in listing if not os.path.exists(f.path)]
print(f"{len(listing)} files in folder, {len(missing)} missing: {[f.path for f in missing]}")
failed = []
for f in missing:
    os.makedirs(os.path.dirname(f.path) or ".", exist_ok=True)
    if gdown.download(id=f.id, output=f.path, quiet=False, resume=True) is None:
        failed.append(f.path)
if failed:
    raise SystemExit(f"still missing: {failed}")
PY
echo "----------"; find . -type f -printf "%12s  %p\n" | sort -k2; du -sh .
echo "Job finished at $(date)"
