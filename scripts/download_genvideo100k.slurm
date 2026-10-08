#!/bin/bash
#SBATCH --job-name=genvideo100k_dl
#SBATCH --account=uva-mira
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=/sfs/weka/scratch/rjr6zk/face-fft/logs/genvideo100k_dl_%j.out

# GenVideo-100K (ModelScope cccnju/GenVideo-100K, CC-BY-NC-SA-4.0, ~231 GB):
# 12 generator archives + a 93.5 GB real-video archive split into 3 parts.
# Resumable (curl -C -) and verified against the SHA-256 the ModelScope API lists
# for every file. Extraction is a separate step: the *_part_* files still need
# their container format confirmed once downloaded.
set -euo pipefail
REPO=/sfs/weka/scratch/rjr6zk/face-fft
DEST="${REPO}/src/face_fft/data/genvideo100k/raw"
API="https://modelscope.cn/api/v1/datasets/cccnju/GenVideo-100K/repo"
mkdir -p "${DEST}"
cd "${DEST}"

curl -s --retry 5 "${API}/tree?Revision=master&Root=/" | python3 -c '
import json, sys
for f in json.load(sys.stdin)["Data"]["Files"]:
    if f["Type"] == "blob":
        print(f["Path"], f["Size"], f["Sha256"])' > manifest.txt
echo "Job started on $(hostname) at $(date); $(wc -l < manifest.txt) files"

fetch() {   # path size sha256
    local p="$1" size="$2" sha="$3"
    if [ -f ".${p}.verified" ]; then echo "  ok (cached) ${p}"; return 0; fi
    for attempt in 1 2 3 4 5; do
        curl -sSL -C - --retry 5 --retry-delay 30 -o "${p}" "${API}?Revision=master&FilePath=${p}" || true
        if [ "$(stat -c %s "${p}" 2>/dev/null || echo 0)" = "${size}" ]; then
            if [ "$(sha256sum "${p}" | cut -d' ' -f1)" = "${sha}" ]; then
                touch ".${p}.verified"; echo "  verified ${p} ($(date +%T))"; return 0
            fi
            echo "  checksum mismatch on ${p}, restarting"; rm -f "${p}"
        fi
        echo "  retry ${attempt} for ${p}"; sleep 60
    done
    echo "  FAILED ${p}"; return 1
}
export -f fetch; export API
# 4 files at a time; the largest are 32 GB
xargs -P 4 -L 1 bash -c 'fetch "$0" "$1" "$2"' < manifest.txt

echo "----------"; du -sh "${DEST}"
echo "verified: $(ls -a | grep -c '\.verified$') of $(wc -l < manifest.txt)"
echo "Job finished at $(date)"
