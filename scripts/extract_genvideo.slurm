#!/bin/bash
#SBATCH --job-name=genvideo_extract
#SBATCH --account=uva-mira
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=/sfs/weka/scratch/rjr6zk/face-fft/logs/genvideo_extract_%j.out

# GenVideo-100K is the TRAINING side only (every archive unpacks to train_<gen>/ or
# Real/). The official unseen-generator TEST set is GenVideo-Val.zip, which lives in
# the full cccnju/Gen-Video repo (2.08 TB); only that one 13.9 GB file is fetched.
# *_part_* files are one tar.gz split into pieces: cat | tar xz.
set -euo pipefail
ROOT=/sfs/weka/scratch/rjr6zk/face-fft/src/face_fft/data/genvideo100k
RAW="${ROOT}/raw"; OUT="${ROOT}/extracted"
mkdir -p "${OUT}"
echo "Job started on $(hostname) at $(date)"

# 1. GenVideo-Val (test), SHA-256 verified
API="https://modelscope.cn/api/v1/datasets/cccnju/Gen-Video/repo"
SHA=$(curl -s "${API}/tree?Revision=master&Root=/" | python3 -c '
import json, sys
print(next(f["Sha256"] for f in json.load(sys.stdin)["Data"]["Files"] if f["Path"] == "GenVideo-Val.zip"))')
if [ ! -f "${RAW}/.GenVideo-Val.zip.verified" ]; then
    for i in 1 2 3 4 5; do
        curl -sSL -C - --retry 5 --retry-delay 30 -o "${RAW}/GenVideo-Val.zip" "${API}?Revision=master&FilePath=GenVideo-Val.zip" || true
        [ "$(sha256sum "${RAW}/GenVideo-Val.zip" | cut -d' ' -f1)" = "${SHA}" ] && { touch "${RAW}/.GenVideo-Val.zip.verified"; break; }
        echo "  GenVideo-Val attempt ${i} failed checksum"; sleep 60
    done
fi
[ -f "${RAW}/.GenVideo-Val.zip.verified" ] || { echo "FATAL: GenVideo-Val.zip not verified"; exit 1; }
echo "[1/3] GenVideo-Val.zip verified ($(date +%T))"

# 2. GenVideo-100K archives (training side)
cd "${RAW}"
for a in *.tar.gz; do
    m="${OUT}/.${a}.done"; [ -f "$m" ] && continue
    echo "  ${a} ($(date +%T))"; tar xzf "$a" -C "${OUT}"; touch "$m"
done
for stem in Real I2VGEN_XL; do
    m="${OUT}/.${stem}_parts.done"; [ -f "$m" ] && continue
    echo "  ${stem}_part_* ($(date +%T))"; cat ${stem}_part_* | tar xz -C "${OUT}"; touch "$m"
done
echo "[2/3] GenVideo-100K extracted ($(date +%T))"

# 3. GenVideo-Val
m="${OUT}/.GenVideo-Val.done"
if [ ! -f "$m" ]; then
    mkdir -p "${OUT}/GenVideo-Val"; 7z x -y -bso0 -bsp0 -o"${OUT}/GenVideo-Val" "${RAW}/GenVideo-Val.zip"; touch "$m"
fi
echo "[3/3] GenVideo-Val extracted ($(date +%T))"

echo "----------"
find "${OUT}" -name '*.mp4' | awk -F/ '{print $(NF-1)}' | sort | uniq -c
du -sh "${OUT}"
echo "Job finished at $(date)"
