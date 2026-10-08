#!/bin/bash
#SBATCH --job-name=dvf_extract
#SBATCH --account=uva-mira
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=/sfs/weka/scratch/rjr6zk/face-fft/logs/dvf_extract_%j.out

# DVF: <generator>/{0_real,1_fake}.zip (some multi-volume .z01...), each unpacking to
# 0_real/ or 1_fake/. Extract into extracted/<generator>/. 7z reads split zips directly.
set -euo pipefail
RAW=/sfs/weka/scratch/rjr6zk/face-fft/src/face_fft/data/dvf/raw
OUT=/sfs/weka/scratch/rjr6zk/face-fft/src/face_fft/data/dvf/extracted
mkdir -p "${OUT}"
echo "Job started on $(hostname) at $(date)"
for z in "${RAW}"/*/*.zip; do
    gen=$(basename "$(dirname "$z")"); m="${OUT}/.${gen}_$(basename "$z").done"
    [ -f "$m" ] && continue
    echo "  ${gen}/$(basename "$z") ($(date +%T))"
    mkdir -p "${OUT}/${gen}"; 7z x -y -bso0 -bsp0 -o"${OUT}/${gen}" "$z"; touch "$m"
done
echo "----------"
find "${OUT}" -type f | awk -F/ '{n=split($NF,e,"."); print $(NF-2)"/"$(NF-1), e[n]}' | sort | uniq -c
du -sh "${OUT}"
echo "Job finished at $(date)"
