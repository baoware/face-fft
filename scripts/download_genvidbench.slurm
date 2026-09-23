#!/bin/bash
#SBATCH --job-name=gvb_download
#SBATCH --account=uva-mira
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/sfs/weka/scratch/rjr6zk/face-fft/logs/gvb_download_%j.out

# Full GenVidBench (jian-0/GenVidBench, public, ~216 GB compressed): Pair1 + Pair2
# archives, the 6.7m subset, labels and verification lists. Archives are kept
# after extraction so a bad extract can be redone without re-downloading.
#
# Lands on scratch (re-downloadable; subject to the 90-day purge). Derived
# manifests / face subsets should go to /standard.

set -euo pipefail

REPO=/sfs/weka/scratch/rjr6zk/face-fft
GVB="${REPO}/src/face_fft/data/genvidbench_dataset"
RAW="${GVB}/raw"
OUT="${GVB}/extracted"
mkdir -p "${RAW}" "${OUT}"

cd "${REPO}"
source .venv/bin/activate
# System /usr/bin/7z (p7zip) cannot read RAR5, which every Pair1/Pair2 archive
# is; the first run failed on ms.rar. unrar handles RAR5; 7z handles the
# multi-volume 7z and the tar.
module load unrar/7.21
UNRAR=/apps/software/standard/core/unrar/7.21/bin/unrar
SEVENZ=/usr/bin/7z

echo "Job started on $(hostname) at $(date)"
echo "unrar: ${UNRAR}   7z: ${SEVENZ}"

echo "[1/2] downloading ($(date))"
python - <<PY
from huggingface_hub import snapshot_download
p = snapshot_download(
    repo_id="jian-0/GenVidBench", repo_type="dataset",
    local_dir="${RAW}", max_workers=8,
)
print("downloaded to", p)
PY
du -sh "${RAW}"

echo "[2/2] extracting ($(date))"
# Each RAR carries its own top-level folder (ms/, vript/, ...), so extract into
# the Pair directory itself; a per-archive marker makes reruns resumable.
unrar_into() {  # $1 archive, $2 destination dir
    local a="$1" d="$2" m="$2/.$(basename "$1").done"
    if [ -f "$m" ]; then echo "  skip $(basename "$a") (done)"; return; fi
    mkdir -p "$d"
    echo "  $(basename "$a") -> ${d#${GVB}/}  ($(date +%T))"
    "${UNRAR}" x -o+ -idq "$a" "$d/"
    touch "$m"
}
for a in "${RAW}"/GenVidBench/Pair1/*.rar; do unrar_into "$a" "${OUT}/Pair1"; done
for a in "${RAW}"/GenVidBench/Pair2/*.rar; do unrar_into "$a" "${OUT}/Pair2"; done

# hd_vg_130m is a 4-volume 7z wrapped around a single RAR: unwrap, then unrar.
INNER="${RAW}/GenVidBench/Pair2/_hd_vg_inner"
if [ ! -f "${OUT}/Pair2/.hd_vg_130m.rar.done" ]; then
    if [ ! -f "${INNER}/.unwrapped" ]; then
        mkdir -p "${INNER}"
        echo "  unwrapping hd_vg_130m.7z.* ($(date +%T))"
        "${SEVENZ}" x -y -bso0 -bsp0 -o"${INNER}" "${RAW}/GenVidBench/Pair2/hd_vg_130m.7z.001"
        touch "${INNER}/.unwrapped"
    fi
    unrar_into "$(ls "${INNER}"/*.rar | head -1)" "${OUT}/Pair2"
fi

# Replacement files for cogvideo/mora/svd/ms (mirrors Pair1/ Pair2/). Kept
# separate, NOT merged over the originals, until we know what it supersedes.
FIX="${OUT}/_fix_cogvideo_mora_svd_ms"
if [ ! -f "${FIX}/.done" ]; then
    mkdir -p "${FIX}"
    echo "  cogvideo_mora_svd_ms_fix.tar -> ${FIX#${GVB}/}  ($(date +%T))"
    tar xf "${RAW}/GenVidBench/cogvideo_mora_svd_ms_fix.tar" -C "${FIX}"
    touch "${FIX}/.done"
fi

echo "----------"
echo "mp4 counts by source:"
for d in "${OUT}"/Pair1/*/ "${OUT}"/Pair2/*/ "${OUT}"/_fix_*/Pair*/*/; do
    printf "  %-40s %7d\n" "${d#${OUT}/}" "$(find "$d" -name '*.mp4' | wc -l)"
done
du -sh "${OUT}"
echo "Job finished at $(date)"
