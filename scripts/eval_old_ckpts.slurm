#!/bin/bash
#SBATCH --job-name=eval_old_ckpts
#SBATCH --account=uva-mira
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/sfs/weka/scratch/rjr6zk/face-fft/logs/eval_old_ckpts_%j.out

# "Before" row: the 8 DeepAction-trained ablation checkpoints on the normalised
# GenVidBench test sets. Caveat: they were trained on raw DeepAction files with 8
# frames spread across each clip and no square crop, so this also measures how
# they cope with a preprocessing change. Their DeepAction row is not a clean
# held-out test: their own split (seed-42 shuffle) differs from the manifest's.
set -euo pipefail
REPO=/sfs/weka/scratch/rjr6zk/face-fft
cd "${REPO}"
source .venv/bin/activate
OUT=/standard/uva-mira-drive/3d-fft_results/old_ckpt_eval
for arm in pixel:checkpoints:pixel_baseline nofilter:checkpoints/no_filter:fft_no_mask; do
    IFS=: read -r name dir mode <<< "${arm}"
    for arch in compact r3d_18 mc3_18 r2plus1d_18; do
        echo "=== ${name} / ${arch} ==="
        PYTHONPATH=src python -m face_fft.eval.eval_cached --cache "${REPO}/cache/v1" \
            --ckpt "${dir}/ablation_${arch}.pt" --mode "${mode}" --arch "${arch}" --T 8 \
            --tests Pair1,Pair2,DeepAction,6.7m --out "${OUT}/${name}_${arch}.json"
    done
done
