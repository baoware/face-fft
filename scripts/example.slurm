#!/bin/bash
#SBATCH -A <allocation-name>
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -t 72:00:00
#SBATCH -J example
#SBATCH -o slurm_files/example.out
#SBATCH -e slurm_files/example.err
#SBATCH --mem=256000

module purge
module load apptainer

APPTAINER="/path/to/apptainer"
CODEDIR="/path/to/your/code"
IMAGE="/path/to/your/face_fft.sif"

${APPTAINER} exec --nv --bind ${CODEDIR}:/mnt/code "${IMAGE}" \
    python3 -m src.run
