#!/bin/bash
#SBATCH --job-name=face_fft_train_deepaction
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1                              
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8                
#SBATCH --mem=64G                
#SBATCH --time=36:00:00            
#SBATCH --output=logs/train_%j.out            

# create a logs directory if it doesn't exist to store the output files
mkdir -p logs

echo "Job started on $(hostname) at $(date)"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"
echo "Training..."
echo "----------"

# load required HPC modules
cd /sfs/weka/scratch/rjr6zk/face-fft/
source .venv/bin/activate
export PYTHONPATH="src:$PYTHONPATH"

# run training script
python src/face_fft/training/train_deepaction.py

echo "Job finished at $(date)"