#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=LESSWarmup
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=02:00:00
#SBATCH --output=install_1_%A.out

module purge
module load 2023

# module load CUDA/12.1.1
# pip3 install torch==2.1.2 torchvision torchaudio 
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
# Module load PyTorch/1.12.0-foss-2022a-CUDA-11.7.0 
