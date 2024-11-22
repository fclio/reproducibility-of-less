#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=LESSWarmup
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=02:00:00
#SBATCH --output=install_2_%A.out

module purge
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

pip install torchvision torchaudio

pip install peft==0.7.1
pip install transformers==4.36.2
pip install traker[fast]==0.1.3

cd ..

pip install -r requirement.txt
pip install -e .