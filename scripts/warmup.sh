#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=LESSWarmup
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=02:00:00
#SBATCH --output=warmup_training_%A.out

module purge
module load 2024

module load CUDA/12.1.1
# pip3 install torch==2.1.2 torchvision torchaudio 
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
# Module load PyTorch/1.12.0-foss-2022a-CUDA-11.7.0 

pip install torchvision torchaudio

pip install peft==0.7.1
pip install transformers==4.36.2
pip install traker[fast]==0.1.3

cd ..
pip install -r requirement.txt
pip install -e .


# cd $HOME/path/to/LESS
# source activate dl2023

# # Set variables for warmup training
# DATA_DIR="../data"
# MODEL_PATH="meta-llama/Llama-2-7b-hf"
# PERCENTAGE=0.05
# DATA_SEED=3
# JOB_NAME="llama2-7b-p${PERCENTAGE}-lora-seed${DATA_SEED}"

# # Run warmup training
# bash /less/scripts/train/warmup_lora_train.sh "$DATA_DIR" "$MODEL_PATH" "$PERCENTAGE" "$DATA_SEED" "$JOB_NAME"
