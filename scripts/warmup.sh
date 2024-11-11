#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=LESSWarmup
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=02:00:00
#SBATCH --output=warmup_training_%A.out

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
