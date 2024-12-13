#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=training
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=05:00:00
#SBATCH --output=slurm_output/training_scifact_%A.out

module purge
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1


DATA_SEED=3
PERCENTAGE=0.05
TARGET_TASK_NAME="scifact"
JOB_NAME_TRAIN="llama2-7b-p${PERCENTAGE}-lora-seed${DATA_SEED}-first"
TRAIN_FILES=/scratch-shared/ir2-less/selected_data/${JOB_NAME_TRAIN}/${TARGET_TASK_NAME}/top_p${PERCENTAGE}.jsonl
MODEL_PATH=meta-llama/Llama-2-7b-hf
JOB_NAME=llama2-7b-less-p${PERCENTAGE}-lora-seed${DATA_SEED}-first-${TARGET_TASK_NAME}


bash less/scripts/train/lora_train.sh "$TRAIN_FILES" "$MODEL_PATH" "$JOB_NAME" 
