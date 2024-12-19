#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=2
#SBATCH --job-name=random_tydiqa
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --time=05:00:00
#SBATCH --output=slurm_output2/random_tydiqa_%A.out

module purge
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

DATA_SEED=4
PERCENTAGE=0.05
TARGET_TASK_NAME="tydiqa"
JOB_NAME_TRAIN="llama2-7b-p${PERCENTAGE}-seed${DATA_SEED}"
TRAIN_FILES=/scratch-shared/ir2-less/selected_data/${JOB_NAME_TRAIN}/${TARGET_TASK_NAME}/random_top_p${PERCENTAGE}.jsonl
MODEL_PATH=meta-llama/Llama-2-13b-hf
JOB_NAME=llama2-13b-less-p${PERCENTAGE}-random-seed${DATA_SEED}-tydiqa


bash less/scripts/train/lora_train.sh "$TRAIN_FILES" "$MODEL_PATH" "$JOB_NAME" 