#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=training
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=05:00:00
#SBATCH --output=slurm_output/training_tydiqa_%A.out

module purge
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

# CKPT=32

# TASK=tydiqa
# MODEL_PATH=../out/llama2-7b-p0.001-lora-seed3/checkpoint-${CKPT}
# OUTPUT_PATH=../grads/llama2-7b-p0.001-lora-seed3/${TRAINING_DATA_NAME}-ckpt${CKPT}-${GRADIENT_TYPE}
# DATA_DIR="data"
# DIMS="4096 8192"

# bash less/scripts/get_info/grad/get_eval_lora_grads.sh "$TASK" "$DATA_DIR" "$MODEL_PATH" $OUTPUT_PATH "$DIMS"


# DIM=8192 # decide which dimension to use
# GRADIENT_PATH=../grads/llama2-7b-p0.001-lora-seed3/{}-ckpt{}-adam/dim${DIM}
# TRAIN_FILE_NAMES="dolly"
# CKPTS="32" # checkpoing index
# CHECKPOINT_WEIGHTS="1.6877e-05 " # average lr of the epoch

# #TODO FIND THE CORRECT CHECKPOINT WEIGHT!

# VALIDATION_GRADIENT_PATH=../grads/llama2-7b-p0.001-lora-seed3/{}-ckpt{}-sgd/dim${DIM}
# TARGET_TASK_NAMES="tydiqa"
# SELECTED_DATA_OUTPUT_PATH="../selected_data"

# bash less/scripts/data_selection/matching.sh "$GRADIENT_PATH" "$TRAIN_FILE_NAMES" "$CKPTS" "$CHECKPOINT_WEIGHTS" "$VALIDATION_GRADIENT_PATH" "$TARGET_TASK_NAMES" "$SELECTED_DATA_OUTPUT_PATH"
DATA_SEED=4
PERCENTAGE=0.05
TARGET_TASK_NAME="tydiqa"
JOB_NAME_TRAIN="llama2-7b-p${PERCENTAGE}-lora-seed${DATA_SEED}"
TRAIN_FILES=/scratch-shared/ir2-less/selected_data/${JOB_NAME_TRAIN}/${TARGET_TASK_NAME}/top_p${PERCENTAGE}.jsonl
MODEL_PATH=meta-llama/Llama-2-7b-hf
JOB_NAME=llama2-7b-less-p${PERCENTAGE}-lora-seed${DATA_SEED}-tydiqa


# CKPTS="422" # checkpoing index
# TARGET_TASK_NAME="tydiqa"



bash less/scripts/train/lora_train.sh "$TRAIN_FILES" "$MODEL_PATH" "$JOB_NAME" 
