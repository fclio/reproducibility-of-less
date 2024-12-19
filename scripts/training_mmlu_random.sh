#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=2
#SBATCH --job-name=random_MMLU
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --time=05:00:00
#SBATCH --output=slurm_output2/random_MMLU_%A.out

module purge
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

# CKPT=32

# TASK=mmlu
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
# TARGET_TASK_NAMES="mmlu"
# SELECTED_DATA_OUTPUT_PATH="../selected_data"

# bash less/scripts/data_selection/matching.sh "$GRADIENT_PATH" "$TRAIN_FILE_NAMES" "$CKPTS" "$CHECKPOINT_WEIGHTS" "$VALIDATION_GRADIENT_PATH" "$TARGET_TASK_NAMES" "$SELECTED_DATA_OUTPUT_PATH"
DATA_SEED=4
PERCENTAGE=0.05
TARGET_TASK_NAME="mmlu"
JOB_NAME_TRAIN="llama2-7b-p${PERCENTAGE}-seed${DATA_SEED}"
TRAIN_FILES=/scratch-shared/ir2-less/selected_data/${JOB_NAME_TRAIN}/${TARGET_TASK_NAME}/random_top_p${PERCENTAGE}.jsonl
MODEL_PATH=meta-llama/Llama-2-13b-hf
JOB_NAME=llama2-13b-less-p${PERCENTAGE}-random-seed${DATA_SEED}-mmlu


bash less/scripts/train/lora_train.sh "$TRAIN_FILES" "$MODEL_PATH" "$JOB_NAME" 