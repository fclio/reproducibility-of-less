#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=matching_1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=slurm_output/matching_MMLU_%A.out

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


DIM=8192 # decide which dimension to use
TRAIN_FILE_NAMES="dolly"
DATA_SEED=4
PERCENTAGE=0.05
CKPTS="422" # checkpoing index
CHECKPOINT_WEIGHTS="1.6877e-05 " # average lr of the epoch
TARGET_TASK_NAMES="mmlu"
JOB_NAME="llama2-7b-p${PERCENTAGE}-lora-seed${DATA_SEED}"

GRADIENT_PATH=/scratch-shared/ir2-less/grads/${JOB_NAME}/{}-ckpt{}-adam/dim${DIM}

#TODO FIND THE CORRECT CHECKPOINT WEIGHT!

VALIDATION_GRADIENT_PATH=/scratch-shared/ir2-less/grads/${JOB_NAME}/{}-ckpt{}-sgd/dim${DIM}
SELECTED_DATA_OUTPUT_PATH="/scratch-shared/ir2-less/selected_data/${JOB_NAME}"

if [[ ! -d $SELECTED_DATA_OUTPUT_PATH ]]; then
    mkdir -p $SELECTED_DATA_OUTPUT_PATH
fi

bash less/scripts/data_selection/matching.sh "$GRADIENT_PATH" "$TRAIN_FILE_NAMES" "$CKPTS" "$CHECKPOINT_WEIGHTS" "$VALIDATION_GRADIENT_PATH" "$TARGET_TASK_NAMES" "$SELECTED_DATA_OUTPUT_PATH"

python3 -m less.data_selection.write_selected_data --target_task_names ${TARGET_TASK_NAMES} --train_file_names ${TRAIN_FILE_NAMES} --train_files data/train/processed/dolly/dolly_data.jsonl --output_path $SELECTED_DATA_OUTPUT_PATH --percentage ${PERCENTAGE}
