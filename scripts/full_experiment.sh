#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=2
#SBATCH --job-name=LESS-warmup
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --time=02:00:00
#SBATCH --output=slurm_output/warmup_training_%A.out

module purge
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

pip install -r requirement.txt
pip install -e .

# Set variables for warmup training
DATA_DIR="data"
MODEL_PATH="meta-llama/Llama-2-7b-hf"
# MODEL_PATH="winglian/Llama-2-3b-hf"
PERCENTAGE=0.001
DATA_SEED=3
JOB_NAME="llama2-7b-p${PERCENTAGE}-lora-seed${DATA_SEED}"

# Run warmup training
bash less/scripts/train/warmup_lora_train.sh "$DATA_DIR" "$MODEL_PATH" "$PERCENTAGE" "$DATA_SEED" "$JOB_NAME"

CKPT=32

TRAINING_DATA_NAME=dolly
TRAINING_DATA_FILE=data/train/processed/dolly/dolly_data.jsonl # when changing data name, change the data path accordingly
GRADIENT_TYPE="adam"
MODEL_PATH=../out/llama2-7b-p0.001-lora-seed3/checkpoint-${CKPT}
OUTPUT_PATH=../grads/llama2-7b-p0.001-lora-seed3/${TRAINING_DATA_NAME}-ckpt${CKPT}-${GRADIENT_TYPE}
DIMS="8192"

bash less/scripts/get_info/grad/get_train_lora_grads.sh "$TRAINING_DATA_FILE" "$MODEL_PATH" "$OUTPUT_PATH" "$DIMS" "$GRADIENT_TYPE"

CKPT=32

TASK=mmlu
MODEL_PATH=../out/llama2-7b-p0.001-lora-seed3/checkpoint-${CKPT}
OUTPUT_PATH=../grads/llama2-7b-p0.001-lora-seed3/${TASK}-ckpt${CKPT}-sgd
DATA_DIR="data"
DIMS="4096 8192"

bash less/scripts/get_info/grad/get_eval_lora_grads.sh "$TASK" "$DATA_DIR" "$MODEL_PATH" $OUTPUT_PATH "$DIMS"

DIM=8192 # decide which dimension to use
GRADIENT_PATH=../grads/llama2-7b-p0.001-lora-seed3/{}-ckpt{}-adam/dim${DIM}
TRAIN_FILE_NAMES="dolly"
CKPTS="32" # checkpoing index
CHECKPOINT_WEIGHTS="1.6877e-05 " # average lr of the epoch

#TODO FIND THE CORRECT CHECKPOINT WEIGHT!

VALIDATION_GRADIENT_PATH=../grads/llama2-7b-p0.001-lora-seed3/{}-ckpt{}-sgd/dim${DIM}
TARGET_TASK_NAMES="mmlu"
SELECTED_DATA_OUTPUT_PATH="../selected_data"

bash less/scripts/data_selection/matching.sh "$GRADIENT_PATH" "$TRAIN_FILE_NAMES" "$CKPTS" "$CHECKPOINT_WEIGHTS" "$VALIDATION_GRADIENT_PATH" "$TARGET_TASK_NAMES" "$SELECTED_DATA_OUTPUT_PATH"

python3 -m less.data_selection.write_selected_data --target_task_names ${TARGET_TASK_NAMES} --train_file_names ${TRAIN_FILE_NAMES} --train_files data/train/processed/dolly/dolly_data.jsonl --output_path $SELECTED_DATA_OUTPUT_PATH --percentage 0.001

TARGET_TASK_NAME="mmlu"
# PERCENTAGE=0.001
TRAIN_FILES=../selected_data/${TARGET_TASK_NAME}/top_p${PERCENTAGE}.jsonl
MODEL_PATH=meta-llama/Llama-2-7b-hf
JOB_NAME=llama2-7b-less-p${PERCENTAGE}-lora

bash less/scripts/train/lora_train.sh "$TRAIN_FILES" "$MODEL_PATH" "$JOB_NAME" 
