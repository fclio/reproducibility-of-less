#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=2
#SBATCH --job-name=LESS-IR
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --time=02:00:00
#SBATCH --output=slurm_output/ir_training_%A.out

module purge
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

pip install -r requirement.txt
pip install -e .

# Set variables for warmup training
DATA_DIR="data/ir_datasets"  # Use IR-specific datasets
MODEL_PATH="meta-llama/Llama-2-7b-hf"
PERCENTAGE=0.001
DATA_SEED=42
JOB_NAME="llama2-7b-ir-p${PERCENTAGE}-lora-seed${DATA_SEED}"

# Run warmup training (IR-specific)
bash less/scripts/train/warmup_lora_train.sh "$DATA_DIR" "$MODEL_PATH" "$PERCENTAGE" "$DATA_SEED" "$JOB_NAME"

# Extract gradients for IR tasks
CKPT=32
TRAINING_DATA_NAME=msmarco  # IR dataset
TRAINING_DATA_FILE=data/train/processed/msmarco/msmarco_train.jsonl
GRADIENT_TYPE="adam"
MODEL_PATH=../out/llama2-7b-ir-p0.001-lora-seed42/checkpoint-${CKPT}
OUTPUT_PATH=../grads/llama2-7b-ir-p0.001-lora-seed42/${TRAINING_DATA_NAME}-ckpt${CKPT}-${GRADIENT_TYPE}
DIMS="8192"

bash less/scripts/get_info/grad/get_train_lora_grads.sh "$TRAINING_DATA_FILE" "$MODEL_PATH" "$OUTPUT_PATH" "$DIMS" "$GRADIENT_TYPE"

# Validation gradients for target task
TASK=trec  # Target IR evaluation task
MODEL_PATH=../out/llama2-7b-ir-p0.001-lora-seed42/checkpoint-${CKPT}
OUTPUT_PATH=../grads/llama2-7b-ir-p0.001-lora-seed42/${TASK}-ckpt${CKPT}-sgd
DATA_DIR="data"
DIMS="4096 8192"

bash less/scripts/get_info/grad/get_eval_lora_grads.sh "$TASK" "$DATA_DIR" "$MODEL_PATH" "$OUTPUT_PATH" "$DIMS"

# Data Selection
DIM=8192  # Choose appropriate dimensionality
GRADIENT_PATH=../grads/llama2-7b-ir-p0.001-lora-seed42/${TRAINING_DATA_NAME}-ckpt${CKPT}-${GRADIENT_TYPE}/dim${DIM}
TRAIN_FILE_NAMES="msmarco"
CKPTS="32"
CHECKPOINT_WEIGHTS="1.6877e-05"  # Adjust for IR task

VALIDATION_GRADIENT_PATH=../grads/llama2-7b-ir-p0.001-lora-seed42/${TASK}-ckpt${CKPT}-sgd/dim${DIM}
TARGET_TASK_NAMES="trec"
SELECTED_DATA_OUTPUT_PATH="../selected_data/ir"

bash less/scripts/data_selection/matching.sh "$GRADIENT_PATH" "$TRAIN_FILE_NAMES" "$CKPTS" "$CHECKPOINT_WEIGHTS" "$VALIDATION_GRADIENT_PATH" "$TARGET_TASK_NAMES" "$SELECTED_DATA_OUTPUT_PATH"

python3 -m less.data_selection.write_selected_data --target_task_names ${TARGET_TASK_NAMES} --train_file_names ${TRAIN_FILE_NAMES} --train_files data/train/processed/msmarco/msmarco_train.jsonl --output_path $SELECTED_DATA_OUTPUT_PATH --percentage 0.001

# Fine-tuning
TARGET_TASK_NAME="trec"
TRAIN_FILES=../selected_data/ir/${TARGET_TASK_NAME}/top_p${PERCENTAGE}.jsonl
MODEL_PATH=meta-llama/Llama-2-7b-hf
JOB_NAME=llama2-7b-ir-less-p${PERCENTAGE}-lora

bash less/scripts/train/lora_train.sh "$TRAIN_FILES" "$MODEL_PATH" "$JOB_NAME"
