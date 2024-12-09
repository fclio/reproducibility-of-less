#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=random_matching
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=slurm_output/random_matching_tydiqa_%A.out

module purge
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

# Variables
TRAIN_FILE_NAMES="dolly oasst1 flan_v2 cot"        # Names of training data files (space-separated if multiple)
DATA_SEED=4
PERCENTAGE=0.05
TARGET_TASK_NAMES="tydiqa"         # Names of evaluation tasks
JOB_NAME="llama2-7b-p${PERCENTAGE}-seed${DATA_SEED}"

TRAIN_DATA_PATH="data/train/processed"  # Base path to training data
EVAL_DATA_PATH="data/eval"              # Base path to evaluation data
SELECTED_DATA_OUTPUT_PATH="/home/scur2832/LESS/selected_data/${JOB_NAME}"

# SELECTED_DATA_OUTPUT_PATH="/scratch-shared/ir2-less/selected_data/${JOB_NAME}"

# Create output directory if it doesn't exist
if [[ ! -d $SELECTED_DATA_OUTPUT_PATH ]]; then
    mkdir -p $SELECTED_DATA_OUTPUT_PATH
fi

# Run BM25 matching
python3 -m less.data_selection.matching_random \
    --train_data_path "$TRAIN_DATA_PATH" \
    --train_file_names $TRAIN_FILE_NAMES \
    --eval_data_path "$EVAL_DATA_PATH" \
    --target_task_names $TARGET_TASK_NAMES \
    --output_path "$SELECTED_DATA_OUTPUT_PATH"

TRAIN_FILES="data/train/processed/dolly/dolly_data.jsonl data/train/processed/oasst1/oasst1_data.jsonl data/train/processed/flan_v2/flan_v2_data.jsonl data/train/processed/cot/cot_data.jsonl"

# Run write_selected_data
python3 -m less.data_selection.write_selected_data_random \
    --target_task_names ${TARGET_TASK_NAMES} \
    --train_file_names ${TRAIN_FILE_NAMES} \
    --train_files ${TRAIN_FILES} \
    --output_path $SELECTED_DATA_OUTPUT_PATH \
    --percentage ${PERCENTAGE}