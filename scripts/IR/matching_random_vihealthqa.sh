#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=rand_vihealthqa
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=slurm_output2/random_matching_vihealthqa_%A.out

module purge
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1



TRAIN_FILE_NAMES="first"
DATA_SEED=4
PERCENTAGE=0.05
TARGET_TASK_NAMES="vihealthqa"
JOB_NAME="llama2-7b-p${PERCENTAGE}-seed${DATA_SEED}"

TRAIN_DATA_PATH="/home/scur2847/ir2-less-data/data/train/processed"
EVAL_DATA_PATH="/home/scur2847/ir2-less-data/data/eval"
# SELECTED_DATA_OUTPUT_PATH="/home/scur2847/ir2-less-data/selected_data/${JOB_NAME}"

SELECTED_DATA_OUTPUT_PATH="/scratch-shared/ir2-less/selected_data/${JOB_NAME}"

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

TRAIN_FILES="data/train/processed/first/first_data.jsonl"

# Run write_selected_data
python3 -m less.data_selection.write_selected_data_random \
    --target_task_names ${TARGET_TASK_NAMES} \
    --train_file_names ${TRAIN_FILE_NAMES} \
    --train_files ${TRAIN_FILES} \
    --output_path $SELECTED_DATA_OUTPUT_PATH \
    --percentage ${PERCENTAGE}