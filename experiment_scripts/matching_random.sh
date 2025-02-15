#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=matching_random
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=slurm_output4/matching_random_%A.out

module purge
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

TARGET_TASK_NAMES=$1
MODEL=$2
DATA_SEED=$3
IR=$4
LESS_REPO_DIR=$5
LESS_OUTPUT_DIR=$6

PERCENTAGE=0.05

if [[ "$4" == "IR" ]]; then
    JOB_NAME="${MODEL}-p${PERCENTAGE}-lora-seed${DATA_SEED}-first"
    TRAIN_FILE_NAMES="first"
    TRAIN_FILES="data/train/processed/first/first_data.jsonl"
else
    JOB_NAME="${MODEL}-p${PERCENTAGE}-lora-seed${DATA_SEED}"
    TRAIN_FILE_NAMES="dolly oasst1 flan_v2 cot"
    TRAIN_FILES="data/train/processed/dolly/dolly_data.jsonl data/train/processed/oasst1/oasst1_data.jsonl data/train/processed/flan_v2/flan_v2_data.jsonl data/train/processed/cot/cot_data.jsonl"
fi

PERCENTAGE=0.05

TRAIN_DATA_PATH="${LESS_REPO_DIR}/data/train/processed"
EVAL_DATA_PATH="${LESS_REPO_DIR}/data/eval"

SELECTED_DATA_OUTPUT_PATH="${LESS_OUTPUT_DIR}/selected_data/${JOB_NAME}/baseline"

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

# Run write_selected_data
python3 -m less.data_selection.write_selected_data_random \
    --target_task_names ${TARGET_TASK_NAMES} \
    --train_file_names ${TRAIN_FILE_NAMES} \
    --train_files ${TRAIN_FILES} \
    --output_path $SELECTED_DATA_OUTPUT_PATH \
    --percentage ${PERCENTAGE}
