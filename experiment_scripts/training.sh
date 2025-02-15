#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=2
#SBATCH --job-name=training
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --time=10:00:00
#SBATCH --output=slurm_output4/training_%A.out

module purge
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

TARGET_TASK_NAME=$1
MODEL=$2
DATA_SEED=$3
CHECKPOINTS=$4
MODE=$5
LESS_REPO_DIR=$6
LESS_OUTPUT_DIR=$7

if [[ "$2" == "llama2-7b" ]]; then
    MODEL_PATH=meta-llama/Llama-2-7b-hf
elif [[ "$2" == "llama2-13b" ]]; then
    MODEL_PATH=meta-llama/Llama-2-13b-hf
else
    echo "Unknown model."
fi

PERCENTAGE=0.05
if [[ "$5" == "T" ]]; then
    JOB_NAME_TRAIN="llama2-7b-p${PERCENTAGE}-lora-seed${DATA_SEED}"
    JOB_NAME=${MODEL}-less-p${PERCENTAGE}-lora-seed${DATA_SEED}-${TARGET_TASK_NAME}-${CHECKPOINTS}-T
    TRAIN_FILES=${LESS_OUTPUT_DIR}/selected_data/${JOB_NAME_TRAIN}/${CHECKPOINTS}/${TARGET_TASK_NAME}/top_p${PERCENTAGE}.jsonl
elif [[ "$5" == "BM25" ]]; then
    JOB_NAME_TRAIN="${MODEL}-p${PERCENTAGE}-lora-seed${DATA_SEED}"
    JOB_NAME=${MODEL}-less-p${PERCENTAGE}-lora-seed${DATA_SEED}-${TARGET_TASK_NAME}-BM25
    TRAIN_FILES=${LESS_OUTPUT_DIR}/selected_data/${JOB_NAME_TRAIN}/baseline/${TARGET_TASK_NAME}/bm25_top_p${PERCENTAGE}.jsonl
elif [[ "$5" == "random" ]]; then
    JOB_NAME_TRAIN="${MODEL}-p${PERCENTAGE}-lora-seed${DATA_SEED}"
    JOB_NAME=${MODEL}-less-p${PERCENTAGE}-lora-seed${DATA_SEED}-${TARGET_TASK_NAME}-random
    TRAIN_FILES=${LESS_OUTPUT_DIR}/selected_data/${JOB_NAME_TRAIN}/baseline/${TARGET_TASK_NAME}/random_top_p${PERCENTAGE}.jsonl
elif [[ "$5" == "IR" ]]; then
    JOB_NAME_TRAIN="${MODEL}-p${PERCENTAGE}-lora-seed${DATA_SEED}-first"
    JOB_NAME=${MODEL}-less-p${PERCENTAGE}-lora-seed${DATA_SEED}-${TARGET_TASK_NAME}-${CHECKPOINTS}
    TRAIN_FILES=${LESS_OUTPUT_DIR}/selected_data/${JOB_NAME_TRAIN}/${CHECKPOINTS}/${TARGET_TASK_NAME}/top_p${PERCENTAGE}.jsonl
elif [[ "$5" == "IR_BM25" ]]; then
    JOB_NAME_TRAIN="${MODEL}-p${PERCENTAGE}-lora-seed${DATA_SEED}-first"
    JOB_NAME=${MODEL}-less-p${PERCENTAGE}-lora-seed${DATA_SEED}-${TARGET_TASK_NAME}-BM25
    TRAIN_FILES=${LESS_OUTPUT_DIR}/selected_data/${JOB_NAME_TRAIN}/baseline/${TARGET_TASK_NAME}/bm25_top_p${PERCENTAGE}.jsonl
elif [[ "$5" == "IR_random" ]]; then
    JOB_NAME_TRAIN="${MODEL}-p${PERCENTAGE}-lora-seed${DATA_SEED}-first"
    JOB_NAME=${MODEL}-less-p${PERCENTAGE}-lora-seed${DATA_SEED}-${TARGET_TASK_NAME}-random
    TRAIN_FILES=${LESS_OUTPUT_DIR}/selected_data/${JOB_NAME_TRAIN}/baseline/${TARGET_TASK_NAME}/random_top_p${PERCENTAGE}.jsonl
else
    JOB_NAME_TRAIN="${MODEL}-p${PERCENTAGE}-lora-seed${DATA_SEED}"
    JOB_NAME=${MODEL}-less-p${PERCENTAGE}-lora-seed${DATA_SEED}-${TARGET_TASK_NAME}-${CHECKPOINTS}
    TRAIN_FILES=${LESS_OUTPUT_DIR}/selected_data/${JOB_NAME_TRAIN}/${CHECKPOINTS}/${TARGET_TASK_NAME}/top_p${PERCENTAGE}.jsonl
fi

bash less/scripts/train/lora_train.sh "$TRAIN_FILES" "$MODEL_PATH" "$JOB_NAME" 
