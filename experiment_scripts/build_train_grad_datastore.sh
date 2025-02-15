#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=train_grad
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=10:00:00
#SBATCH --output=slurm_output_IR/train_grad_%A.out

module purge
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

TRAINING_DATA_NAME=$1
CKPT=$2
MODEL=$3
DATA_SEED=$4
IR=$5
LESS_REPO_DIR=$6
LESS_OUTPUT_DIR=$7

if [[ "$1" == "cot" ]]; then
    TRAINING_DATA_FILE="data/train/processed/cot/cot_data.jsonl"
    GRADIENT_TYPE="adam"
elif [[ "$1" == "dolly" ]]; then
    TRAINING_DATA_FILE="data/train/processed/dolly/dolly_data.jsonl"
    GRADIENT_TYPE="adam"
elif [[ "$1" == "flan_v2" ]]; then
    TRAINING_DATA_FILE="data/train/processed/flan_v2/flan_v2_data.jsonl"
    GRADIENT_TYPE="adam"
elif [[ "$1" == "oasst1" ]]; then
    TRAINING_DATA_FILE="data/train/processed/oasst1/oasst1_data.jsonl"
    GRADIENT_TYPE="adam"
elif [[ "$1" == "first" ]]; then
    TRAINING_DATA_FILE=data/train/processed/first/first_data.jsonl
    GRADIENT_TYPE="adam"
elif [[ "$1" == "msmarco" ]]; then
    TRAINING_DATA_FILE=data/train/processed/msmarco/msmarco_data.jsonl
    GRADIENT_TYPE="adam"
elif [[ "$1" == "nfcorpus" ]]; then
    TRAINING_DATA_FILE=data/eval/nfcorpus/nfcorpus_dev.jsonl
    GRADIENT_TYPE="sgd"
elif [[ "$1" == "scifact" ]]; then
    TRAINING_DATA_FILE=data/eval/scifact/scifact_dev.jsonl
    GRADIENT_TYPE="sgd"
elif [[ "$1" == "vihealthqa" ]]; then
    TRAINING_DATA_FILE=data/eval/vihealthqa/vihealthqa_dev.jsonl
    GRADIENT_TYPE="sgd"
elif [[ "$1" == "fiqa" ]]; then
    TRAINING_DATA_FILE=data/eval/vihealthqa/vihealthqa_dev.jsonl
    GRADIENT_TYPE="sgd"
else
    echo "Unknown training data task."
fi

PERCENTAGE=0.05

if [[ "$5" == "IR" ]]; then
    JOB_NAME="${MODEL}-p${PERCENTAGE}-lora-seed${DATA_SEED}-msmarco"
else
    JOB_NAME="${MODEL}-p${PERCENTAGE}-lora-seed${DATA_SEED}"
fi
MODEL_PATH=${LESS_OUTPUT_DIR}/out/${JOB_NAME}/checkpoint-${CKPT}
OUTPUT_PATH=${LESS_OUTPUT_DIR}/grads/${JOB_NAME}/${TRAINING_DATA_NAME}-ckpt${CKPT}-${GRADIENT_TYPE}
DIMS="8192"

if [[ ! -d $OUTPUT_PATH ]]; then
    mkdir -p $OUTPUT_PATH
fi

bash less/scripts/get_info/grad/get_train_lora_grads.sh "$TRAINING_DATA_FILE" "$MODEL_PATH" "$OUTPUT_PATH" "$DIMS" "$GRADIENT_TYPE"
