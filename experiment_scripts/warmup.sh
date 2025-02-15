#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=2
#SBATCH --job-name=warmup
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --time=24:00:00
#SBATCH --output=slurm_output_IR/warmup_%A.out

module purge
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

MODEL=$1
DATA_SEED=$2
IR=$3
LESS_REPO_DIR=$4
LESS_OUTPUT_DIR=$5

if [[ "$1" == "llama2-7b" ]]; then
    MODEL_PATH=meta-llama/Llama-2-7b-hf
elif [[ "$1" == "llama2-13b" ]]; then
    MODEL_PATH=meta-llama/Llama-2-13b-hf
else
    echo "Unknown model."
fi

# Set variables for warmup training
DATA_DIR="data"
PERCENTAGE=0.05

if [[ "$3" == "IR" ]]; then
    JOB_NAME="${MODEL}-p${PERCENTAGE}-lora-seed${DATA_SEED}-msmarco"
else
    JOB_NAME="${MODEL}-p${PERCENTAGE}-lora-seed${DATA_SEED}"
fi

# Save in scratch-shared
OUTPUT_DIR=${LESS_OUTPUT_DIR}/out/${JOB_NAME}
if [[ ! -d $OUTPUT_DIR ]]; then
    mkdir -p $OUTPUT_DIR
fi

if [[ "$3" == "IR" ]]; then
    bash less/scripts/train/warmup_lora_train_ir.sh "$DATA_DIR" "$MODEL_PATH" "$PERCENTAGE" "$DATA_SEED" "$JOB_NAME" "$OUTPUT_DIR"
else
    bash less/scripts/train/warmup_lora_train.sh "$DATA_DIR" "$MODEL_PATH" "$PERCENTAGE" "$DATA_SEED" "$JOB_NAME" "$OUTPUT_DIR"
fi

