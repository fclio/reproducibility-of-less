#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=eval_bbh_after_finetuning
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=slurm_output/eval_bbh_after_finetuning_%A.out

# Load necessary modules
module purge
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

# Source the evaluation scripts
source eval.sh
source eval_bbh.sh

JOB_NAME=$1
LESS_REPO_DIR=$2
LESS_OUTPUT_DIR=$3
# JOB_NAME=${MODEL}-less-p${PERCENTAGE}-lora-seed${DATA_SEED}-${TARGET_TASK_NAME}-${CHECKPOINTS}

# Set the model directory relative to the script's location
MODEL_DIR="${LESS_OUTPUT_DIR}/out/${JOB_NAME}"

# Run the evaluation on bbh dataset
eval_bbh "$MODEL_DIR"

# # Extract and print the evaluation results
RESULT=$(extract_bbh "$MODEL_DIR")
echo "bbh Evaluation Result: $RESULT%"
echo -e "${JOB_NAME} ${RESULT}\n" >> ../experiment_results.txt
