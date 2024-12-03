#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=eval_mmlu_after_finetuning
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=slurm_output/eval_mmlu_after_finetuning_%A.out

# Load necessary modules
module purge
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

# MAKE SURE TO INSTALL OPEN INSTRUCT AND ADD THE REQUIRED FUNCTION!!!!!!!!!!!!!!!!!

cd evaluation

# Source the evaluation scripts
source eval.sh
source eval_mmlu.sh


DATA_SEED=4
PERCENTAGE=0.05
JOB_NAME=llama2-7b-less-p${PERCENTAGE}-lora-seed${DATA_SEED}

# Set the model directory relative to the script's location
MODEL_DIR="../../out/${JOB_NAME}"


# Run the evaluation on MMLU dataset
eval_mmlu "$MODEL_DIR"

# Extract and print the evaluation results
RESULT=$(extract_mmlu "$MODEL_DIR")
echo "MMLU Evaluation Result: $RESULT%"
