#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=eval_grad_datastore
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=slurm_output/build_eval_grad_scifact_%A.out

module purge
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

#7b: 422 845 1268 1688 
#13b: 211 422 634 844
#7b: 11,22,34,44

#first-0.5: 62 124 187 248
#msmarco-0.5: 573 1146 1720 2292
# slurm_output/build_eval_grad_first_%A_msmarco_0.05_1146.out

# Constants
PERCENTAGE=0.05
DATA_SEED=3
TRAINING_DATA_NAME=scifact
GRADIENT_TYPE="sgd"
SHOT=15
TASK="scifact"
DATA_DIR="data"
DIMS="8192"

# CKPT values to iterate over
CKPT_VALUES=(62 124 187 248)  # Add more checkpoint numbers as needed

# Loop over CKPT values
for CKPT in "${CKPT_VALUES[@]}"; do
  # Update paths and names dynamically based on the current CKPT
  TRAINING_DATA_FILE="${DATA_DIR}/eval/${TRAINING_DATA_NAME}/${TRAINING_DATA_NAME}_dev.jsonl"
  JOB_NAME="llama2-7b-p${PERCENTAGE}-lora-seed${DATA_SEED}-first"
  MODEL_PATH="/scratch-shared/ir2-less/out/${JOB_NAME}/checkpoint-${CKPT}"
  OUTPUT_PATH="/scratch-shared/ir2-less/grads/${JOB_NAME}/shot_${SHOT}/${TRAINING_DATA_NAME}-ckpt${CKPT}-${GRADIENT_TYPE}"
  
  # Print information for debugging/logging
  echo "Running with CKPT=${CKPT}, TRAINING_DATA_NAME=${TRAINING_DATA_NAME}"
  echo "MODEL_PATH=${MODEL_PATH}"
  echo "OUTPUT_PATH=${OUTPUT_PATH}"
  
  # Execute gradient computation for training
  bash less/scripts/get_info/grad/get_train_lora_grads.sh "$TRAINING_DATA_FILE" "$MODEL_PATH" "$OUTPUT_PATH" "$DIMS" "$GRADIENT_TYPE"
done