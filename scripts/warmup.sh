#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=2
#SBATCH --job-name=LESS-warmup
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus-per-node=2
#SBATCH --time=02:00:00
#SBATCH --output=slurm_output/warmup_training_%A.out

module purge
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

# Set variables for warmup training
DATA_DIR="data"
MODEL_PATH="meta-llama/Llama-2-7b-hf"
PERCENTAGE=0.0001
DATA_SEED=3
JOB_NAME="llama2-7b-p${PERCENTAGE}-lora-seed${DATA_SEED}"

# # Save locally in home/scur2847
# OUTPUT_DIR=../out/${job_name}
# if [[ ! -d $OUTPUT_DIR ]]; then
#     mkdir -p $OUTPUT_DIR
# fi

# Save in scratch-shared
OUTPUT_DIR=/scratch-shared/ir2-less/out/${JOB_NAME}
if [[ ! -d $OUTPUT_DIR ]]; then
    mkdir -p $OUTPUT_DIR
fi

# Run warmup training
bash less/scripts/train/warmup_lora_train.sh "$DATA_DIR" "$MODEL_PATH" "$PERCENTAGE" "$DATA_SEED" "$JOB_NAME" "$OUTPUT_DIR"