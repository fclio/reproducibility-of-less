#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=matching_1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=slurm_output/matching_scifact_%A.out

module purge
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1



DIM=8192 # decide which dimension to use
TRAIN_FILE_NAMES="first"
TRAIN_FILES="data/train/processed/first/first_data.jsonl"
DATA_SEED=3
PERCENTAGE=0.05


#7b: 11,22,34,44
#7b: 422 845 1268 1688 
#13b: 211 422 634 844
#first-0.5: 62 124 187 248
#msmarco-0.5: 573 1146 1720 2292
#first-13b-0.5: 31 62 93 124
CKPTS="62 124 187 248" # checkpoing index

# to fill in!!!
CHECKPOINT_WEIGHTS="1.724331e-05 1.28895e-05 7.71515e-06 2.56565e-06" # average lr of the epoch
TARGET_TASK_NAMES="scifact"

JOB_NAME="llama2-7b-p${PERCENTAGE}-lora-seed${DATA_SEED}-first"

GRADIENT_PATH=/scratch-shared/ir2-less/grads/${JOB_NAME}/{}-ckpt{}-adam/dim${DIM}

#TODO FIND THE CORRECT CHECKPOINT WEIGHT!
SHOT=15

VALIDATION_GRADIENT_PATH=/scratch-shared/ir2-less/grads/${JOB_NAME}/shot_${SHOT}/{}-ckpt{}-sgd/dim${DIM}
SELECTED_DATA_OUTPUT_PATH="/scratch-shared/ir2-less/selected_data/${JOB_NAME}/shot_${SHOT}"

if [[ ! -d $SELECTED_DATA_OUTPUT_PATH ]]; then
    mkdir -p $SELECTED_DATA_OUTPUT_PATH
fi

bash less/scripts/data_selection/matching.sh "$GRADIENT_PATH" "$TRAIN_FILE_NAMES" "$CKPTS" "$CHECKPOINT_WEIGHTS" "$VALIDATION_GRADIENT_PATH" "$TARGET_TASK_NAMES" "$SELECTED_DATA_OUTPUT_PATH"

python3 -m less.data_selection.write_selected_data --target_task_names ${TARGET_TASK_NAMES} --train_file_names ${TRAIN_FILE_NAMES} --train_files ${TRAIN_FILES} --output_path $SELECTED_DATA_OUTPUT_PATH --percentage ${PERCENTAGE}
