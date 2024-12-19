#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=matching_ir
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=slurm_output/matching_ir_%A.out

module purge
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

TARGET_TASK_NAMES=$1
MODEL=$2
DATA_SEED=$3
CHECKPOINTS=$4

foo=${CHECKPOINTS}
CKPTS=""
if [[ "$2" == "llama2-7b" ]]; then
    for (( i=0; i<${#foo}; i++ )); do
        if [[ "${foo:$i:1}" == "1" ]]; then
            CKPTS+=" 62"
        elif [[ "${foo:$i:1}" == "2" ]]; then
            CKPTS+=" 124"
        elif [[ "${foo:$i:1}" == "3" ]]; then
            CKPTS+=" 187"
        elif [[ "${foo:$i:1}" == "4" ]]; then
            CKPTS+=" 248"
        else
            echo "Unknown checkpoint idx."
        fi
    done
elif [[ "$2" == "llama2-13b" ]]; then
    for (( i=0; i<${#foo}; i++ )); do
        if [[ "${foo:$i:1}" == "1" ]]; then
            CKPTS+=" 31"
        elif [[ "${foo:$i:1}" == "2" ]]; then
            CKPTS+=" 62"
        elif [[ "${foo:$i:1}" == "3" ]]; then
            CKPTS+=" 93"
        elif [[ "${foo:$i:1}" == "4" ]]; then
            CKPTS+=" 124"
        else
            echo "Unknown checkpoint idx."
        fi
    done
else
    echo "Unknown model."
fi

CHECKPOINT_WEIGHTS=""
for (( i=0; i<${#foo}; i++ )); do
    if [[ "${foo:$i:1}" == "1" ]]; then
        CHECKPOINT_WEIGHTS+=" 1.724331e-05"
    elif [[ "${foo:$i:1}" == "2" ]]; then
        CHECKPOINT_WEIGHTS+=" 1.28895e-05"
    elif [[ "${foo:$i:1}" == "3" ]]; then
        CHECKPOINT_WEIGHTS+=" 7.71515e-06"
    elif [[ "${foo:$i:1}" == "4" ]]; then
        CHECKPOINT_WEIGHTS+=" 2.56565e-06"
    else
        echo "Unknown checkpoint idx."
    fi
done
# CHECKPOINT_WEIGHTS="1.724331e-05 1.28895e-05 7.71515e-06 2.56565e-06" # average lr of the epoch

echo ${CKPTS}
echo ${CHECKPOINT_WEIGHTS}


DIM=8192 # decide which dimension to use
TRAIN_FILE_NAMES=first
TRAIN_FILES="data/train/processed/first/first_data.jsonl"
PERCENTAGE=0.05

JOB_NAME="${MODEL}-p${PERCENTAGE}-lora-seed${DATA_SEED}-first"

GRADIENT_PATH=/scratch-shared/ir2-less/grads/${JOB_NAME}/{}-ckpt{}-adam/dim${DIM}

#TODO FIND THE CORRECT CHECKPOINT WEIGHT!

VALIDATION_GRADIENT_PATH=/scratch-shared/ir2-less/grads/${JOB_NAME}/{}-ckpt{}-sgd/dim${DIM}
SELECTED_DATA_OUTPUT_PATH="/scratch-shared/ir2-less/selected_data/${JOB_NAME}/${CHECKPOINTS}"

if [[ ! -d $SELECTED_DATA_OUTPUT_PATH ]]; then
    mkdir -p $SELECTED_DATA_OUTPUT_PATH
fi

bash less/scripts/data_selection/matching.sh "$GRADIENT_PATH" "$TRAIN_FILE_NAMES" "$CKPTS" "$CHECKPOINT_WEIGHTS" "$VALIDATION_GRADIENT_PATH" "$TARGET_TASK_NAMES" "$SELECTED_DATA_OUTPUT_PATH"

python3 -m less.data_selection.write_selected_data --target_task_names ${TARGET_TASK_NAMES} --train_file_names ${TRAIN_FILE_NAMES} --train_files ${TRAIN_FILES} --output_path $SELECTED_DATA_OUTPUT_PATH --percentage ${PERCENTAGE}
