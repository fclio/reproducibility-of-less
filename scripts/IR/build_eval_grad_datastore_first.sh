#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=34_eval_grad_datastore
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=slurm_output/build_eval_grad_first_%A_on_msmarco_model.out

module purge
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

#7b: 422 845 1268 1688 
#13b: 211 422 634 844

CKPT=34
PERCENTAGE=0.001
DATA_SEED=4
TRAINING_DATA_NAME=first
TRAINING_DATA_FILE=data/train/processed/first/first_data.jsonl # when changing data name, change the data path accordingly
GRADIENT_TYPE="sgd"
JOB_NAME="llama2-7b-p${PERCENTAGE}-lora-seed${DATA_SEED}-msmarco"
MODEL_PATH=/scratch-shared/ir2-less/out/${JOB_NAME}/checkpoint-${CKPT}
OUTPUT_PATH=/scratch-shared/ir2-less/grads/${JOB_NAME}/${TRAINING_DATA_NAME}-ckpt${CKPT}-${GRADIENT_TYPE}
DIMS="8192"

bash less/scripts/get_info/grad/get_train_lora_grads.sh "$TRAINING_DATA_FILE" "$MODEL_PATH" "$OUTPUT_PATH" "$DIMS" "$GRADIENT_TYPE"