#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=eval_grad_datastore
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=slurm_output/build_eval_grad_qa_124_13b_%A.out

module purge
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

#7b: 422 845 1268 1688 
#13b: 211 422 634 844
#7b: 11,22,34,44

#first-0.5: 62 124 187 248
#first-13b-0.5: 31 62 93 124
#msmarco-0.5: 573 1146 1720 2292

CKPT=124
PERCENTAGE=0.05
DATA_SEED=3
TRAINING_DATA_NAME=qa
TRAINING_DATA_FILE=data/eval/qa/qa_data.jsonl # when changing data name, change the data path accordingly
GRADIENT_TYPE="sgd"
JOB_NAME="llama2-13b-p${PERCENTAGE}-lora-seed${DATA_SEED}-first"



MODEL_PATH=/scratch-shared/ir2-less/out/${JOB_NAME}/checkpoint-${CKPT}
OUTPUT_PATH=/scratch-shared/ir2-less/grads/${JOB_NAME}/${TRAINING_DATA_NAME}-ckpt${CKPT}-${GRADIENT_TYPE}
DIMS="8192"

bash less/scripts/get_info/grad/get_train_lora_grads.sh "$TRAINING_DATA_FILE" "$MODEL_PATH" "$OUTPUT_PATH" "$DIMS" "$GRADIENT_TYPE"