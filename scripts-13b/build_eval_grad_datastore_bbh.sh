#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=eval_grad_datastore
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=slurm_output/build_eval_grad_bbh_%A.out

module purge
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

#13b: 211 422 634 844
CKPT=844
PERCENTAGE=0.05
DATA_SEED=4

TASK=bbh
JOB_NAME="llama2-13b-p${PERCENTAGE}-lora-seed${DATA_SEED}"
MODEL_PATH=/scratch-shared/ir2-less/out/${JOB_NAME}/checkpoint-${CKPT}
# OUTPUT_PATH=../grads/llama2-13b-p0.001-lora-seed3/${TASK}-ckpt${CKPT}-sgd
OUTPUT_PATH=/scratch-shared/ir2-less/grads/${JOB_NAME}/${TASK}-ckpt${CKPT}-sgd
DATA_DIR="data"
DIMS="8192"

if [[ ! -d $OUTPUT_PATH ]]; then
    mkdir -p $OUTPUT_PATH
fi

bash less/scripts/get_info/grad/get_eval_lora_grads.sh "$TASK" "$DATA_DIR" "$MODEL_PATH" $OUTPUT_PATH "$DIMS"
