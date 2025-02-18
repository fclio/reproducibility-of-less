#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=eval_ranking
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=06:00:00
#SBATCH --output=../slurm_output_IR_eval/eval_ranking_%A.out

# Load necessary modules
module purge
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

# TASK_NAME=fiqa
# TASK_NAME=nfcorpus
# TASK_NAME=scifact
# TASK_NAME=vihealthqa

# # Define paths
# JOB_NAME="llama2-7b-less-p0.05-lora-seed3-first-${TASK_NAME}-shot-15"

JOB_NAME=$1
TASK_NAME=$2
LESS_REPO_DIR=$3
LESS_OUTPUT_DIR=$4

MODEL_DIR="${LESS_OUTPUT_DIR}/out/${JOB_NAME}"
# DATASET_PATH="${LESS_REPO_DIR}/data_old/${TASK_NAME}/${TASK_NAME}_data.jsonl"
DATASET_PATH="${LESS_REPO_DIR}/data/eval/${TASK_NAME}/${TASK_NAME}_data.jsonl"
OUTPUT_PATH="${LESS_REPO_DIR}/eval/result2/${JOB_NAME}/ranking_results_finetune_${TASK_NAME}.json"

# Run the ranking evaluation script
python eval/ir/ranking_new.py --model_path "$MODEL_DIR" --dataset_path "$DATASET_PATH" --output_path "$OUTPUT_PATH"
# python eval/ir/ranking_new.py --model_path "meta-llama/Llama-2-7b-hf" --dataset_path "$DATASET_PATH" --output_path "$OUTPUT_PATH"
# python eval/ir/ranking_new.py --model_path "/scratch-shared/ir2-less/out/llama2-7b-p0.05-lora-seed14-msmarco" --dataset_path "$DATASET_PATH" --output_path "$OUTPUT_PATH"

# Check and print the result if the script generates a summary or result
if [[ -f "$OUTPUT_PATH" ]]; then
    echo "Ranking Evaluation Completed. Results saved to: $OUTPUT_PATH"
else
    echo "Ranking Evaluation Failed. Please check the script and logs."
fi
