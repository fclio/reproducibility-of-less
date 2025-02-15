#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=fiqa_eval_ranking
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=slurm_output3/eval_ranking_baseline_fiqa_llama7b_%A.out

# Load necessary modules
module purge
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

TASK_NAME=fiqa
# TASK_NAME=nfcorpus
# TASK_NAME=scifact
# TASK_NAME=vihealthqa


TYPE="bm25"
# TYPE="random"

# Define paths
JOB_NAME="llama2-7b-less-p0.05-lora-seed6-first-${TASK_NAME}"

MODEL_DIR="/scratch-shared/ir2-less/out/${JOB_NAME}-${TYPE}"
DATASET_PATH="/home/scur2847/ir2-less-data/data/eval/${TASK_NAME}/${TASK_NAME}_data.jsonl"
OUTPUT_PATH="/home/scur2847/ir2-less-data/eval/result/${JOB_NAME}-alpha/ranking_results_${TYPE}_${TASK_NAME}.json"

# Run the ranking evaluation script
python eval/ir/ranking_relevant.py --model_path "$MODEL_DIR" --dataset_path "$DATASET_PATH" --output_path "$OUTPUT_PATH"

# Check and print the result if the script generates a summary or result
if [[ -f "$OUTPUT_PATH" ]]; then
    echo "Ranking Evaluation Completed. Results saved to: $OUTPUT_PATH"
else
    echo "Ranking Evaluation Failed. Please check the script and logs."
fi
