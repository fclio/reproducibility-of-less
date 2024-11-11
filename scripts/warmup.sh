#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=LESSWarmup
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=02:00:00
#SBATCH --output=slurm_output/warmup_training_%A.out

# cd $HOME/path/to/LESS
# source activate dl2023

module purge
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

# pip install torchvision torchaudio
# pip install peft==0.7.1
# pip install transformers==4.36.2
# pip install traker[fast]==0.1.3
pip install -r requirement.txt
pip install -e .


# Set variables for warmup training
DATA_DIR="data"
MODEL_PATH="meta-llama/Llama-2-7b-hf"
PERCENTAGE=0.05
DATA_SEED=3
JOB_NAME="llama2-7b-p${PERCENTAGE}-lora-seed${DATA_SEED}"


# Run warmup training
bash less/scripts/train/warmup_lora_train.sh "$DATA_DIR" "$MODEL_PATH" "$PERCENTAGE" "$DATA_SEED" "$JOB_NAME"

# ileNotFoundError: Unable to find '/gpfs/home1/scur2847/ir2-less-data/../data/train/processed/flan_v2/flan_v2_data.jsonl'
