#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=install
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=install_%A.out

# Load necessary modules
module purge
module load 2023
module load CUDA/12.1.1

# Verify CUDA installation
echo "CUDA_HOME: $CUDA_HOME"
which nvcc
nvcc --version

# Define virtual environment path
VENV_DIR="$HOME/venvs/install_env"

# Remove the venv if it exists (optional: uncomment if you want a fresh start)
rm -rf "$VENV_DIR"

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    python -m venv "$VENV_DIR"
fi

# Activate the virtual environment
source "$VENV_DIR/bin/activate"

# Ensure latest pip and build tools
pip install --upgrade pip setuptools wheel

# Explicitly install PyTorch inside the venv
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install required packages
pip install \
    peft==0.7.1 \
    transformers==4.44.2 \
    traker[fast]==0.1.3 \
    datasets \
    torch==2.5.1+cu121 \
    torchvision==0.20.1+cu121 \
    torchaudio==2.5.1+cu121 \
    accelerate==0.34.2 \
    wandb \
    deepspeed 

if [ -f "setup.py" ] || [ -f "pyproject.toml" ]; then
    pip install -e .
else
    echo "Warning: No Python package setup found (setup.py or pyproject.toml missing)."
fi

echo "Installating FlagEmbedding"

cd ..

# Clone and install FlagEmbedding repository
git clone https://github.com/FlagOpen/FlagEmbedding.git
cd FlagEmbedding

# Install with finetune dependency
pip install -e .[finetune]

cd ..

echo "Installation complete. Virtual environment located at $VENV_DIR"

# Ensure necessary package is installed
pip install --upgrade beir

# Check if CUDA is available in PyTorch
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Keep script running for debugging (optional)
# tail -f /dev/null
