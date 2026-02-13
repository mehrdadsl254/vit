#!/bin/bash
# Setup script for Qwen2.5-VL activation steering experiment
# Creates a new conda env with compatible PyTorch and transformers

set -e

ENV_NAME="qwen-vl"

echo "=== Creating conda environment: $ENV_NAME ==="
conda create -n $ENV_NAME python=3.10 -y

echo "=== Activating environment ==="
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

echo "=== Installing PyTorch (CUDA 11.8) ==="
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

echo "=== Installing transformers and dependencies ==="
pip install transformers accelerate
pip install qwen_vl_utils
pip install Pillow numpy scipy

echo ""
echo "=== Setup complete! ==="
echo "Activate with: conda activate $ENV_NAME"
echo "Then run:"
echo "  python src/experiment_activation_steering.py \\"
echo "    --image_path ./image.png \\"
echo "    --model_path /home/mmd/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct \\"
echo "    --layers 14 \\"
echo "    --alphas 1.0 5.0 10.0 20.0"
