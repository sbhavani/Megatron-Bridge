#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Docker launcher for Mixtral testing on 2× A100 80GB
#
# This script launches the NVIDIA PyTorch container with proper GPU access
# and volume mounts for testing Mixtral with Megatron-Bridge.
#
# Usage:
#   bash examples/mixtral/run_docker.sh [command]
#
# Examples:
#   # Interactive shell
#   bash examples/mixtral/run_docker.sh
#
#   # Run automated tests
#   bash examples/mixtral/run_docker.sh bash examples/mixtral/test_2xa100.sh
#
#   # Run training with mock data
#   bash examples/mixtral/run_docker.sh torchrun --nproc_per_node=2 \
#       examples/mixtral/train_mixtral.py --mock_data

set -e

# Configuration
DOCKER_IMAGE="nvcr.io/nvidia/pytorch:25.10-py3"
CONTAINER_NAME="mixtral-megatron-bridge"

# Get absolute path to repo root (assumes script is in examples/mixtral/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Workspace inside container
WORKSPACE="/workspace/Megatron-Bridge"

# Cache directories (optional, for faster model downloads)
HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"
TORCH_CACHE_DIR="${TORCH_CACHE_DIR:-$HOME/.cache/torch}"

echo "========================================================================"
echo "Mixtral Docker Environment"
echo "========================================================================"
echo "Image: $DOCKER_IMAGE"
echo "Repository: $REPO_ROOT"
echo "Workspace: $WORKSPACE"
echo "GPUs: All available (filtered by CUDA_VISIBLE_DEVICES if set)"
echo "========================================================================"

# Create cache directories if they don't exist
mkdir -p "$HF_CACHE_DIR"
mkdir -p "$TORCH_CACHE_DIR"

# Check if container already exists
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Removing existing container: $CONTAINER_NAME"
    docker rm -f "$CONTAINER_NAME" > /dev/null 2>&1
fi

# Docker run arguments
DOCKER_ARGS=(
    --gpus all                                    # Enable all GPUs
    --ipc=host                                    # Shared memory for multi-GPU
    --ulimit memlock=-1                           # Unlimited locked memory
    --ulimit stack=67108864                       # Stack size for NCCL
    --name "$CONTAINER_NAME"                      # Container name
    --rm                                          # Remove on exit
    -v "$REPO_ROOT:$WORKSPACE"                    # Mount repository
    -v "$HF_CACHE_DIR:/root/.cache/huggingface"  # HuggingFace cache
    -v "$TORCH_CACHE_DIR:/root/.cache/torch"     # PyTorch cache
    -w "$WORKSPACE"                               # Working directory
    --shm-size=16g                                # Shared memory size
)

# Environment variables
ENV_VARS=(
    -e CUDA_DEVICE_MAX_CONNECTIONS=1
    -e NCCL_DEBUG=INFO
    -e PYTHONUNBUFFERED=1
)

# Optional: Pass through networking for distributed training
if [ -n "$MASTER_ADDR" ]; then
    ENV_VARS+=(-e MASTER_ADDR="$MASTER_ADDR")
fi
if [ -n "$MASTER_PORT" ]; then
    ENV_VARS+=(-e MASTER_PORT="$MASTER_PORT")
fi
if [ -n "$NODE_RANK" ]; then
    ENV_VARS+=(-e NODE_RANK="$NODE_RANK")
fi
if [ -n "$WORLD_SIZE" ]; then
    ENV_VARS+=(-e WORLD_SIZE="$WORLD_SIZE")
fi

# Optional: WandB API key
if [ -n "$WANDB_API_KEY" ]; then
    ENV_VARS+=(-e WANDB_API_KEY="$WANDB_API_KEY")
fi

# Determine command to run
if [ $# -eq 0 ]; then
    # Interactive mode
    echo ""
    echo "Starting interactive shell..."
    echo "To run tests: bash examples/mixtral/test_2xa100.sh"
    echo ""
    DOCKER_CMD=("/bin/bash")
    INTERACTIVE_FLAGS="-it"
else
    # Command mode
    echo ""
    echo "Running command: $@"
    echo ""
    DOCKER_CMD=("$@")
    INTERACTIVE_FLAGS="-i"
fi

# Pull latest image if needed
echo "Checking for image updates..."
docker pull "$DOCKER_IMAGE"

# Run container
docker run \
    $INTERACTIVE_FLAGS \
    ${DOCKER_ARGS[@]} \
    ${ENV_VARS[@]} \
    "$DOCKER_IMAGE" \
    "${DOCKER_CMD[@]}"
