#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#
# Mixtral 8x7B training script using Megatron-Bridge
#
# This script demonstrates distributed training of Mixtral 8x7B with recommended
# parallelism settings for optimal performance.
#
# Usage:
#   bash examples/mixtral/train_mixtral_8x7b.sh <HF_MODEL_PATH> <DATA_PATH> <OUTPUT_PATH>
#
# Example:
#   bash examples/mixtral/train_mixtral_8x7b.sh \
#       "mistralai/Mixtral-8x7B-v0.1" \
#       "/path/to/training/data" \
#       "/path/to/checkpoints"
#

set -e  # Exit on error

# Parse arguments
HF_MODEL_PATH=${1:-"mistralai/Mixtral-8x7B-v0.1"}
DATA_PATH=${2:-"MOCK"}  # Use "MOCK" for synthetic data, or path to real data
OUTPUT_PATH=${3:-"/workspace/checkpoints/mixtral"}

# Distributed training configuration
export CUDA_DEVICE_MAX_CONNECTIONS=1
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-6000}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
WORLD_SIZE=$((GPUS_PER_NODE * NNODES))

echo "========================================================================"
echo "Mixtral 8x7B Training with Megatron-Bridge"
echo "========================================================================"
echo "Model: $HF_MODEL_PATH"
echo "Data: $DATA_PATH"
echo "Output: $OUTPUT_PATH"
echo "Distributed config:"
echo "  - Nodes: $NNODES"
echo "  - GPUs per node: $GPUS_PER_NODE"
echo "  - Total GPUs: $WORLD_SIZE"
echo "  - Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "========================================================================"

# Distributed arguments
DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

# Model parallelism configuration
# Recommended for 8 GPUs: TP1-PP4-EP2 or TP1-PP1-EP8
# For more GPUs, scale up PP and EP accordingly
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
PIPELINE_PARALLEL_SIZE=${PIPELINE_PARALLEL_SIZE:-4}
EXPERT_PARALLEL_SIZE=${EXPERT_PARALLEL_SIZE:-2}
CONTEXT_PARALLEL_SIZE=${CONTEXT_PARALLEL_SIZE:-1}

MODEL_PARALLEL_ARGS=(
    --tensor_model_parallel_size $TENSOR_PARALLEL_SIZE
    --pipeline_model_parallel_size $PIPELINE_PARALLEL_SIZE
    --expert_model_parallel_size $EXPERT_PARALLEL_SIZE
    --context_parallel_size $CONTEXT_PARALLEL_SIZE
)

# Training hyperparameters
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-256}
SEQ_LENGTH=${SEQ_LENGTH:-4096}
TRAIN_ITERS=${TRAIN_ITERS:-10000}
LR=${LR:-1e-4}
MIN_LR=${MIN_LR:-1e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}
CLIP_GRAD=${CLIP_GRAD:-1.0}

TRAINING_ARGS=(
    --micro_batch_size $MICRO_BATCH_SIZE
    --global_batch_size $GLOBAL_BATCH_SIZE
    --seq_length $SEQ_LENGTH
    --train_iters $TRAIN_ITERS
    --lr $LR
    --min_lr $MIN_LR
    --weight_decay $WEIGHT_DECAY
    --clip_grad $CLIP_GRAD
    --adam_beta1 0.9
    --adam_beta2 0.999
    --adam_eps 1e-8
)

# MoE-specific arguments
MOE_AUX_LOSS_COEFF=${MOE_AUX_LOSS_COEFF:-0.01}

MOE_ARGS=(
    --moe_aux_loss_coeff $MOE_AUX_LOSS_COEFF
)

# Optimization flags
OPTIMIZATION_ARGS=(
    --use_distributed_optimizer
    --overlap_grad_reduce
    --bf16
)

# Data arguments - support mock data for testing
DATA_ARGS=()
if [ "$DATA_PATH" = "MOCK" ]; then
    echo "Using MOCK data for testing (no real dataset required)"
    DATA_ARGS+=(--mock_data)
else
    echo "Using real data from: $DATA_PATH"
    DATA_ARGS+=(--data_path $DATA_PATH)
fi

# Logging and checkpointing
LOG_INTERVAL=${LOG_INTERVAL:-10}
SAVE_INTERVAL=${SAVE_INTERVAL:-1000}

LOGGING_ARGS=(
    --log_interval $LOG_INTERVAL
    --save_interval $SAVE_INTERVAL
    --output_path $OUTPUT_PATH
)

# Optional: WandB logging
if [ -n "${WANDB_API_KEY}" ]; then
    echo "WandB logging enabled"
    # Add wandb arguments to training script if needed
fi

# Create output directory
mkdir -p $OUTPUT_PATH

# Print configuration summary
echo ""
echo "Training Configuration:"
echo "  Model Parallelism:"
echo "    - Tensor Parallel: $TENSOR_PARALLEL_SIZE"
echo "    - Pipeline Parallel: $PIPELINE_PARALLEL_SIZE"
echo "    - Expert Parallel: $EXPERT_PARALLEL_SIZE"
echo "    - Context Parallel: $CONTEXT_PARALLEL_SIZE"
echo "  Training:"
echo "    - Micro batch size: $MICRO_BATCH_SIZE"
echo "    - Global batch size: $GLOBAL_BATCH_SIZE"
echo "    - Sequence length: $SEQ_LENGTH"
echo "    - Training iterations: $TRAIN_ITERS"
echo "    - Learning rate: $LR"
echo "    - Min learning rate: $MIN_LR"
echo "  MoE:"
echo "    - Aux loss coefficient: $MOE_AUX_LOSS_COEFF"
echo "========================================================================"
echo ""

# Launch training
torchrun ${DISTRIBUTED_ARGS[@]} examples/mixtral/train_mixtral.py \
    --hf_model_path $HF_MODEL_PATH \
    ${DATA_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${OPTIMIZATION_ARGS[@]} \
    ${LOGGING_ARGS[@]}

echo ""
echo "========================================================================"
echo "Training complete!"
echo "Checkpoints saved to: $OUTPUT_PATH"
echo "========================================================================"
