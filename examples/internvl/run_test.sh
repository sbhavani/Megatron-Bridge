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

# Kill any existing Python processes
pkill -9 python || true

# Get the directory where this script is located
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

# Set Python path to include Megatron-Bridge
export PYTHONPATH="${SCRIPT_DIR}/../..:${PYTHONPATH}"

# CUDA settings
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Network interface (adjust as needed)
export NCCL_SOCKET_IFNAME=eth0

# Distributed training settings
export GPUS_PER_NODE=${GPUS_PER_NODE:-1}
export TP_SIZE=${TP_SIZE:-1}
export PP_SIZE=${PP_SIZE:-1}
export CP_SIZE=${CP_SIZE:-1}

# Calculate world size
export MASTER_ADDR=localhost
export MASTER_PORT=65535
export NNODES=${NNODES:-1}
export NODE_RANK=${OMPI_COMM_WORLD_RANK:-0}
export WORLD_SIZE=$((GPUS_PER_NODE * NNODES))

# Model path (adjust to your model location)
MODEL_PATH=${MODEL_PATH:-"OpenGVLab/InternVL3-1B"}

echo "========================================"
echo "Testing InternVL with Megatron-Bridge"
echo "========================================"
echo "Model: ${MODEL_PATH}"
echo "GPUs per node: ${GPUS_PER_NODE}"
echo "Tensor Parallel Size: ${TP_SIZE}"
echo "Pipeline Parallel Size: ${PP_SIZE}"
echo "Context Parallel Size: ${CP_SIZE}"
echo "========================================"

# Run inference test
echo ""
echo "Running inference with Megatron Bridge"
echo "------------------------------------"
torchrun \
    --nproc_per_node=${GPUS_PER_NODE} \
    --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    "${SCRIPT_DIR}/load_model_and_inference.py" \
    "${MODEL_PATH}" \
    --tp_size ${TP_SIZE} \
    --pp_size ${PP_SIZE} \
    --cp_size ${CP_SIZE} \
    --max_new_tokens 50 \
    --prompt "What is the capital of France?"

echo ""
echo "========================================"
echo "All tests completed!"
echo "========================================"
