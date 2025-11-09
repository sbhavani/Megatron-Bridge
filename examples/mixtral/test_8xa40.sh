#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Pre-training test script for Mixtral on 8× A40 45GB
#
# This script tests full pre-training capability that requires 8+ GPUs.
# Uses vanilla SGD (momentum=0.0) to fit in A40 45GB memory.
# For LoRA fine-tuning on limited hardware, see test_2xa100_lora.sh
#
# Usage:
#   bash examples/mixtral/test_8xa40.sh

set -e  # Exit on error

echo "========================================================================"
echo "Mixtral Pre-training Test - 8× A40 45GB"
echo "========================================================================"
echo "This will test:"
echo "  1. Checkpoint conversion (HF → Megatron format)"
echo "  2. Full pre-training with mock data"
echo "  3. Basic generation to verify"
echo ""
echo "Configuration: TP=2, EP=4, vanilla SGD (momentum=0.0)"
echo "Expected memory: ~43-44GB per GPU during training"
echo "========================================================================"

# Configuration
MODEL_PATH="mistralai/Mixtral-8x7B-v0.1"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print test headers
print_test() {
    echo ""
    echo -e "${BLUE}========================================================================"
    echo "TEST $1: $2"
    echo -e "========================================================================${NC}"
}

# Function to print success
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Function to print warning
print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Check GPU availability
print_test "0" "Environment Check"
echo "Checking GPU availability..."
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

# Verify we have 8 GPUs
GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())")
if [ "$GPU_COUNT" -ne 8 ]; then
    echo -e "\033[0;31mError: Expected 8 GPUs but found $GPU_COUNT\033[0m"
    echo "Please ensure CUDA_VISIBLE_DEVICES is set correctly and 8 GPUs are available"
    exit 1
fi

# Check GPU memory
GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "$GPU_MEMORY" -lt 45000 ]; then
    print_warning "Detected ${GPU_MEMORY}MB per GPU - this may be insufficient for training"
    print_warning "A40 (48GB) or larger recommended. Continuing anyway..."
fi

print_success "Environment check passed (8 GPUs detected)"

# Checkpoint conversion
print_test "1" "Checkpoint Conversion (HF → Megatron with TP=2, EP=4)"
echo "Converting HuggingFace checkpoint to Megatron format"
echo "Output: /tmp/mixtral_tp2_ep4"
echo "Configuration: TP=2, EP=4"
echo ""

if [ -f "/tmp/mixtral_tp2_ep4/latest_checkpointed_iteration.txt" ]; then
    echo "Checkpoint already exists at /tmp/mixtral_tp2_ep4, skipping conversion..."
    print_success "Using existing checkpoint"
else
    mkdir -p /tmp/mixtral_tp2_ep4

    torchrun --nproc_per_node=8 examples/mixtral/convert_checkpoint.py \
        --hf_model_path=$MODEL_PATH \
        --output_path=/tmp/mixtral_tp2_ep4 \
        --tensor_model_parallel_size=2 \
        --expert_model_parallel_size=4

    print_success "Checkpoint conversion completed"
fi

# Test: Pre-training
print_test "2" "Pre-training Test (10 iterations, TP=2, EP=4)"
echo "Testing full pre-training with mock data"
echo "Configuration: TP=2, EP=4 with vanilla SGD (momentum=0.0)"
echo "Expected memory: ~43-44GB per GPU (fits in A40 45GB)"
echo ""

torchrun --nproc_per_node=8 examples/mixtral/train_mixtral.py \
    --load=/tmp/mixtral_tp2_ep4 \
    --mock_data \
    --output_path=/tmp/mixtral_pretrain_8xa40 \
    --tensor_model_parallel_size=2 \
    --pipeline_model_parallel_size=1 \
    --expert_model_parallel_size=4 \
    --optimizer=sgd \
    --momentum=0.0 \
    --recompute_activations \
    --micro_batch_size=1 \
    --global_batch_size=8 \
    --train_iters=10 \
    --seq_length=128 \
    --log_interval=2

print_success "Test 2 passed: Pre-training works on 8× A40"

# Test: Generation
print_test "3" "Basic Generation (TP=2, EP=4)"
echo "Configuration: TP=2, EP=4"
echo "Expected memory: ~8-12GB per GPU"
echo ""

torchrun --nproc_per_node=8 examples/mixtral/generate_text.py \
    --load=/tmp/mixtral_tp2_ep4 \
    --tensor_model_parallel_size=2 \
    --expert_model_parallel_size=4 \
    --prompt="What is the capital of France?" \
    --max_tokens=50

print_success "Test 3 passed: Generation works"

# Summary
echo ""
echo "========================================================================"
echo "Test Summary"
echo "========================================================================"
echo -e "${GREEN}All tests passed successfully!${NC}"
echo ""
echo "Tested configurations:"
echo "  ✓ Checkpoint conversion (HF → Megatron, TP=2, EP=4)"
echo "  ✓ Full pre-training with vanilla SGD (TP=2, EP=4, seq_length=128)"
echo "  ✓ Basic generation verification"
echo ""
echo "Your 8× A40 45GB setup is working with Mixtral pre-training!"
echo ""
echo "Configuration Details:"
echo "  - Tensor Parallelism (TP=2): Splits model tensors across 2 GPUs"
echo "  - Expert Parallelism (EP=4): Splits 8 experts across 4 GPUs"
echo "  - Vanilla SGD (momentum=0.0): No momentum buffers to save memory"
echo "  - Activation checkpointing enabled for memory efficiency"
echo "  - Converted checkpoint: /tmp/mixtral_tp2_ep4"
echo ""
echo "Memory Usage:"
echo "  - Training: ~43-44GB per GPU (fits tightly in A40 45GB)"
echo "  - Inference: ~8-12GB per GPU"
echo ""
echo "Next Steps:"
echo "  - For real pre-training: Use --data_path instead of --mock_data"
echo "  - For LoRA fine-tuning: See test_2xa100_lora.sh"
echo "  - Consider seq_length=256 for production (may require tuning)"
echo ""
echo "Production Recommendations:"
echo "  - Use TP=2, EP=4 configuration"
echo "  - Use vanilla SGD with momentum=0.0 (required for A40 45GB)"
echo "  - Keep --recompute_activations enabled"
echo "  - seq_length=128 fits safely, 256 may work with tuning"
echo "  - Use gradient accumulation for larger effective batch sizes"
echo "========================================================================"
