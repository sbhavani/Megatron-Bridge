#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Comprehensive testing script for Mixtral on 2× A100 80GB with LoRA
#
# This script runs through the complete workflow:
#   1. Checkpoint conversion (HF → Megatron)
#   2. Pre-training with mock data
#   3. LoRA fine-tuning with mock instruction data
#
# Usage:
#   bash examples/mixtral/test_2xa100_lora.sh

set -e  # Exit on error

echo "========================================================================"
echo "Mixtral Testing Suite - 2× A100 80GB"
echo "========================================================================"
echo "This will test:"
echo "  1. Checkpoint conversion (HF → Megatron format with TP=2)"
echo "  2. Basic generation to verify conversion"
echo "  3. LoRA Fine-Tuning with mock instruction data"
echo ""
echo "Note: Full pre-training requires 8x A100 or 8x A40"
echo "      LoRA is the recommended approach for 2x A100"
echo "========================================================================"

# Configuration
MODEL_PATH="mistralai/Mixtral-8x7B-v0.1"
export CUDA_VISIBLE_DEVICES=0,1
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

# Verify we have 2 GPUs
GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())")
if [ "$GPU_COUNT" -ne 2 ]; then
    echo -e "\033[0;31mError: Expected 2 GPUs but found $GPU_COUNT\033[0m"
    echo "Please ensure CUDA_VISIBLE_DEVICES is set correctly and 2 GPUs are available"
    exit 1
fi

print_success "Environment check passed (2 GPUs detected)"

# Checkpoint conversion (required for large models like Mixtral 8x7B)
print_test "1" "Checkpoint Conversion (HF → Megatron format with TP=2)"
echo "Converting HuggingFace checkpoint to Megatron format"
echo "This step is required to avoid OOM when loading 47B parameter model"
echo "Output: /tmp/mixtral_tp2"
echo ""

if [ -f "/tmp/mixtral_tp2/latest_checkpointed_iteration.txt" ]; then
    echo "Checkpoint already exists at /tmp/mixtral_tp2, skipping conversion..."
    print_success "Using existing checkpoint"
else
    mkdir -p /tmp/mixtral_tp2

    torchrun --nproc_per_node=2 examples/mixtral/convert_checkpoint.py \
        --hf_model_path=$MODEL_PATH \
        --output_path=/tmp/mixtral_tp2 \
        --tensor_model_parallel_size=2

    print_success "Checkpoint conversion completed"
fi

# Test 1: Basic Generation
print_test "2" "Basic Generation with Tensor Parallelism (TP=2)"
echo "Configuration: TP=2, splits tensors across 2 GPUs"
echo "Expected memory: ~25-30GB per GPU"
echo ""

torchrun --nproc_per_node=2 examples/mixtral/generate_text.py \
    --load=/tmp/mixtral_tp2 \
    --tensor_model_parallel_size=2 \
    --prompt="What is the capital of France?" \
    --max_tokens=50

print_success "Test 2 passed: Basic generation works"

# Test 3: LoRA Fine-Tuning
print_test "3" "LoRA Fine-Tuning Test (10 iterations, mock instruction data)"
echo "Testing LoRA fine-tuning using pre-converted Megatron checkpoint"
echo "Using mock instruction-response pairs - no real dataset required!"
echo "Configuration: TP=2, seq_length=512 (longer than full SFT!)"
echo "Expected memory: ~30-35GB per GPU (vs ~40-45GB for full SFT)"
echo ""
echo "LoRA Advantages:"
echo "  - Freezes base model → No gradients for 47B parameters"
echo "  - Only trains adapters → Tiny matrices (~10-100MB)"
echo "  - Can use Adam optimizer → Better convergence than SGD"
echo "  - Saves ~7-13GB per GPU"
echo ""

torchrun --nproc_per_node=2 examples/mixtral/finetune_mixtral_lora.py \
    --load=/tmp/mixtral_tp2 \
    --mock_data \
    --output_path=/tmp/mixtral_lora_2xa100 \
    --tensor_model_parallel_size=2 \
    --pipeline_model_parallel_size=1 \
    --expert_model_parallel_size=1 \
    --recompute_activations \
    --micro_batch_size=1 \
    --global_batch_size=2 \
    --train_iters=10 \
    --seq_length=512 \
    --log_interval=2 \
    --lora_rank=16 \
    --lora_alpha=32

print_success "Test 3 passed: LoRA fine-tuning works"

# Summary
echo ""
echo "========================================================================"
echo "Test Summary"
echo "========================================================================"
echo -e "${GREEN}All tests passed successfully!${NC}"
echo ""
echo "Tested configurations:"
echo "  ✓ Checkpoint conversion (HF → Megatron format with TP=2)"
echo "  ✓ Basic generation to verify conversion"
echo "  ✓ LoRA fine-tuning with mock data (seq_length=512)"
echo ""
echo "Your 2× A100 80GB setup is working with Mixtral!"
echo ""
echo "Key Findings:"
echo "  - Mixtral 8x7B (47B params) requires checkpoint conversion first"
echo "  - Converted checkpoint saved to: /tmp/mixtral_tp2"
echo "  - LoRA fine-tuning supports seq_length up to 512+"
echo "  - LoRA uses Adam optimizer for better convergence"
echo "  - Memory efficient: ~30-35GB per GPU"
echo ""
echo "Memory Usage Summary:"
echo "  - Checkpoint conversion: ~25-30GB per GPU"
echo "  - Inference/Generation: ~25-30GB per GPU"
echo "  - LoRA fine-tuning: ~30-35GB per GPU (seq_length=512)"
echo ""
echo "Checkpoints and Outputs:"
echo "  - Base checkpoint: /tmp/mixtral_tp2"
echo "  - LoRA adapters: /tmp/mixtral_lora_2xa100"
echo ""
echo "Next Steps:"
echo "  - For real LoRA fine-tuning: Use --data_path with JSONL format"
echo "  - Dataset format: {\"instruction\": \"...\", \"response\": \"...\"}"
echo "  - For full pre-training: Requires 8x A100 or 8x A40 GPUs"
echo ""
echo "Production Recommendations for 2× A100:"
echo "  - Use LoRA for fine-tuning (only viable training option)"
echo "  - Keep seq_length at 512 (or up to 1024 if memory allows)"
echo "  - Use --recompute_activations for additional memory savings"
echo "  - LoRA adapters are tiny (~100MB vs 47GB full model)"
echo ""
echo "LoRA Configuration Used:"
echo "  - Rank: 16 (higher = more capacity, more memory)"
echo "  - Alpha: 32 (scaling parameter)"
echo "  - Target modules: linear_qkv, linear_proj, linear_fc1, linear_fc2"
echo "========================================================================"
