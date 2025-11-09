# Testing Mixtral on 2× A100 80GB

This guide walks you through testing the Mixtral implementation on your 2× A100 80GB setup.

## Quick Start (Automated Testing)

The fastest way to verify everything works:

```bash
# Clone and setup
git clone https://github.com/sbhavani/Megatron-Bridge.git
cd Megatron-Bridge

# Install dependencies
pip install -e .
pip install transformers torch

# Run comprehensive test suite
bash examples/mixtral/test_2xa100.sh
```

This will automatically test all major features in ~10-15 minutes.

## Step-by-Step Manual Testing

### Step 0: Environment Setup

```bash
# Verify GPUs
nvidia-smi

# Expected output: 2× A100 80GB

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Verify PyTorch sees both GPUs
python -c "import torch; print(f'GPUs available: {torch.cuda.device_count()}')"
```

### Step 1: Basic Generation (Pipeline Parallelism)

**Time**: ~3-5 minutes (first run downloads model)
**Memory**: ~40-45GB per GPU

```bash
torchrun --nproc_per_node=2 examples/mixtral/generate_text.py \
    --hf_model_path="mistralai/Mixtral-8x7B-v0.1" \
    --pipeline_model_parallel_size=2 \
    --prompt="What is the capital of France?" \
    --max_tokens=50
```

**Expected output:**
```
Loading Mixtral model from mistralai/Mixtral-8x7B-v0.1...
Model configuration:
  Hidden size: 4096
  Num layers: 32
  Num experts: 8
  Router top-k: 2
  Parallelism: TP=1, PP=2, EP=1

Prompt: What is the capital of France?

Generating...
Generated text:
What is the capital of France? The capital of France is Paris...
```

**What this tests:**
- ✅ Model loading from HuggingFace
- ✅ Pipeline parallelism (splits layers across GPUs)
- ✅ Basic generation
- ✅ Memory efficiency

### Step 2: Expert Parallelism

**Time**: ~2-3 minutes
**Memory**: ~42-47GB per GPU

```bash
torchrun --nproc_per_node=2 examples/mixtral/generate_text.py \
    --hf_model_path="mistralai/Mixtral-8x7B-v0.1" \
    --expert_model_parallel_size=2 \
    --prompt="Explain quantum computing briefly:" \
    --max_tokens=100
```

**What this tests:**
- ✅ Expert parallelism (distributes 8 experts: 4 per GPU)
- ✅ MoE routing
- ✅ Cross-GPU expert communication

**Performance tip:** Expert parallelism is optimal for Mixtral MoE models.

### Step 3: Advanced Sampling

**Time**: ~2 minutes

```bash
torchrun --nproc_per_node=2 examples/mixtral/generate_text.py \
    --hf_model_path="mistralai/Mixtral-8x7B-v0.1" \
    --pipeline_model_parallel_size=2 \
    --prompt="Write a creative story opening:" \
    --temperature=0.8 \
    --top_p=0.95 \
    --top_k=50 \
    --max_tokens=150
```

**What this tests:**
- ✅ Temperature scaling
- ✅ Nucleus (top-p) sampling
- ✅ Top-k sampling
- ✅ Creative generation

### Step 4: Interactive Mode

**Time**: Interactive (try for 5-10 minutes)

```bash
python examples/mixtral/generate_text.py \
    --hf_model_path="mistralai/Mixtral-8x7B-v0.1" \
    --pipeline_model_parallel_size=2 \
    --interactive
```

**Try these commands:**
```
> What is machine learning?
> set temp 0.9
> Write a poem about coding
> set maxtokens 200
> show settings
> quit
```

**What this tests:**
- ✅ Interactive chat interface
- ✅ Runtime configuration changes
- ✅ Multiple generations without reloading

### Step 5: Checkpoint Conversion

**Time**: ~5-10 minutes
**Disk**: ~94GB for output checkpoint

```bash
torchrun --nproc_per_node=2 examples/mixtral/convert_checkpoint.py \
    --hf_model_path="mistralai/Mixtral-8x7B-v0.1" \
    --output_path=/tmp/mixtral_megatron \
    --pipeline_model_parallel_size=2
```

**What this tests:**
- ✅ HF → Megatron conversion
- ✅ Weight mapping correctness
- ✅ Checkpoint saving

**Verify conversion:**
```bash
ls -lh /tmp/mixtral_megatron/
# Should see: mp_rank_00_000/, mp_rank_00_001/
```

### Step 6: Long Sequence Generation

**Time**: ~3-5 minutes
**Memory**: ~45-50GB per GPU (with long context)

```bash
torchrun --nproc_per_node=2 examples/mixtral/generate_text.py \
    --hf_model_path="mistralai/Mixtral-8x7B-v0.1" \
    --expert_model_parallel_size=2 \
    --prompt="Write a detailed explanation of how neural networks work:" \
    --max_tokens=500 \
    --temperature=0.7
```

**What this tests:**
- ✅ Long context handling
- ✅ Memory efficiency with longer sequences
- ✅ Generation quality

### Step 7: Training Test (Optional)

**Time**: ~10-30 minutes (depends on iterations)
**Requirements**: Training data

```bash
# Create dummy data for testing (or use your own)
mkdir -p /tmp/test_data

# Run light training test
torchrun --nproc_per_node=2 examples/mixtral/train_mixtral.py \
    --hf_model_path="mistralai/Mixtral-8x7B-v0.1" \
    --data_path=/tmp/test_data \
    --output_path=/tmp/mixtral_training \
    --pipeline_model_parallel_size=2 \
    --micro_batch_size=1 \
    --global_batch_size=8 \
    --train_iters=10 \
    --seq_length=512  # Shorter for testing
```

**What this tests:**
- ✅ Training loop
- ✅ Gradient computation
- ✅ Optimizer step
- ✅ Checkpoint saving during training

## Monitoring GPU Usage

During testing, monitor GPU memory in another terminal:

```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# Or more detailed
watch -n 1 'nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv'
```

**Expected memory usage:**
- **PP=2**: ~40-45GB per GPU (evenly split)
- **EP=2**: ~42-47GB per GPU (slightly higher due to expert routing)
- **Training**: ~45-55GB per GPU (includes gradients and optimizer states)

## Performance Benchmarks

On 2× A100 80GB, you should see:

| Test | Tokens/sec | Memory/GPU | Notes |
|------|-----------|------------|-------|
| PP=2 Generation | ~15-25 | 40-45GB | Balanced |
| EP=2 Generation | ~20-30 | 42-47GB | Slightly faster |
| Training (BS=1) | ~10-15 | 45-55GB | Depends on seq length |

## Troubleshooting

### Issue: Out of Memory

**Solution 1**: Increase pipeline parallelism
```bash
# This won't work with 2 GPUs, but shows the concept
--pipeline_model_parallel_size=4  # Would need 4 GPUs
```

**Solution 2**: Reduce sequence length
```bash
--max_tokens=100  # Instead of 500
```

**Solution 3**: For training, reduce batch size
```bash
--micro_batch_size=1 --seq_length=1024
```

### Issue: Slow Generation

**Check 1**: Verify both GPUs are being used
```bash
nvidia-smi  # Both should show ~40-45GB used
```

**Check 2**: Try expert parallelism instead
```bash
# Expert parallelism is often faster for MoE
--expert_model_parallel_size=2
```

### Issue: NCCL Errors

**Solution**: Set environment variables
```bash
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0  # Adjust to your network interface
export NCCL_TIMEOUT=1800
```

### Issue: Model Download Slow

**Solution**: Download manually first
```python
from huggingface_hub import snapshot_download
snapshot_download(
    "mistralai/Mixtral-8x7B-v0.1",
    local_dir="./mixtral-hf",
    ignore_patterns=["*.pt"]
)
```

Then use `--hf_model_path=./mixtral-hf`

## Next Steps After Testing

Once all tests pass:

### 1. Try Different Parallelism Strategies
```bash
# Compare performance
bash examples/mixtral/benchmark_parallelism.sh  # If you create this
```

### 2. Fine-tune on Your Data
```bash
bash examples/mixtral/train_mixtral_8x7b.sh \
    "mistralai/Mixtral-8x7B-v0.1" \
    "/path/to/your/data" \
    "/path/to/output"
```

### 3. Export for Production
```bash
# Convert trained model back to HF format
python examples/mixtral/convert_checkpoint.py \
    --megatron_checkpoint=/path/to/checkpoint \
    --output_path=/path/to/hf/output \
    --reverse
```

## Expected Test Results Summary

After running all tests, you should have:

✅ **Generation**: Works with PP=2 and EP=2
✅ **Interactive Mode**: Responds to prompts and commands
✅ **Advanced Sampling**: Creative outputs with temperature/top-p
✅ **Long Sequences**: Handles 500+ tokens
✅ **Checkpoint Conversion**: Creates Megatron checkpoint
✅ **Training** (optional): Loss decreases over iterations

**Total testing time**: ~20-30 minutes
**Disk usage**: ~100GB (model + checkpoints)
**Memory**: 40-50GB per GPU (well within 80GB limit)

## Support

If you encounter issues:
1. Check `examples/mixtral/README.md`
2. Review error messages carefully
3. Open an issue with full error log and GPU info

## Quick Reference

```bash
# Basic generation
torchrun --nproc_per_node=2 examples/mixtral/generate_text.py \
    --hf_model_path="mistralai/Mixtral-8x7B-v0.1" \
    --pipeline_model_parallel_size=2 \
    --prompt="Your prompt here"

# Interactive mode
python examples/mixtral/generate_text.py \
    --hf_model_path="mistralai/Mixtral-8x7B-v0.1" \
    --pipeline_model_parallel_size=2 \
    --interactive

# Training
bash examples/mixtral/train_mixtral_8x7b.sh \
    "mistralai/Mixtral-8x7B-v0.1" \
    "/data/path" \
    "/output/path"
```
