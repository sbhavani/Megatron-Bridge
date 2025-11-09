# Quick Start with NGC PyTorch 25.09 Container

Streamlined setup using NVIDIA's pre-built PyTorch container.

## Prerequisites

- 8 GPUs (32GB+ VRAM each)
- 128GB+ CPU RAM
- Docker with NVIDIA Container Toolkit installed
- 500GB+ free disk space

## Step 1: Verify Docker and GPU Access

```bash
# Test GPU access
docker run --rm --gpus all nvcr.io/nvidia/pytorch:25.09-py3 nvidia-smi

# Should show all 8 GPUs
```

## Step 2: Clone Repository on Host

```bash
# Clone the repository on your host machine
cd ~
git clone https://github.com/sbhavani/Megatron-Bridge.git
cd Megatron-Bridge

# Verify you have the latest changes
git pull
```

## Step 3: Start NGC Container

```bash
# Create checkpoint directory
mkdir -p ~/megatron_checkpoints

# Start NGC container with Megatron-Bridge mounted
docker run -it --rm \
    --gpus all \
    --shm-size=32g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v ~/Megatron-Bridge:/workspace/Megatron-Bridge \
    -v ~/megatron_checkpoints:/checkpoints \
    -w /workspace/Megatron-Bridge \
    nvcr.io/nvidia/pytorch:25.09-py3 \
    /bin/bash
```

**Important**: We mount the code from host, so you can edit files outside the container and changes are reflected immediately.

## Step 4: Install Megatron-Bridge (Inside Container)

```bash
# Inside the container:

# Install Megatron-Bridge in editable mode
pip install -e .

# Install additional dependencies if needed
pip install transformers datasets accelerate

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
```

## Step 5: Convert Checkpoint

```bash
# Still inside the container:

torchrun --nproc_per_node=8 examples/mixtral/convert_checkpoint.py \
    --hf_model_path=mistralai/Mixtral-8x7B-v0.1 \
    --output_path=/checkpoints/mixtral_tp2_ep4 \
    --tensor_model_parallel_size=2 \
    --expert_model_parallel_size=4
```

**Expected output:**
```
Memory optimization: Using CPU initialization (no GPU allocation)
...
✓ Conversion complete!
Megatron checkpoint saved to: /checkpoints/mixtral_tp2_ep4
Checkpoint format: torch_dist (Megatron Core distributed checkpoint)
```

**Time**: ~3-5 minutes
**GPU Memory**: ~1-2GB per GPU
**CPU RAM**: ~100-120GB

## Step 6: Test LoRA Fine-tuning

```bash
torchrun --nproc_per_node=8 examples/mixtral/finetune_mixtral_lora.py \
    --load=/checkpoints/mixtral_tp2_ep4 \
    --mock_data \
    --output_path=/checkpoints/mixtral_lora_test \
    --tensor_model_parallel_size=2 \
    --expert_model_parallel_size=4 \
    --recompute_activations \
    --micro_batch_size=1 \
    --global_batch_size=8 \
    --train_iters=10 \
    --seq_length=512 \
    --log_interval=2 \
    --lora_rank=16 \
    --lora_alpha=32
```

**Expected output:**
```
Loading checkpoint from: /checkpoints/mixtral_tp2_ep4
Using mock instruction data for testing
LoRA Configuration:
  Rank: 16
  Alpha: 32
...
iteration       10/      10 | ... | lm loss: X.XXX
✓ LoRA Fine-Tuning Complete!
Adapter checkpoints saved to: /checkpoints/mixtral_lora_test
```

**Time**: ~2-3 minutes for 10 iterations
**GPU Memory**: ~27-28GB per GPU

## Step 7: Run Full Test Suite (Optional)

```bash
# Verify everything works
bash examples/mixtral/test_8xa40.sh
```

This runs:
1. Checkpoint conversion (if needed)
2. Pretraining test (10 iterations)
3. LoRA fine-tuning test (10 iterations)
4. Generation test

**Time**: ~5-10 minutes total

---

## Advantages of NGC Container

✅ **No build time**: Pre-built image, ready to use
✅ **Optimized**: NVIDIA-tuned PyTorch with all optimizations
✅ **Latest CUDA**: Supports newest GPU features
✅ **Smaller size**: No unnecessary dependencies
✅ **Regular updates**: New versions released monthly

---

## Working with the Container

### Re-entering the Container

If you exit and need to restart:

```bash
docker run -it --rm \
    --gpus all \
    --shm-size=32g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v ~/Megatron-Bridge:/workspace/Megatron-Bridge \
    -v ~/megatron_checkpoints:/checkpoints \
    -w /workspace/Megatron-Bridge \
    nvcr.io/nvidia/pytorch:25.09-py3 \
    /bin/bash
```

### Running Commands Without Interactive Shell

```bash
# Run conversion without entering container
docker run --rm \
    --gpus all \
    --shm-size=32g \
    -v ~/Megatron-Bridge:/workspace/Megatron-Bridge \
    -v ~/megatron_checkpoints:/checkpoints \
    -w /workspace/Megatron-Bridge \
    nvcr.io/nvidia/pytorch:25.09-py3 \
    bash -c "
        pip install -e . && \
        torchrun --nproc_per_node=8 examples/mixtral/convert_checkpoint.py \
            --hf_model_path=mistralai/Mixtral-8x7B-v0.1 \
            --output_path=/checkpoints/mixtral_tp2_ep4 \
            --tensor_model_parallel_size=2 \
            --expert_model_parallel_size=4
    "
```

### Editing Code on Host

Since we mount the code directory, you can edit files on the host:

```bash
# On host machine (outside container)
cd ~/Megatron-Bridge
vim examples/mixtral/finetune_mixtral_lora.py

# Changes are immediately available inside the container!
```

---

## Troubleshooting

### Issue: "pip install -e ." fails

```bash
# Install build dependencies first
apt-get update
apt-get install -y python3-dev

# Then install Megatron-Bridge
pip install -e .
```

### Issue: HuggingFace token needed

```bash
# Inside container, login to HuggingFace
huggingface-cli login

# Or set token as environment variable
export HF_TOKEN=your_token_here
```

### Issue: Out of shared memory

Restart container with larger `--shm-size`:

```bash
docker run -it --rm \
    --gpus all \
    --shm-size=64g \  # Increased from 32g
    ...
```

### Issue: Container disk space full

```bash
# Check Docker disk usage
docker system df

# Clean up unused images/containers
docker system prune -a
```

---

## Quick Reference

### Start Container
```bash
docker run -it --rm --gpus all --shm-size=32g \
    -v ~/Megatron-Bridge:/workspace/Megatron-Bridge \
    -v ~/megatron_checkpoints:/checkpoints \
    -w /workspace/Megatron-Bridge \
    nvcr.io/nvidia/pytorch:25.09-py3 /bin/bash
```

### Install Megatron-Bridge
```bash
pip install -e .
```

### Set Environment
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
```

### Convert Checkpoint
```bash
torchrun --nproc_per_node=8 examples/mixtral/convert_checkpoint.py \
    --hf_model_path=mistralai/Mixtral-8x7B-v0.1 \
    --output_path=/checkpoints/mixtral_tp2_ep4 \
    --tensor_model_parallel_size=2 \
    --expert_model_parallel_size=4
```

### LoRA Fine-tuning
```bash
torchrun --nproc_per_node=8 examples/mixtral/finetune_mixtral_lora.py \
    --load=/checkpoints/mixtral_tp2_ep4 \
    --mock_data \
    --output_path=/checkpoints/mixtral_lora_test \
    --tensor_model_parallel_size=2 \
    --expert_model_parallel_size=4 \
    --recompute_activations \
    --train_iters=10
```

---

## What's Different from Building Custom Image

| Aspect | Custom Dockerfile | NGC 25.09 Container |
|--------|------------------|---------------------|
| Build time | 10-15 minutes | 0 minutes (pull ~2 min) |
| Size | ~15-20GB | ~8-10GB |
| PyTorch version | May vary | Optimized 2.5+ |
| CUDA version | May vary | Latest CUDA 12.x |
| Updates | Manual rebuild | `docker pull` |
| Compatibility | Need to maintain | NVIDIA guaranteed |

**Recommendation**: Always use NGC containers for production workloads.

---

## Next Steps

1. ✅ Start NGC container
2. ✅ Install Megatron-Bridge with `pip install -e .`
3. ✅ Run checkpoint conversion
4. ✅ Test LoRA fine-tuning
5. 🎯 Train on your real data!

For production training with real data, see the main guide in `REMOTE_SERVER_SETUP.md`.
