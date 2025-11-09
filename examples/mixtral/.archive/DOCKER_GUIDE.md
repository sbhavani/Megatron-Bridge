# Running Mixtral with Docker on 2× A100 80GB

This guide shows how to run Mixtral examples using the NVIDIA PyTorch Docker container.

## Prerequisites

1. **Docker with GPU support**:
   ```bash
   # Install Docker
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh

   # Install NVIDIA Container Toolkit
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
       sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

2. **Verify GPU access in Docker**:
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
   ```

## Quick Start

### Method 1: Using the Helper Script (Recommended)

```bash
# Interactive shell (starts container, drops you into bash)
bash examples/mixtral/run_docker.sh

# Inside container, run tests
bash examples/mixtral/test_2xa100.sh
```

### Method 2: Using Docker Compose

```bash
# Start container
cd examples/mixtral
docker-compose up -d

# Execute commands inside
docker-compose exec mixtral bash examples/mixtral/test_2xa100.sh

# Stop container
docker-compose down
```

### Method 3: Direct Docker Run

```bash
docker run --gpus all -it --rm \
    --ipc=host \
    --shm-size=16g \
    -v $(pwd):/workspace/Megatron-Bridge \
    -w /workspace/Megatron-Bridge \
    nvcr.io/nvidia/pytorch:25.10-py3 \
    bash
```

## Detailed Usage

### 1. Interactive Development

```bash
# Start interactive container
bash examples/mixtral/run_docker.sh

# Inside container:
# Install package in development mode
pip install -e .

# Run any test
bash examples/mixtral/test_2xa100.sh

# Or individual tests
torchrun --nproc_per_node=2 examples/mixtral/generate_text.py \
    --hf_model_path="mistralai/Mixtral-8x7B-v0.1" \
    --pipeline_model_parallel_size=2 \
    --prompt="Hello, world!"
```

### 2. Run Specific Commands

```bash
# Run automated test suite
bash examples/mixtral/run_docker.sh \
    bash examples/mixtral/test_2xa100.sh

# Run training with mock data
bash examples/mixtral/run_docker.sh \
    torchrun --nproc_per_node=2 examples/mixtral/train_mixtral.py \
    --hf_model_path="mistralai/Mixtral-8x7B-v0.1" \
    --mock_data \
    --pipeline_model_parallel_size=2 \
    --train_iters=10

# Run interactive generation
bash examples/mixtral/run_docker.sh \
    python examples/mixtral/generate_text.py \
    --hf_model_path="mistralai/Mixtral-8x7B-v0.1" \
    --pipeline_model_parallel_size=2 \
    --interactive
```

### 3. Docker Compose for Persistent Development

```bash
cd examples/mixtral

# Start container in background
docker-compose up -d

# Check it's running
docker-compose ps

# Execute commands
docker-compose exec mixtral bash examples/mixtral/test_2xa100.sh

# View logs
docker-compose logs -f

# Shell access
docker-compose exec mixtral bash

# Stop when done
docker-compose down
```

## Volume Mounts

The Docker setup mounts these directories:

| Host | Container | Purpose |
|------|-----------|---------|
| `Megatron-Bridge/` | `/workspace/Megatron-Bridge` | Code repository |
| `~/.cache/huggingface` | `/root/.cache/huggingface` | HuggingFace models cache |
| `~/.cache/torch` | `/root/.cache/torch` | PyTorch cache |
| `examples/mixtral/checkpoints` | `/workspace/checkpoints` | Training checkpoints |

**Benefits:**
- ✅ Model downloads are cached on host
- ✅ Code changes reflect immediately
- ✅ Checkpoints persist after container stops

## Environment Variables

### Passed Automatically

```bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=INFO
export PYTHONUNBUFFERED=1
```

### Optional (set before running)

```bash
# Distributed training
export MASTER_ADDR=10.0.0.1
export MASTER_PORT=6000
export NODE_RANK=0
export WORLD_SIZE=4

# WandB logging
export WANDB_API_KEY=your_key_here

# Then run
bash examples/mixtral/run_docker.sh
```

## Common Workflows

### Workflow 1: Quick Test

```bash
# One-liner to test everything
bash examples/mixtral/run_docker.sh bash examples/mixtral/test_2xa100.sh
```

### Workflow 2: Development Cycle

```bash
# Start persistent container
cd examples/mixtral
docker-compose up -d

# Edit code on host (your IDE)
vim ../../src/megatron/bridge/models/mixtral/mixtral_provider.py

# Test inside container
docker-compose exec mixtral python -c "from megatron.bridge.models import MixtralModelProvider; print('Import works!')"

# Run full test
docker-compose exec mixtral bash examples/mixtral/test_2xa100.sh

# Stop when done
docker-compose down
```

### Workflow 3: Training Run

```bash
# Create checkpoints directory
mkdir -p examples/mixtral/checkpoints

# Start container with volume mount
docker-compose up -d

# Run training
docker-compose exec mixtral bash examples/mixtral/train_mixtral_8x7b.sh \
    "mistralai/Mixtral-8x7B-v0.1" \
    "MOCK" \
    "/workspace/checkpoints/mixtral_run1"

# Checkpoints persist in examples/mixtral/checkpoints/ on host
```

### Workflow 4: Multi-Node Training

On each node:

```bash
# Node 0 (master)
export MASTER_ADDR=10.0.0.1
export MASTER_PORT=6000
export NODE_RANK=0
export WORLD_SIZE=2

bash examples/mixtral/run_docker.sh \
    bash examples/mixtral/train_mixtral_8x7b.sh

# Node 1 (worker)
export MASTER_ADDR=10.0.0.1
export MASTER_PORT=6000
export NODE_RANK=1
export WORLD_SIZE=2

bash examples/mixtral/run_docker.sh \
    bash examples/mixtral/train_mixtral_8x7b.sh
```

## GPU Configuration

### Use Specific GPUs

```bash
# Use only GPUs 0 and 1
CUDA_VISIBLE_DEVICES=0,1 bash examples/mixtral/run_docker.sh
```

### Check GPU Usage

```bash
# From host
watch nvidia-smi

# Inside container
docker-compose exec mixtral nvidia-smi
```

## Troubleshooting

### Issue: "docker: Error response from daemon: could not select device driver"

**Solution**: Install NVIDIA Container Toolkit
```bash
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Issue: "Failed to initialize NVML"

**Solution**: Ensure GPU is accessible
```bash
# Test GPU access
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

### Issue: Out of shared memory

**Solution**: Increase `--shm-size`
```bash
# In run_docker.sh, increase:
--shm-size=32g  # Instead of 16g
```

Or in `docker-compose.yml`:
```yaml
shm_size: 32gb
```

### Issue: Permission denied on volume mounts

**Solution**: Run with user mapping
```bash
docker run --gpus all -it --rm \
    --user $(id -u):$(id -g) \
    -v $(pwd):/workspace \
    ...
```

### Issue: Slow model downloads

**Solution**: Cache is working if you see:
```
~/.cache/huggingface mounted to /root/.cache/huggingface
```

Check cache on host:
```bash
ls -lh ~/.cache/huggingface/hub/
```

## Performance Optimization

### 1. Use Local SSD for Cache

```bash
# Create cache on fast storage
export HF_CACHE_DIR=/fast/ssd/huggingface
export TORCH_CACHE_DIR=/fast/ssd/torch

bash examples/mixtral/run_docker.sh
```

### 2. Persistent Container

Keep container running to avoid startup overhead:
```bash
docker-compose up -d
# Run multiple tests without restarting
docker-compose exec mixtral bash examples/mixtral/test_2xa100.sh
docker-compose exec mixtral python examples/mixtral/generate_text.py ...
```

### 3. Pre-pull Image

```bash
# Download image in advance (saves time later)
docker pull nvcr.io/nvidia/pytorch:25.10-py3
```

## Resource Monitoring

### Monitor from Host

```bash
# GPU usage
watch -n 1 nvidia-smi

# Container stats
docker stats mixtral-megatron-bridge

# Detailed GPU info
nvidia-smi dmon -s ucm
```

### Monitor Inside Container

```bash
docker-compose exec mixtral bash

# Inside container
nvidia-smi
htop
```

## Cleanup

### Remove Container

```bash
# If using docker-compose
docker-compose down

# If using run_docker.sh (automatic with --rm flag)
# Container is removed on exit

# Manual cleanup
docker rm -f mixtral-megatron-bridge
```

### Remove Image

```bash
docker rmi nvcr.io/nvidia/pytorch:25.10-py3
```

### Clean Cache

```bash
# Remove downloaded models (careful!)
rm -rf ~/.cache/huggingface/hub/*

# Remove PyTorch cache
rm -rf ~/.cache/torch/*
```

## Complete Example: End-to-End Test

```bash
# 1. Start from repo root
cd Megatron-Bridge

# 2. Run automated test suite in Docker
bash examples/mixtral/run_docker.sh bash examples/mixtral/test_2xa100.sh

# Expected output:
# ========================================================================
# TEST 1: Basic Generation - Pipeline Parallelism (PP=2)
# ========================================================================
# ...
# ✓ All core tests passed!
```

## Advanced: Custom Docker Image

If you need additional dependencies:

```dockerfile
# Dockerfile.mixtral
FROM nvcr.io/nvidia/pytorch:25.10-py3

# Install additional packages
RUN pip install wandb flash-attn

# Copy code
COPY . /workspace/Megatron-Bridge
WORKDIR /workspace/Megatron-Bridge

# Install package
RUN pip install -e .

CMD ["/bin/bash"]
```

Build and run:
```bash
docker build -f Dockerfile.mixtral -t mixtral-custom .
docker run --gpus all -it mixtral-custom
```

## Reference

### Helper Script Flags

```bash
bash examples/mixtral/run_docker.sh [command]

# Examples:
bash examples/mixtral/run_docker.sh  # Interactive
bash examples/mixtral/run_docker.sh bash test.sh  # Run script
bash examples/mixtral/run_docker.sh python train.py  # Run Python
```

### Docker Compose Commands

```bash
docker-compose up -d          # Start in background
docker-compose ps             # Show status
docker-compose logs -f        # Follow logs
docker-compose exec mixtral bash  # Shell access
docker-compose down           # Stop and remove
docker-compose restart        # Restart
```

## Quick Reference Card

| Task | Command |
|------|---------|
| **Run all tests** | `bash examples/mixtral/run_docker.sh bash examples/mixtral/test_2xa100.sh` |
| **Interactive shell** | `bash examples/mixtral/run_docker.sh` |
| **Training (mock)** | `bash examples/mixtral/run_docker.sh bash examples/mixtral/train_mixtral_8x7b.sh` |
| **Start persistent** | `docker-compose up -d` |
| **Stop persistent** | `docker-compose down` |
| **View GPUs** | `docker-compose exec mixtral nvidia-smi` |
| **Shell access** | `docker-compose exec mixtral bash` |
