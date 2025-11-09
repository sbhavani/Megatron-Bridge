# Remote Server Setup for Mixtral 8x7B with LoRA Fine-tuning

Complete guide to set up and run Mixtral 8x7B checkpoint conversion and LoRA fine-tuning on a remote server with 8 GPUs.

## Prerequisites

### Hardware Requirements
- **8 GPUs** (minimum 32GB VRAM each, recommended: A40 48GB or A100 40GB/80GB)
- **128GB+ CPU RAM** (critical for checkpoint conversion)
- **500GB+ free disk space** for models and checkpoints
- **Fast internet** for downloading Mixtral model (~90GB)

### Software Requirements
- Ubuntu 20.04+ or similar Linux distribution
- NVIDIA Driver 525+ (for CUDA 12.x)
- Docker with NVIDIA Container Toolkit
- Git

---

## Step 1: Install Docker and NVIDIA Container Toolkit

### 1.1 Install Docker

```bash
# Update package list
sudo apt-get update

# Install dependencies
sudo apt-get install -y ca-certificates curl gnupg lsb-release

# Add Docker's official GPG key
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Set up Docker repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Add your user to docker group (to run without sudo)
sudo usermod -aG docker $USER

# Log out and back in for group changes to take effect
# Or run: newgrp docker
```

### 1.2 Install NVIDIA Container Toolkit

```bash
# Add NVIDIA Container Toolkit repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install NVIDIA Container Toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker to use NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker

# Restart Docker
sudo systemctl restart docker
```

### 1.3 Verify GPU Access in Docker

```bash
# Test GPU access
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# You should see all 8 GPUs listed
```

---

## Step 2: Clone and Build Megatron-Bridge

### 2.1 Clone Repository

```bash
# Clone the repository
git clone https://github.com/sbhavani/Megatron-Bridge.git
cd Megatron-Bridge

# Verify you're on the correct branch
git branch
# Should show: * main
```

### 2.2 Build Docker Image

The repository includes a Dockerfile. Build it:

```bash
# Build the Docker image (this takes ~10-15 minutes)
docker build -t megatron-bridge:latest .

# If there's no Dockerfile, create one (see Option B below)
```

#### Option B: Create Dockerfile if not present

If the repo doesn't have a Dockerfile, create one:

```bash
cat > Dockerfile << 'EOF'
FROM nvcr.io/nvidia/pytorch:24.01-py3

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy Megatron-Bridge code
COPY . /workspace/Megatron-Bridge

# Install Megatron-Bridge
WORKDIR /workspace/Megatron-Bridge
RUN pip install --upgrade pip
RUN pip install -e .

# Install additional dependencies
RUN pip install transformers datasets accelerate

# Set environment variables
ENV PYTHONPATH=/workspace/Megatron-Bridge:$PYTHONPATH
ENV CUDA_DEVICE_MAX_CONNECTIONS=1
ENV PYTORCH_ALLOC_CONF=expandable_segments:True

WORKDIR /workspace/Megatron-Bridge
EOF

# Build the image
docker build -t megatron-bridge:latest .
```

---

## Step 3: Run Checkpoint Conversion

### 3.1 Start Docker Container

```bash
# Create a directory for checkpoints on host (persistent storage)
mkdir -p ~/megatron_checkpoints

# Start container with GPU access and mounted checkpoint directory
docker run -it --rm \
    --gpus all \
    --shm-size=32g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v ~/megatron_checkpoints:/checkpoints \
    -w /workspace/Megatron-Bridge \
    megatron-bridge:latest \
    /bin/bash
```

**Important flags:**
- `--gpus all`: Access to all GPUs
- `--shm-size=32g`: Increase shared memory (needed for PyTorch DataLoader)
- `-v ~/megatron_checkpoints:/checkpoints`: Mount host directory for persistent storage
- `--ulimit memlock=-1`: Remove memory lock limits

### 3.2 Run Conversion Inside Container

```bash
# Inside the container:

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Run checkpoint conversion (uses CPU initialization - minimal GPU memory!)
torchrun --nproc_per_node=8 examples/mixtral/convert_checkpoint.py \
    --hf_model_path=mistralai/Mixtral-8x7B-v0.1 \
    --output_path=/checkpoints/mixtral_tp2_ep4 \
    --tensor_model_parallel_size=2 \
    --expert_model_parallel_size=4
```

**Expected output:**
```
Memory optimization: Using CPU initialization (no GPU allocation)
Step 1: Loading HuggingFace model...
Step 2: Creating Megatron provider...
Step 3: Finalizing provider and initializing model parallelism...
Step 4: Creating distributed model on CPU (avoiding GPU allocation)...
Step 5: Saving Megatron checkpoint to /checkpoints/mixtral_tp2_ep4...
✓ Conversion complete!
```

**Time estimate:** 3-5 minutes (depends on download speed and disk I/O)

**Resource usage during conversion:**
- GPU: ~1-2GB per GPU (just torch.distributed overhead)
- CPU RAM: ~100-120GB
- Disk: ~90GB for checkpoint

### 3.3 Verify Checkpoint

```bash
# Check checkpoint structure
ls -lh /checkpoints/mixtral_tp2_ep4/

# You should see:
# - latest_checkpointed_iteration.txt
# - iter_0000001/ (directory with torch_dist format checkpoint)

# Verify it's torch_dist format
ls -lh /checkpoints/mixtral_tp2_ep4/iter_0000001/

# Should see multiple .distcp files and metadata
```

---

## Step 4: Run LoRA Fine-tuning

### 4.1 Test with Mock Data (Quick Test)

```bash
# Inside the container (or start a new one with same mount):

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
  Target modules: ['linear_qkv', 'linear_proj', 'linear_fc1', 'linear_fc2']
...
iteration       10/      10 | ... | lm loss: X.XXX | ...
✓ LoRA Fine-Tuning Complete!
```

**Time estimate:** 2-3 minutes for 10 iterations
**GPU memory:** ~27-28GB per GPU

### 4.2 Fine-tune with Real Data

Prepare your dataset in JSONL format:
```json
{"instruction": "What is the capital of France?", "response": "The capital of France is Paris."}
{"instruction": "Explain photosynthesis.", "response": "Photosynthesis is the process by which plants..."}
```

Then run:
```bash
# Copy your dataset into the container
# (Or mount it as a volume when starting the container)

torchrun --nproc_per_node=8 examples/mixtral/finetune_mixtral_lora.py \
    --load=/checkpoints/mixtral_tp2_ep4 \
    --data_path=/path/to/your/data.jsonl \
    --output_path=/checkpoints/mixtral_lora_finetuned \
    --tensor_model_parallel_size=2 \
    --expert_model_parallel_size=4 \
    --recompute_activations \
    --micro_batch_size=1 \
    --global_batch_size=64 \
    --train_iters=1000 \
    --seq_length=2048 \
    --log_interval=10 \
    --save_interval=200 \
    --lora_rank=16 \
    --lora_alpha=32 \
    --lr=1e-4 \
    --min_lr=1e-6
```

---

## Step 5: Run Full Test Suite

To verify everything works:

```bash
# Inside container:
bash examples/mixtral/test_8xa40.sh
```

This will:
1. Convert checkpoint (if not already done)
2. Run pretraining test (10 iterations)
3. Run LoRA fine-tuning test (10 iterations)
4. Run generation test

**Time estimate:** 5-10 minutes total

---

## Troubleshooting

### Issue 1: Out of Memory (OOM) during conversion

**Symptom:** Process killed during conversion
**Cause:** Not enough CPU RAM

**Solution:**
```bash
# Check available RAM
free -h

# If less than 128GB, you can try:
# 1. Close other applications
# 2. Add swap space (slower but works):
sudo fallocate -l 64G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Issue 2: NCCL timeout or GPU communication errors

**Symptom:** `NCCL error` or timeout messages

**Solution:**
```bash
# Increase NCCL timeout
export NCCL_TIMEOUT=3600

# Use NCCL_DEBUG for more info
export NCCL_DEBUG=INFO

# Verify all GPUs are visible
nvidia-smi

# Check GPU topology
nvidia-smi topo -m
```

### Issue 3: HuggingFace download issues

**Symptom:** Download fails or is very slow

**Solution:**
```bash
# Use HuggingFace CLI to pre-download
huggingface-cli login  # Optional: for gated models
huggingface-cli download mistralai/Mixtral-8x7B-v0.1

# Or set cache directory
export HF_HOME=/checkpoints/hf_cache
mkdir -p $HF_HOME
```

### Issue 4: Docker shared memory too small

**Symptom:** `RuntimeError: DataLoader worker exited unexpectedly`

**Solution:**
```bash
# Restart container with larger shared memory
docker run -it --rm \
    --gpus all \
    --shm-size=64g \  # Increased from 32g
    ...
```

### Issue 5: Permission issues with mounted volumes

**Symptom:** Cannot write to `/checkpoints` directory

**Solution:**
```bash
# On host, fix permissions
sudo chown -R $USER:$USER ~/megatron_checkpoints

# Or run container with user ID
docker run -it --rm \
    --gpus all \
    --user $(id -u):$(id -g) \
    ...
```

---

## Quick Reference Commands

### Start Container
```bash
docker run -it --rm --gpus all --shm-size=32g \
    -v ~/megatron_checkpoints:/checkpoints \
    -w /workspace/Megatron-Bridge \
    megatron-bridge:latest /bin/bash
```

### Convert Checkpoint
```bash
torchrun --nproc_per_node=8 examples/mixtral/convert_checkpoint.py \
    --hf_model_path=mistralai/Mixtral-8x7B-v0.1 \
    --output_path=/checkpoints/mixtral_tp2_ep4 \
    --tensor_model_parallel_size=2 \
    --expert_model_parallel_size=4
```

### LoRA Fine-tuning (Mock Data)
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

### Run Full Test Suite
```bash
bash examples/mixtral/test_8xa40.sh
```

---

## Additional Resources

- **Megatron-Bridge Documentation**: https://docs.nvidia.com/nemo/megatron-bridge/
- **LoRA Paper**: https://arxiv.org/abs/2106.09685
- **Mixtral Paper**: https://arxiv.org/abs/2401.04088

---

## Summary of What Each Component Does

1. **convert_checkpoint.py**: Converts HuggingFace weights to Megatron torch_dist format
   - Uses CPU initialization (minimal GPU memory)
   - Output: Distributed checkpoint ready for training/inference

2. **finetune_mixtral_lora.py**: Fine-tunes with LoRA adapters
   - Memory efficient: ~27-28GB per GPU (vs ~34-41GB for full fine-tuning)
   - Only trains adapters (~100MB), freezes base model
   - Can use Adam optimizer (better convergence than SGD)

3. **test_8xa40.sh**: Comprehensive test suite
   - Verifies all functionality works on your hardware
   - Quick smoke test before production training

## What Makes This Setup Special

- ✅ **CPU Initialization**: Checkpoint conversion uses CPU RAM, not GPU memory
- ✅ **torch_dist Format**: PEFT-compatible distributed checkpoint format
- ✅ **Proper Provider Pattern**: Follows Megatron-Bridge best practices
- ✅ **Memory Optimized**: LoRA + activation checkpointing fits in 32GB GPUs
- ✅ **Docker Support**: Reproducible environment across systems
