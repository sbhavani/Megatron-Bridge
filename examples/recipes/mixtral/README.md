# Mixtral Recipe Examples

This directory contains recipe-based examples for training and finetuning Mixtral models using Megatron-Bridge. These examples follow the modern recipe pattern used across all model families in Megatron-Bridge.

## Overview

Mixtral is a sparse Mixture-of-Experts (MoE) model with 8 experts per layer. The recipe pattern provides:

- ✅ **ConfigContainer-based configuration** - Structured, type-safe configs
- ✅ **YAML + CLI overrides** - Flexible Hydra-style configuration
- ✅ **Framework-provided training** - No manual training loops
- ✅ **Consistent interface** - Same pattern as Llama, Qwen, DeepSeek, etc.

## Available Scripts

### Pretraining

**Script:** `pretrain_mixtral_8x7b.py` (185 lines)

Train Mixtral 8x7B from scratch using the recipe pattern.

```bash
# Basic usage with default config
torchrun --nproc_per_node=8 examples/recipes/mixtral/pretrain_mixtral_8x7b.py

# With YAML config file
torchrun --nproc_per_node=8 pretrain_mixtral_8x7b.py \
    --config-file conf/mixtral_8x7b_pretrain_override_example.yaml

# With CLI overrides (MoE-specific parallelism)
torchrun --nproc_per_node=8 pretrain_mixtral_8x7b.py \
    model.tensor_model_parallel_size=4 \
    model.expert_model_parallel_size=4 \
    train.train_iters=100000

# Testing with mock data
torchrun --nproc_per_node=8 pretrain_mixtral_8x7b.py --mock-data
```

### Finetuning

**Script:** `finetune_mixtral_8x7b.py` (203 lines)

Finetune Mixtral 8x7B for instruction following or domain adaptation.

```bash
# Basic usage (loads pretrained checkpoint)
torchrun --nproc_per_node=8 examples/recipes/mixtral/finetune_mixtral_8x7b.py \
    --pretrained-checkpoint /path/to/megatron_ckpt

# With custom data
torchrun --nproc_per_node=8 finetune_mixtral_8x7b.py \
    --data-path /path/to/instructions.jsonl \
    --pretrained-checkpoint /path/to/megatron_ckpt

# With YAML config
torchrun --nproc_per_node=8 finetune_mixtral_8x7b.py \
    --config-file conf/mixtral_8x7b_finetune_override_example.yaml

# Testing with mock data
torchrun --nproc_per_node=8 finetune_mixtral_8x7b.py --mock-data

# Using 8x22B model instead
torchrun --nproc_per_node=16 finetune_mixtral_8x7b.py \
    --recipe mixtral_8x22b_finetune_config
```

## Configuration Structure

### Configuration Precedence

1. **Base Recipe** - Default configuration from `megatron.bridge.recipes.mixtral`
2. **YAML Overrides** - Applied from `--config-file` (if provided)
3. **CLI Overrides** - Highest precedence, Hydra-style syntax

### Example CLI Overrides

```bash
# Model parallelism (MoE-specific)
model.tensor_model_parallel_size=4
model.pipeline_model_parallel_size=2
model.expert_model_parallel_size=8
model.sequence_parallel=true

# Training hyperparameters
train.train_iters=50000
train.global_batch_size=512
train.micro_batch_size=2

# Optimizer settings
optimizer.lr=0.0002
optimizer.min_lr=0.00002
scheduler.lr_warmup_iters=500

# Checkpoint configuration
checkpoint.save=/path/to/checkpoints
checkpoint.save_interval=1000
```

## MoE-Specific Parallelism

Mixtral uses three types of parallelism for distributing the model:

| Parallelism Type | Flag | Description | Recommended |
|------------------|------|-------------|-------------|
| **Tensor Parallel (TP)** | `tensor_model_parallel_size` | Splits tensors across GPUs | 2-4 |
| **Pipeline Parallel (PP)** | `pipeline_model_parallel_size` | Splits layers across GPUs | 1-2 |
| **Expert Parallel (EP)** | `expert_model_parallel_size` | Splits MoE experts across GPUs | 4-8 |

### Recommended Configurations

**8x A40/A100 80GB (Mixtral 8x7B):**
```yaml
model:
  tensor_model_parallel_size: 2
  pipeline_model_parallel_size: 1
  expert_model_parallel_size: 4
  sequence_parallel: true
```

**16x A100 80GB (Mixtral 8x22B):**
```yaml
model:
  tensor_model_parallel_size: 4
  pipeline_model_parallel_size: 2
  expert_model_parallel_size: 8
  sequence_parallel: true
```

## YAML Config Examples

### Pretraining Config

See `conf/mixtral_8x7b_pretrain_override_example.yaml`:

```yaml
model:
  seq_length: 4096
  tensor_model_parallel_size: 2
  expert_model_parallel_size: 4
  sequence_parallel: true
  recompute_granularity: "selective"

train:
  train_iters: 20
  global_batch_size: 8
  micro_batch_size: 1

optimizer:
  lr: 0.0003
  min_lr: 0.00003
```

### Finetuning Config

See `conf/mixtral_8x7b_finetune_override_example.yaml`:

```yaml
model:
  seq_length: 4096
  tensor_model_parallel_size: 2
  expert_model_parallel_size: 4

train:
  train_iters: 1000
  global_batch_size: 8

optimizer:
  lr: 0.00001  # Lower LR for finetuning
  min_lr: 0.000001

checkpoint:
  load: /path/to/pretrained_checkpoint
  save: /path/to/finetuned_checkpoints
```

## Available Recipe Functions

Import from `megatron.bridge.recipes.mixtral`:

```python
from megatron.bridge.recipes.mixtral import (
    mixtral_8x7b_pretrain_config,
    mixtral_8x7b_finetune_config,
    mixtral_8x22b_pretrain_config,
    mixtral_8x22b_finetune_config,
)

# Load a config programmatically
cfg = mixtral_8x7b_pretrain_config(
    tensor_model_parallel_size=4,
    expert_model_parallel_size=8,
    train_iters=100000,
)
```

### Recipe Parameters

All recipe functions accept these common parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hf_path` | str | Model-specific | HuggingFace model path |
| `tensor_model_parallel_size` | int | 2 | Tensor parallelism degree |
| `pipeline_model_parallel_size` | int | 1 | Pipeline parallelism degree |
| `expert_model_parallel_size` | int | 4 | Expert parallelism degree (MoE) |
| `sequence_parallel` | bool | True | Enable sequence parallelism |
| `train_iters` | int | 300000 | Number of training iterations |
| `global_batch_size` | int | 32 | Global batch size |
| `micro_batch_size` | int | 2 | Micro batch size per GPU |
| `seq_length` | int | 4096 | Sequence length |
| `lr` | float | 3e-4 | Learning rate |
| `mock` | bool | False | Use mock data for testing |

## Comparison to Legacy Scripts

### Code Reduction

| Old Scripts | Lines | New Scripts | Lines | Reduction |
|-------------|-------|-------------|-------|-----------|
| `train_mixtral.py` | 699 | `pretrain_mixtral_8x7b.py` | 185 | **73%** |
| `finetune_mixtral.py` | 906 | `finetune_mixtral_8x7b.py` | 203 | **78%** |
| **Total** | **1,605** | **Total** | **388** | **76%** |

### Features Gained

| Feature | Legacy Scripts | Recipe Scripts |
|---------|----------------|----------------|
| YAML configs | ❌ | ✅ |
| Hydra-style CLI overrides | ❌ | ✅ |
| Type-safe configuration | ❌ | ✅ |
| Framework-provided training | ❌ | ✅ |
| Consistent with other models | ❌ | ✅ |
| Manual training loops | ✅ (700+ lines) | ❌ |
| Manual argparse | ✅ (60+ args) | ❌ |

## Checkpoint Conversion

To use pretrained HuggingFace checkpoints:

```bash
# Convert HF checkpoint to Megatron format
python examples/conversion/convert_checkpoints.py import \
    --hf-model mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --megatron-path /path/to/megatron_ckpt \
    --tensor-model-parallel-size 2 \
    --expert-model-parallel-size 4

# Then use in finetuning
torchrun --nproc_per_node=8 examples/recipes/mixtral/finetune_mixtral_8x7b.py \
    --pretrained-checkpoint /path/to/megatron_ckpt
```

## Troubleshooting

**OOM (Out of Memory):**
- Increase `expert_model_parallel_size` to distribute experts across more GPUs
- Reduce `micro_batch_size` to 1
- Enable selective recompute: `model.recompute_granularity=selective`
- Use gradient accumulation: increase `global_batch_size` without increasing `micro_batch_size`

**Slow performance:**
- Ensure `sequence_parallel=true` when using `tensor_model_parallel_size > 1`
- Use expert parallelism: `expert_model_parallel_size=4` or `8`
- Enable overlap: `ddp.overlap_grad_reduce=true`

**Config not applying:**
- Check YAML indentation (spaces, not tabs)
- Verify CLI override syntax: `section.key=value` (no spaces around `=`)
- Check precedence: CLI overrides > YAML > base recipe

## Migration from Legacy Scripts

### Old Pattern (Manual)
```python
# Old: train_mixtral.py (699 lines)
parser = argparse.ArgumentParser()
parser.add_argument("--tensor-model-parallel-size", type=int, default=1)
parser.add_argument("--expert-model-parallel-size", type=int, default=1)
# ... 60+ more arguments

# Manual training loop
for epoch in range(args.num_epochs):
    for batch in dataloader:
        outputs = model(input_ids)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

### New Pattern (Recipe)
```python
# New: pretrain_mixtral_8x7b.py (185 lines)
from megatron.bridge.recipes.mixtral import mixtral_8x7b_pretrain_config
from megatron.bridge.training.pretrain import pretrain

cfg = mixtral_8x7b_pretrain_config()
# Apply YAML/CLI overrides
pretrain(config=cfg, forward_step_func=forward_step)
```

## Additional Resources

- [Megatron-Bridge Documentation](https://docs.nvidia.com/megatron-bridge)
- [Recipe Pattern Guide](../../docs/recipe_pattern.md)
- [MoE Best Practices](../../docs/moe_best_practices.md)
- [Legacy Mixtral Examples](../../mixtral/) (deprecated)
