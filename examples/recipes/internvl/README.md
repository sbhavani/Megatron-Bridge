# InternVL3 Finetuning with Megatron-Bridge

Recipes for finetuning InternVL3 models using Megatron-Bridge.

## Quick Start

### Basic Finetuning (Mock Data)

```bash
torchrun --nproc_per_node=1 finetune_internvl.py \
    --recipe internvl3_1b_finetune_config \
    train.train_iters=100
```

### With Custom Dataset

```bash
torchrun --nproc_per_node=1 finetune_internvl.py \
    --recipe internvl3_1b_finetune_config \
    --data-path /path/to/dataset.jsonl \
    --image-folder /path/to/images \
    --dataset-type preloaded
```

### From Pretrained Checkpoint

```bash
# Convert HF checkpoint to Megatron format
python examples/conversion/convert_checkpoints.py import \
    --hf-model OpenGVLab/InternVL3-1B \
    --megatron-path /path/to/megatron_ckpt

# Run finetuning
torchrun --nproc_per_node=1 finetune_internvl.py \
    --recipe internvl3_1b_finetune_config \
    --pretrained-checkpoint /path/to/megatron_ckpt \
    --data-path /path/to/dataset.jsonl
```

### Multi-GPU Training

```bash
torchrun --nproc_per_node=2 finetune_internvl.py \
    --recipe internvl3_4b_finetune_config \
    model.tensor_model_parallel_size=2
```

## Available Recipes

- `internvl3_1b_finetune_config` - 1B model (TP=1)
- `internvl3_2b_finetune_config` - 2B model (TP=1)
- `internvl3_4b_finetune_config` - 4B model (TP=2)
- `internvl3_8b_finetune_config` - 8B model (TP=2)

## Configuration

### Using YAML Config

```bash
torchrun --nproc_per_node=1 finetune_internvl.py \
    --config-file conf/my_config.yaml
```

See `conf/internvl_pretrain_override_example.yaml` for reference.

### Using CLI Overrides

```bash
torchrun --nproc_per_node=1 finetune_internvl.py \
    model.seq_length=4096 \
    train.global_batch_size=16 \
    train.micro_batch_size=2 \
    optimizer.lr=0.00002
```

## Dataset Format

JSONL format with conversation structure:

```json
{
  "messages": [
    {"role": "user", "content": "<image>\nWhat is in this image?"},
    {"role": "assistant", "content": "This image shows a cat."}
  ],
  "images": ["path/to/image.jpg"]
}
```

## Freezing Components

```bash
# Freeze vision encoder (train language model only)
model.freeze_vision_model=true

# Freeze language model (train vision components only)
model.freeze_language_model=true

# Freeze projection layer
model.freeze_vision_projection=true
```

## Monitoring

View training logs with TensorBoard:
```bash
tensorboard --logdir nemo_experiments/internvl_finetune/tb_logs/
```
