# Mixtral 8x7B with Megatron-Bridge

Examples for using Mixtral Mixture-of-Experts models with Megatron-Bridge.

## Quick Start

### Prerequisites

```bash
pip install torch transformers megatron-core flash-attn
```

### Checkpoint Conversion

For large models like Mixtral 8x7B, convert to Megatron format first to avoid OOM:

```bash
torchrun --nproc_per_node=2 examples/mixtral/convert_checkpoint.py \
    --hf_model_path="mistralai/Mixtral-8x7B-v0.1" \
    --output_path=/path/to/megatron/checkpoint \
    --tensor_model_parallel_size=2
```

### Text Generation

**Single-GPU inference:**

```bash
python examples/mixtral/generate_text.py \
    --hf_model_path="mistralai/Mixtral-8x7B-v0.1" \
    --prompt="Explain quantum computing:" \
    --max_tokens=200
```

**Multi-GPU with converted checkpoint:**

```bash
torchrun --nproc_per_node=2 examples/mixtral/generate_text.py \
    --load=/path/to/megatron/checkpoint \
    --tensor_model_parallel_size=2 \
    --prompt="Explain quantum computing:" \
    --max_tokens=200
```

### LoRA Fine-Tuning (Recommended for 2x A100 80GB)

LoRA is the recommended approach for fine-tuning Mixtral on limited hardware:

```bash
torchrun --nproc_per_node=2 examples/mixtral/finetune_mixtral_lora.py \
    --load=/path/to/megatron/checkpoint \
    --data_path=/path/to/instructions.jsonl \
    --output_path=/path/to/lora/adapters \
    --tensor_model_parallel_size=2 \
    --train_iters=1000 \
    --lora_rank=16 \
    --lora_alpha=32 \
    --seq_length=512
```

**With mock data (for testing):**

```bash
torchrun --nproc_per_node=2 examples/mixtral/finetune_mixtral_lora.py \
    --load=/path/to/megatron/checkpoint \
    --mock_data \
    --output_path=/path/to/lora/test \
    --tensor_model_parallel_size=2 \
    --train_iters=10
```

**LoRA Benefits:**
- Memory efficient: ~30-35GB per GPU vs ~75-80GB for full fine-tuning
- Supports longer sequences (512+ tokens)
- Can use Adam optimizer
- Only trains small adapter matrices (~100MB)

## Hardware Requirements

### 2x A100 80GB
- ✅ Checkpoint conversion
- ✅ Generation/inference
- ✅ **LoRA fine-tuning** (recommended)
- ❌ Full pre-training (insufficient memory)

### 8x A100 80GB or 8x A40 48GB
- ✅ All operations including full pre-training
- Use configuration: TP=2, EP=4 or TP=4, EP=2

## Testing

Comprehensive test script for 2x A100:

```bash
bash examples/mixtral/test_2xa100_lora.sh
```

This tests:
1. Checkpoint conversion (HF → Megatron format)
2. Basic generation verification
3. LoRA fine-tuning with mock data

## Files

- `convert_checkpoint.py` - HuggingFace ↔ Megatron checkpoint conversion
- `generate_text.py` - Text generation and interactive chat
- `finetune_mixtral_lora.py` - LoRA fine-tuning (recommended for limited hardware)
- `train_mixtral.py` - Full pre-training (requires 8+ GPUs)
- `test_2xa100_lora.sh` - Automated testing for 2x A100 setup

## Configuration

### Parallelism Options

- `--tensor_model_parallel_size` (TP): Splits tensors across GPUs
- `--pipeline_model_parallel_size` (PP): Splits layers across GPUs
- `--expert_model_parallel_size` (EP): Splits MoE experts across GPUs

### Recommended Configurations

**2x A100 80GB:**
```bash
--tensor_model_parallel_size=2 \
--expert_model_parallel_size=1
```

**8x A100 80GB:**
```bash
--tensor_model_parallel_size=2 \
--expert_model_parallel_size=4
```

## Troubleshooting

**OOM during conversion:**
- Use CPU initialization (enabled by default in convert_checkpoint.py)
- Convert with TP=1, use TP>1 at runtime

**OOM during fine-tuning:**
- Use LoRA instead of full fine-tuning
- Reduce sequence length
- Enable activation checkpointing: `--recompute_activations`

**Slow performance:**
- Ensure flash-attn is installed: `pip install flash-attn`
- Use expert parallelism for multi-GPU: `--expert_model_parallel_size=N`
