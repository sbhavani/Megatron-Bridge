# InternVL Examples

Examples for using InternVL vision-language models with Megatron-Bridge.

## Quick Start

### Prerequisites

```bash
pip install 'transformers>=4.37.2' torch pillow
```

InternVL models require `trust_remote_code=True` as they use custom architectures.

### Basic Inference

**Single GPU:**
```bash
python load_model_and_inference.py OpenGVLab/InternVL3-1B \
    --prompt "What is artificial intelligence?" \
    --max_new_tokens 50
```

**Multi-GPU (Tensor Parallelism):**
```bash
torchrun --nproc_per_node=2 load_model_and_inference.py OpenGVLab/InternVL3-2B \
    --tp_size 2 \
    --prompt "Explain quantum computing in simple terms." \
    --max_new_tokens 100
```

**Quick Test:**
```bash
bash run_test.sh
```

## Available Models

- `OpenGVLab/InternVL3-1B` - 1B parameters (single GPU)
- `OpenGVLab/InternVL3-2B` - 2B parameters (1-2 GPUs)
- `OpenGVLab/InternVL3-4B` - 4B parameters (2+ GPUs)
- `OpenGVLab/InternVL3-8B` - 8B parameters (4+ GPUs)

## Parallelism

| Model | Recommended Config |
|-------|-------------------|
| 1B    | TP=1 (1 GPU) |
| 2B    | TP=1-2 (1-2 GPUs) |
| 4B    | TP=2 (2 GPUs) |
| 8B    | TP=2-4 (4+ GPUs) |

For finetuning examples, see `examples/recipes/internvl/`.

## Troubleshooting

**Import Error:** Ensure `transformers>=4.37.2` is installed.

**Shape Mismatch:** InternVL uses actual vocab size (151674) vs config (151936). The bridge handles this automatically.

**Trust Remote Code:** InternVL requires `trust_remote_code=True` when loading models.
