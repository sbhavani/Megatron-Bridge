# SFT Dataset Formats

This guide explains how to format datasets for Supervised Fine-Tuning (SFT) in Megatron Bridge, including how to control whether the model trains on the entire conversation or just the assistant responses.

## Key Concept: answer_only_loss

The most important parameter for SFT is `answer_only_loss`:

| Setting | Behavior | Use Case |
|---------|----------|----------|
| `answer_only_loss=True` (default) | Loss computed **only on assistant/output tokens** | Standard fine-tuning where you want the model to learn to generate responses |
| `answer_only_loss=False` | Loss computed on **entire sequence** | When you want the model to also learn from the context/prompt |

This is equivalent to HuggingFace TRL's distinction between training on completions only vs. the full conversation.

## Supported Dataset Formats

### 1. Standard JSONL Format

The simplest format for SFT. Each line is a JSON object with `input` and `output` fields:

```jsonl
{"input": "What is the capital of France?", "output": "The capital of France is Paris."}
{"input": "Explain photosynthesis briefly.", "output": "Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen."}
```

**Configuration:**
```python
dataset_kwargs = {
    "answer_only_loss": True,  # Train only on "output" field
    "prompt_template": "{input} {output}",
    "label_key": "output",
}
```

### 2. Chat Format (HuggingFace Style)

For multi-turn conversations using the standard HuggingFace chat format:

```jsonl
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is AI?"}, {"role": "assistant", "content": "AI stands for Artificial Intelligence..."}]}
{"messages": [{"role": "user", "content": "Hello!"}, {"role": "assistant", "content": "Hello! How can I help you today?"}]}
```

**Configuration:**
```python
dataset_kwargs = {
    "chat": True,
    "use_hf_tokenizer_chat_template": True,
    "answer_only_loss": True,  # Train only on assistant responses
}
```

When using `use_hf_tokenizer_chat_template=True`, the tokenizer's `apply_chat_template()` method is used to format the conversation and automatically identify which tokens belong to assistant responses.

### 3. ShareGPT Format

An alternative chat format commonly used in the community:

```jsonl
{"conversations": [{"from": "User", "value": "What is machine learning?"}, {"from": "Assistant", "value": "Machine learning is a subset of AI..."}]}
{"conversations": [{"from": "User", "value": "Can you explain more?"}, {"from": "Assistant", "value": "Sure! Let me elaborate..."}]}
```

**Configuration:**
```python
dataset_kwargs = {
    "chat": True,
    "use_hf_tokenizer_chat_template": False,  # Uses legacy ShareGPT processing
    "answer_only_loss": True,
}
```

### 4. Packed Sequences (.npy)

For maximum efficiency, sequences can be packed together to minimize padding. This format uses pre-tokenized NumPy binary files.

See [Packed Sequences](packed-sequences.md) for details on creating and using packed datasets.

## How Loss Masking Works

When `answer_only_loss=True`, the loss mask is constructed as follows:

```
Input:    [What] [is] [2+2?] [The] [answer] [is] [4] [.]
          |---- context ----|-------- answer --------|
Loss Mask:  0     0     0      1      1      1    1   1
```

Only tokens with `loss_mask=1` contribute to the training loss. This ensures the model learns to generate the answer without being penalized for the prompt tokens.

The implementation in `sft.py`:

```python
def _build_loss_mask(self, processed_example):
    input_ids = processed_example["input_ids"]
    answer_start_idx = processed_example["answer_start_idx"]
    if self.answer_only_loss:
        # Only include tokens from answer_start_idx onwards
        loss_mask = [float(idx >= answer_start_idx) for idx in range(len(input_ids))]
    else:
        # Include all tokens
        loss_mask = [1.0] * len(input_ids)
    return loss_mask
```

## Configuration Examples

### Basic SFT with Answer-Only Loss

```python
from megatron.bridge.training.config import FinetuningDatasetConfig

dataset_config = FinetuningDatasetConfig(
    dataset_root=Path("/path/to/data"),  # Contains training.jsonl, validation.jsonl
    seq_length=2048,
    dataset_kwargs={
        "answer_only_loss": True,
        "prompt_template": "{input} {output}",
        "truncation_field": "input",  # Truncate context if too long, preserve answer
    }
)
```

### Chat Fine-Tuning with HuggingFace Template

```python
dataset_config = FinetuningDatasetConfig(
    dataset_root=Path("/path/to/data"),
    seq_length=4096,
    dataset_kwargs={
        "chat": True,
        "use_hf_tokenizer_chat_template": True,
        "answer_only_loss": True,
    }
)
```

### Full Sequence Training (Context + Answer)

If you want the model to learn from both the context and the answer:

```python
dataset_config = FinetuningDatasetConfig(
    dataset_root=Path("/path/to/data"),
    seq_length=2048,
    dataset_kwargs={
        "answer_only_loss": False,  # Train on entire sequence
    }
)
```

## Comparison with HuggingFace TRL

| Megatron Bridge | HuggingFace TRL | Behavior |
|-----------------|-----------------|----------|
| `answer_only_loss=True` | `DataCollatorForCompletionOnlyLM` | Train only on completions/answers |
| `answer_only_loss=False` | `DataCollatorForLanguageModeling` | Train on full sequence |
| `chat=True` with `use_hf_tokenizer_chat_template=True` | `apply_chat_template()` | Use tokenizer's chat template |

## Custom Prompt Templates

For datasets with different field names, use `prompt_template`:

```jsonl
{"context": "You are a math tutor.", "question": "What is 2+2?", "answer": "4"}
```

```python
dataset_kwargs = {
    "prompt_template": "{context}\n\nQuestion: {question}\nAnswer: {answer}",
    "label_key": "answer",
    "answer_only_loss": True,
    "truncation_field": "context,question",  # Multiple fields to truncate
}
```

## Truncation Behavior

When sequences exceed `seq_length`, the `truncation_field` parameter controls which fields get truncated:

- `"input"` (default): Truncate only the input/context
- `"output"`: Truncate only the output/answer (not recommended)
- `"input,output"`: Distribute truncation across both fields proportionally

The `truncation_method` parameter controls direction:
- `"right"` (default): Remove tokens from the end
- `"left"`: Remove tokens from the beginning

## File Organization

Place your dataset files in the `dataset_root` directory:

```
/path/to/data/
├── training.jsonl      # Required
├── validation.jsonl    # Optional (set do_validation=False to skip)
└── test.jsonl          # Optional (set do_test=False to skip)
```

## Common Issues

### Training on Entire Message Instead of Just Answer

**Symptom:** Model loss includes context tokens, not just answer tokens.

**Solution:** Ensure `answer_only_loss=True` is set in `dataset_kwargs`:

```python
dataset_kwargs = {
    "answer_only_loss": True,  # This is the default, but be explicit
}
```

### Chat Template Not Applied

**Symptom:** Special tokens or formatting not appearing correctly.

**Solution:** Enable both `chat` and `use_hf_tokenizer_chat_template`:

```python
dataset_kwargs = {
    "chat": True,
    "use_hf_tokenizer_chat_template": True,
}
```

### Context Getting Truncated Too Aggressively

**Symptom:** Important context is removed before the answer.

**Solution:** Increase `seq_length` or adjust truncation settings:

```python
dataset_kwargs = {
    "truncation_field": "input",
    "truncation_method": "left",  # Remove from beginning instead of end
}
```

## API Reference

### create_sft_dataset()

Factory function that creates the appropriate SFT dataset based on parameters:

```python
def create_sft_dataset(
    path: Path,                           # Path to .jsonl or .npy file
    tokenizer: MegatronTokenizer,         # Tokenizer instance
    seq_length: int = 2048,               # Maximum sequence length
    add_bos: bool = False,                # Add beginning-of-sentence token
    add_eos: bool = True,                 # Add end-of-sentence token
    add_sep: bool = False,                # Add separator between input/output
    label_key: str = "output",            # JSON key for target output
    answer_only_loss: bool = True,        # Train only on answers
    truncation_field: str = "input",      # Field(s) to truncate
    prompt_template: str = "{input} {output}",  # Template for combining fields
    truncation_method: str = "right",     # "left" or "right"
    chat: bool = False,                   # Use chat dataset
    use_hf_tokenizer_chat_template: bool = False,  # Apply HF chat template
    ...
) -> GPTSFTDataset
```

## Related Documentation

- [Packed Sequences](packed-sequences.md) - Efficient sequence packing
- [PEFT](peft.md) - Parameter-efficient fine-tuning (LoRA)
- [Configuration Container Overview](config-container-overview.md) - Full configuration reference
