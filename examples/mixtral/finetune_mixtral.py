#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Mixtral Supervised Fine-Tuning (SFT) with Megatron-Bridge.

This script demonstrates how to fine-tune Mixtral models on instruction-following
datasets using Megatron-Bridge with distributed training support.

Supports both full fine-tuning and LoRA (Low-Rank Adaptation) for memory efficiency.

Supported dataset formats:
    - Alpaca format: {"instruction": "...", "input": "...", "output": "..."}
    - ShareGPT format: {"conversations": [{"from": "human/gpt", "value": "..."}]}
    - Custom JSONL with instruction/response pairs

LoRA Benefits:
    - Freezes base model → No gradients for 47B parameters
    - Only trains adapters → Tiny matrices (~10-100MB)
    - Can use Adam → Adapter overhead is negligible
    - Memory efficient: ~27-28GB per GPU (vs 34-41GB for full fine-tuning)

Examples:
    # Full fine-tuning from HuggingFace checkpoint (requires high memory)
    torchrun --nproc_per_node=8 examples/mixtral/finetune_mixtral.py \\
        --hf_model_path="mistralai/Mixtral-8x7B-v0.1" \\
        --data_path=/path/to/instructions.jsonl \\
        --output_path=/path/to/finetuned \\
        --expert_model_parallel_size=8 \\
        --train_iters=1000

    # LoRA fine-tuning (memory-efficient, recommended for 2-4 GPUs)
    torchrun --nproc_per_node=8 examples/mixtral/finetune_mixtral.py \\
        --load=/path/to/megatron/checkpoint \\
        --data_path=/path/to/instructions.jsonl \\
        --output_path=/path/to/lora_adapters \\
        --use_lora \\
        --lora_rank=16 \\
        --lora_alpha=32 \\
        --tensor_model_parallel_size=2 \\
        --expert_model_parallel_size=4 \\
        --train_iters=1000

    # Test with mock instruction data
    torchrun --nproc_per_node=2 examples/mixtral/finetune_mixtral.py \\
        --load=/tmp/mixtral_tp2 \\
        --mock_data \\
        --output_path=/tmp/mixtral_sft_test \\
        --tensor_model_parallel_size=2 \\
        --train_iters=10
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from megatron.bridge import AutoBridge
from megatron.bridge.utils.common_utils import print_rank_0


def setup_distributed():
    """Initialize distributed training environment."""
    if not dist.is_initialized():
        dist.init_process_group("nccl")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)


class MockInstructionDataset(Dataset):
    """Mock instruction-following dataset for testing."""

    def __init__(self, seq_length: int, vocab_size: int, num_samples: int = 1000):
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.num_samples = num_samples

        # Common instruction templates
        self.instructions = [
            "What is the capital of France?",
            "Explain quantum computing in simple terms.",
            "Write a short poem about nature.",
            "How do I make a cake?",
            "Translate 'hello' to Spanish.",
        ]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random tokens for instruction and response
        # In real SFT, we'd mask the instruction tokens in the loss
        input_ids = torch.randint(
            0, self.vocab_size, (self.seq_length,), dtype=torch.long
        )

        # Create attention mask (all ones for mock data)
        attention_mask = torch.ones(self.seq_length, dtype=torch.long)

        # Create labels (same as input_ids for language modeling)
        # In real SFT, we'd set instruction tokens to -100 to ignore in loss
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class InstructionDataset(Dataset):
    """
    Dataset for instruction-following fine-tuning.

    Supports multiple formats:
    - Alpaca: {"instruction": "...", "input": "...", "output": "..."}
    - ShareGPT: {"conversations": [{"from": "human/gpt", "value": "..."}]}
    - Simple: {"instruction": "...", "response": "..."}
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        seq_length: int,
        mask_instruction: bool = True,
    ):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.mask_instruction = mask_instruction

        # Load dataset
        print_rank_0(f"Loading instruction dataset from {data_path}")
        self.examples = []

        with open(data_path, 'r') as f:
            for line in f:
                self.examples.append(json.loads(line))

        print_rank_0(f"Loaded {len(self.examples)} examples")

    def format_example(self, example: Dict) -> tuple[str, str]:
        """Format example into instruction and response."""
        # Alpaca format
        if "instruction" in example and "output" in example:
            instruction = example["instruction"]
            if example.get("input"):
                instruction = f"{instruction}\n\nInput: {example['input']}"
            response = example["output"]
            return instruction, response

        # ShareGPT format
        elif "conversations" in example:
            convs = example["conversations"]
            instruction = ""
            response = ""
            for conv in convs:
                if conv["from"] in ["human", "user"]:
                    instruction = conv["value"]
                elif conv["from"] in ["gpt", "assistant"]:
                    response = conv["value"]
                    break
            return instruction, response

        # Simple format
        elif "instruction" in example and "response" in example:
            return example["instruction"], example["response"]

        else:
            raise ValueError(f"Unknown dataset format: {example.keys()}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        instruction, response = self.format_example(example)

        # Format with chat template if available
        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
            messages = [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": response},
            ]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        else:
            # Fallback: simple concatenation with special tokens
            text = f"[INST] {instruction} [/INST] {response}</s>"

        # Tokenize
        tokens = self.tokenizer(
            text,
            max_length=self.seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)

        # Create labels
        labels = input_ids.clone()

        # Mask instruction tokens if requested (only compute loss on response)
        if self.mask_instruction:
            # Find where the response starts (after [/INST])
            inst_end_token = self.tokenizer.encode("[/INST]", add_special_tokens=False)
            if inst_end_token:
                # Simple masking: find the instruction end marker
                for i in range(len(input_ids) - len(inst_end_token)):
                    if all(input_ids[i + j] == inst_end_token[j] for j in range(len(inst_end_token))):
                        # Mask everything before and including [/INST]
                        labels[:i + len(inst_end_token)] = -100
                        break

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def load_model(args):
    """Load and configure Mixtral model."""
    if args.load:
        # Load from pre-converted Megatron checkpoint
        print_rank_0(f"Loading Mixtral model from checkpoint: {args.load}...")

        # Create provider directly with Mixtral 8x7B config
        from megatron.bridge.models import MixtralModelProvider

        provider = MixtralModelProvider(
            tensor_model_parallel_size=args.tensor_model_parallel_size,
            pipeline_model_parallel_size=args.pipeline_model_parallel_size,
            expert_model_parallel_size=args.expert_model_parallel_size,
            context_parallel_size=args.context_parallel_size,
            seq_length=args.seq_length,
            params_dtype=torch.bfloat16 if args.bf16 else torch.float32,
            bf16=args.bf16,
            fp16=args.fp16,
            # Sequence parallelism required when using TP > 1 with MoE
            sequence_parallel=args.tensor_model_parallel_size > 1,
            # Enable activation checkpointing for memory efficiency
            # Selective recompute checkpoints attention and MoE activations
            recompute_granularity="selective" if args.recompute_activations else None,
            recompute_modules=["core_attn", "moe"] if args.recompute_activations else None,
        )

        # Skip weight initialization since we're loading from checkpoint
        provider.perform_initialization = False

    else:
        # Load from HuggingFace
        print_rank_0(f"Loading Mixtral model from {args.hf_model_path}...")

        # Load model using AutoBridge
        bridge = AutoBridge.from_hf_pretrained(
            args.hf_model_path,
            trust_remote_code=True,
        )

        # Convert to Megatron provider
        provider = bridge.to_megatron_provider()

        # Configure parallelism
        provider.tensor_model_parallel_size = args.tensor_model_parallel_size
        provider.pipeline_model_parallel_size = args.pipeline_model_parallel_size
        provider.expert_model_parallel_size = args.expert_model_parallel_size
        provider.context_parallel_size = args.context_parallel_size

        # Configure precision
        provider.bf16 = args.bf16
        provider.fp16 = args.fp16
        if args.bf16:
            provider.params_dtype = torch.bfloat16
        elif args.fp16:
            provider.params_dtype = torch.float16

        # Override sequence length if specified
        if args.seq_length:
            provider.seq_length = args.seq_length

    return provider


def apply_lora(model, args):
    """Apply LoRA adapters to model and freeze base parameters."""
    print_rank_0("\n" + "=" * 80)
    print_rank_0("Applying LoRA Adapters")
    print_rank_0("=" * 80)
    print_rank_0(f"LoRA Configuration:")
    print_rank_0(f"  Rank: {args.lora_rank}")
    print_rank_0(f"  Alpha: {args.lora_alpha}")
    print_rank_0(f"  Dropout: {args.lora_dropout}")
    print_rank_0(f"  Target modules: {args.lora_target_modules}")
    print_rank_0("=" * 80 + "\n")

    from megatron.bridge.peft.lora import LoRA

    # Create LoRA configuration
    lora_config = LoRA(
        target_modules=args.lora_target_modules,
        dim=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )

    # Apply LoRA to the model
    # The model is a list with one element (the actual model)
    model_module = model[0]

    # Freeze all base model parameters
    for name, param in model_module.named_parameters():
        param.requires_grad = False

    # Apply LoRA adapters (this will add trainable parameters)
    lora_config.apply(model_module)

    # Count trainable parameters
    total_params = sum(p.numel() for p in model_module.parameters())
    trainable_params = sum(p.numel() for p in model_module.parameters() if p.requires_grad)

    print_rank_0(f"Total parameters: {total_params:,}")
    print_rank_0(f"Trainable parameters (LoRA): {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print_rank_0(f"Frozen parameters: {total_params - trainable_params:,} ({100 * (total_params - trainable_params) / total_params:.2f}%)\n")


def get_dataloader(args, tokenizer):
    """Create dataloader for instruction-following data."""
    if args.mock_data:
        print_rank_0("Using mock instruction data for testing")
        dataset = MockInstructionDataset(
            seq_length=args.seq_length,
            vocab_size=32000,  # Mixtral vocab size
            num_samples=100,
        )
    else:
        dataset = InstructionDataset(
            data_path=args.data_path,
            tokenizer=tokenizer,
            seq_length=args.seq_length,
            mask_instruction=args.mask_instruction,
        )

    # Use DistributedSampler for multi-GPU training
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.micro_batch_size,
        sampler=sampler,
        num_workers=args.num_workers if not args.mock_data else 0,
        pin_memory=True,
    )

    return dataloader


def finetune(args):
    """Main fine-tuning function."""
    # Validate arguments
    if not args.mock_data and not args.data_path:
        raise ValueError("Either --mock_data or --data_path must be specified")
    if not args.load and not args.hf_model_path:
        raise ValueError("Either --load or --hf_model_path must be specified")

    # Setup distributed
    setup_distributed()

    # Initialize model parallelism
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        expert_model_parallel_size=args.expert_model_parallel_size,
        context_parallel_size=args.context_parallel_size,
    )

    # Initialize RNG state for model parallelism
    # This is required for tensor parallel and sequence parallel training
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    model_parallel_cuda_manual_seed(args.seed)
    print_rank_0(f"Initialized CUDA RNG with seed: {args.seed}")

    # Load model
    provider = load_model(args)

    # Check if we need DDP (only when DP > 1)
    # For MoE models with expert parallelism, use expert data parallel size
    ep_size = parallel_state.get_expert_model_parallel_world_size()
    if ep_size > 1:
        dp_size = parallel_state.get_expert_data_parallel_world_size()
        print_rank_0(f"\nExpert model parallel size: {ep_size}")
        print_rank_0(f"Expert data parallel size: {dp_size}")
    else:
        dp_size = parallel_state.get_data_parallel_world_size()
        print_rank_0(f"\nData parallel size: {dp_size}")

    need_ddp = dp_size > 1

    # Distributed optimizer requires DP > 1 (needs multiple ranks to distribute across)
    use_distributed_optimizer = args.use_distributed_optimizer and need_ddp
    if args.use_distributed_optimizer and not need_ddp:
        print_rank_0("Note: Distributed optimizer disabled (requires DP > 1)")

    # overlap_grad_reduce requires distributed optimizer (uses main_grad buffers)
    overlap_grad_reduce = args.overlap_grad_reduce and use_distributed_optimizer
    if args.overlap_grad_reduce and not use_distributed_optimizer:
        print_rank_0("Note: overlap_grad_reduce disabled (requires distributed optimizer)")

    # gradient_accumulation_fusion requires distributed optimizer (uses main_grad buffers)
    if not use_distributed_optimizer and provider.gradient_accumulation_fusion:
        print_rank_0("Note: gradient_accumulation_fusion disabled (requires distributed optimizer)")
        provider.gradient_accumulation_fusion = False

    print_rank_0(f"DDP wrapping: {'enabled' if need_ddp else 'disabled (DP=1, no gradient synchronization needed)'}")
    print_rank_0(f"Distributed optimizer: {'enabled' if use_distributed_optimizer else 'disabled'}")
    print_rank_0(f"Overlap grad reduce: {'enabled' if overlap_grad_reduce else 'disabled'}")
    print_rank_0(f"Gradient accumulation fusion: {'enabled' if provider.gradient_accumulation_fusion else 'disabled'}")

    # Configure DDP
    ddp_config = DistributedDataParallelConfig(
        grad_reduce_in_fp32=True,
        overlap_grad_reduce=overlap_grad_reduce,
        use_distributed_optimizer=use_distributed_optimizer,
    )

    # Get distributed model
    print_rank_0("\nInitializing distributed model...")
    model = provider.provide_distributed_model(
        ddp_config=ddp_config,
        wrap_with_ddp=need_ddp,
    )

    # Load checkpoint weights if loading from pre-converted checkpoint
    if args.load:
        print_rank_0(f"Loading weights from {args.load}...")
        checkpoint_base = Path(args.load)

        # Read iteration from tracker file (Megatron format)
        tracker_filename = checkpoint_base / "latest_checkpointed_iteration.txt"
        if tracker_filename.exists():
            with open(tracker_filename, 'r') as f:
                iteration_str = f.read().strip()
                if iteration_str == 'release':
                    iteration_dir = checkpoint_base / 'release'
                else:
                    iteration = int(iteration_str)
                    iteration_dir = checkpoint_base / f"iter_{iteration:07d}"
            print_rank_0(f"  Loading from iteration: {iteration_str}")
        else:
            # Fallback: try to find iteration directory
            print_rank_0(f"  Warning: Tracker file not found, looking for iteration directories...")
            iter_dirs = sorted(checkpoint_base.glob("iter_*"))
            if iter_dirs:
                iteration_dir = iter_dirs[-1]  # Use latest
                print_rank_0(f"  Found iteration directory: {iteration_dir.name}")
            else:
                raise FileNotFoundError(f"No checkpoint iterations found in {checkpoint_base}")

        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        ep_rank = parallel_state.get_expert_model_parallel_rank()

        checkpoint_name = f"mp_rank_{tp_rank:02d}_{pp_rank:03d}_{ep_rank:03d}"
        checkpoint_path = iteration_dir / checkpoint_name / "model_optim_rng.pt"

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model[0].load_state_dict(checkpoint['model'])
        print_rank_0("✓ Checkpoint loaded successfully")

    # Apply LoRA if requested
    if args.use_lora:
        apply_lora(model, args)

    # Load tokenizer
    tokenizer_path = args.hf_model_path if args.hf_model_path else "mistralai/Mixtral-8x7B-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create dataloader
    print_rank_0("\nPreparing data...")
    dataloader = get_dataloader(args, tokenizer)

    # Configure optimizer
    print_rank_0("\nConfiguring optimizer...")
    optimizer_config = OptimizerConfig(
        lr=args.lr,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_eps=args.adam_eps,
        clip_grad=args.clip_grad,
        bf16=args.bf16,
        fp16=args.fp16,
        use_distributed_optimizer=args.use_distributed_optimizer,
    )

    optimizer = provider.provide_optimizer(model, optimizer_config)

    # Training loop
    print_rank_0("\n" + "=" * 80)
    training_mode = "LoRA Fine-Tuning" if args.use_lora else "Full Fine-Tuning"
    print_rank_0(f"Starting {training_mode}")
    print_rank_0("=" * 80)
    print_rank_0(f"Total iterations: {args.train_iters}")
    print_rank_0(f"Micro batch size: {args.micro_batch_size}")
    print_rank_0(f"Global batch size: {args.global_batch_size}")
    print_rank_0(f"Learning rate: {args.lr}")
    print_rank_0(f"Mask instruction tokens: {args.mask_instruction}")
    if args.use_lora:
        print_rank_0(f"LoRA rank: {args.lora_rank}, alpha: {args.lora_alpha}")
    print_rank_0("=" * 80 + "\n")

    model[0].train()
    iteration = 0
    total_loss = 0.0

    data_iterator = iter(dataloader)

    while iteration < args.train_iters:
        try:
            batch = next(data_iterator)
        except StopIteration:
            # Reset iterator when epoch ends
            data_iterator = iter(dataloader)
            batch = next(data_iterator)

        # Move batch to GPU
        batch = {k: v.cuda() for k, v in batch.items()}

        # Create position_ids if not in batch
        if "position_ids" not in batch:
            batch_size, seq_length = batch["input_ids"].shape
            batch["position_ids"] = torch.arange(
                seq_length, device=batch["input_ids"].device
            ).unsqueeze(0).expand(batch_size, -1)

        # Create attention_mask if not in batch
        if "attention_mask" not in batch:
            batch_size, seq_length = batch["input_ids"].shape
            batch["attention_mask"] = torch.ones(
                batch_size, seq_length, device=batch["input_ids"].device, dtype=torch.bool
            )

        # Forward pass
        def loss_func(output_tensor):
            """Compute cross-entropy loss."""
            logits = output_tensor
            labels = batch["labels"]

            # Shift for causal language modeling
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Compute loss using vocab-parallel cross entropy
            # vocab_parallel_cross_entropy expects [seq_len, batch, vocab_size]
            # but we have [batch, seq_len, vocab_size], so transpose
            from megatron.core.tensor_parallel.cross_entropy import vocab_parallel_cross_entropy

            shift_logits_transposed = shift_logits.transpose(0, 1).contiguous()  # [seq_len, batch, vocab]
            shift_labels_transposed = shift_labels.transpose(0, 1).contiguous()  # [seq_len, batch]

            loss_tensor = vocab_parallel_cross_entropy(
                shift_logits_transposed,
                shift_labels_transposed,
            )

            # Handle ignore_index=-100 by masking out those positions
            loss_mask = (shift_labels != -100).float()  # [batch, seq_len]
            loss_mask_transposed = loss_mask.transpose(0, 1)  # [seq_len, batch]

            # Apply mask and compute mean over valid tokens only
            masked_loss = loss_tensor * loss_mask_transposed
            num_valid_tokens = loss_mask.sum()
            loss = masked_loss.sum() / num_valid_tokens.clamp(min=1.0)

            return loss, {}

        # Forward-backward
        from megatron.core.pipeline_parallel import get_forward_backward_func

        forward_backward_func = get_forward_backward_func()

        losses_reduced = forward_backward_func(
            forward_step_func=lambda data_iterator, model: (
                model(
                    batch["input_ids"],
                    position_ids=batch["position_ids"],
                    attention_mask=batch["attention_mask"],
                ),
                loss_func,
            ),
            data_iterator=iter([batch]),
            model=model,
            num_microbatches=1,
            forward_only=False,
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
        )

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()

        # Track loss
        if losses_reduced:
            batch_loss = losses_reduced[0].item()
            total_loss += batch_loss

            if iteration % args.log_interval == 0:
                avg_loss = total_loss / (iteration + 1)
                print_rank_0(
                    f"Iteration {iteration}/{args.train_iters} | "
                    f"Loss: {batch_loss:.4f} | "
                    f"Avg Loss: {avg_loss:.4f}"
                )

        iteration += 1

        # Save checkpoint
        if iteration % args.save_interval == 0:
            print_rank_0(f"\nSaving checkpoint at iteration {iteration}...")
            save_checkpoint(args, model, optimizer, iteration)

    # Final save
    print_rank_0("\n" + "=" * 80)
    print_rank_0(f"{training_mode} complete!")
    print_rank_0("=" * 80)
    print_rank_0(f"Saving final checkpoint...")
    save_checkpoint(args, model, optimizer, iteration)
    checkpoint_type = "LoRA adapters" if args.use_lora else "model checkpoint"
    print_rank_0(f"Final {checkpoint_type} saved to: {args.output_path}")


def save_checkpoint(args, model, optimizer, iteration):
    """Save checkpoint in Megatron format."""
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    iteration_dir = output_path / f"iter_{iteration:07d}"
    iteration_dir.mkdir(parents=True, exist_ok=True)

    if parallel_state.get_data_parallel_rank() == 0:
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        ep_rank = parallel_state.get_expert_model_parallel_rank()

        checkpoint_name = f"mp_rank_{tp_rank:02d}_{pp_rank:03d}_{ep_rank:03d}"
        checkpoint_path = iteration_dir / checkpoint_name / "model_optim_rng.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        # Create state dict
        import types

        args_namespace = types.SimpleNamespace(**vars(args))

        # For LoRA, only save LoRA parameters
        if args.use_lora:
            # Save only LoRA adapter parameters
            lora_state_dict = {
                name: param for name, param in model[0].named_parameters()
                if param.requires_grad and 'lora' in name.lower()
            }
            model_state_dict = lora_state_dict
            print_rank_0(f"  Saving {len(lora_state_dict)} LoRA adapter parameters")
        else:
            model_state_dict = model[0].state_dict()

        state_dict = {
            "args": args_namespace,
            "checkpoint_version": 3.0,
            "iteration": iteration,
            "model": model_state_dict,
            "optimizer": optimizer.state_dict() if optimizer else None,
        }

        torch.save(state_dict, checkpoint_path)
        print_rank_0(f"  Saved rank checkpoint: {checkpoint_path}")

    # Write tracker file (only on rank 0)
    if torch.distributed.get_rank() == 0:
        tracker_filename = output_path / "latest_checkpointed_iteration.txt"
        with open(tracker_filename, 'w') as f:
            f.write(str(iteration))
        print_rank_0(f"  Saved tracker file: {tracker_filename}")

    dist.barrier()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Mixtral Supervised Fine-Tuning (SFT) with optional LoRA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model arguments
    parser.add_argument(
        "--hf_model_path",
        type=str,
        default=None,
        help="HuggingFace model path or identifier",
    )
    parser.add_argument(
        "--load",
        type=str,
        default=None,
        help="Path to pre-converted Megatron checkpoint",
    )

    # Data arguments
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to instruction dataset (JSONL format)",
    )
    parser.add_argument(
        "--mock_data",
        action="store_true",
        help="Use mock instruction data for testing",
    )
    parser.add_argument(
        "--mask_instruction",
        action="store_true",
        default=True,
        help="Mask instruction tokens in loss (only compute loss on response)",
    )

    # LoRA arguments
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Use LoRA (Low-Rank Adaptation) for memory-efficient fine-tuning",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=16,
        help="LoRA rank (higher = more capacity, more memory). Only used with --use_lora",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha scaling parameter. Only used with --use_lora",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="LoRA dropout rate. Only used with --use_lora",
    )
    parser.add_argument(
        "--lora_target_modules",
        nargs="+",
        default=["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"],
        help="Target modules for LoRA. Only used with --use_lora",
    )

    # Training arguments
    parser.add_argument(
        "--micro_batch_size",
        type=int,
        default=1,
        help="Micro batch size per GPU",
    )
    parser.add_argument(
        "--global_batch_size",
        type=int,
        default=8,
        help="Global batch size across all GPUs",
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=2048,
        help="Sequence length",
    )
    parser.add_argument(
        "--train_iters",
        type=int,
        default=1000,
        help="Number of training iterations",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    # Parallelism arguments
    parser.add_argument(
        "--tensor_model_parallel_size",
        type=int,
        default=1,
        help="Tensor model parallel size",
    )
    parser.add_argument(
        "--pipeline_model_parallel_size",
        type=int,
        default=1,
        help="Pipeline model parallel size",
    )
    parser.add_argument(
        "--expert_model_parallel_size",
        type=int,
        default=1,
        help="Expert model parallel size",
    )
    parser.add_argument(
        "--context_parallel_size",
        type=int,
        default=1,
        help="Context parallel size",
    )

    # DDP arguments
    parser.add_argument(
        "--use_distributed_optimizer",
        action="store_true",
        help="Use distributed optimizer",
    )
    parser.add_argument(
        "--overlap_grad_reduce",
        action="store_true",
        default=True,
        help="Overlap gradient reduction with computation",
    )
    parser.add_argument(
        "--recompute_activations",
        action="store_true",
        help="Enable selective activation checkpointing to save memory (checkpoints attention and MoE activations)",
    )

    # Optimizer arguments
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate (lower for SFT)")
    parser.add_argument("--min_lr", type=float, default=2e-6, help="Minimum learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="Adam beta2")
    parser.add_argument("--adam_eps", type=float, default=1e-8, help="Adam epsilon")
    parser.add_argument("--clip_grad", type=float, default=1.0, help="Gradient clipping")

    # Precision arguments
    parser.add_argument("--bf16", action="store_true", default=True, help="Use BF16 precision")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 precision")

    # Logging and checkpointing
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output path for checkpoints",
    )
    parser.add_argument("--log_interval", type=int, default=10, help="Logging interval")
    parser.add_argument("--save_interval", type=int, default=500, help="Save interval")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    finetune(args)
