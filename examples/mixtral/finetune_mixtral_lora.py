#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
"""
Mixtral 8x7B LoRA Fine-Tuning Example

This script demonstrates how to fine-tune Mixtral 8x7B using LoRA (Low-Rank Adaptation)
on memory-constrained hardware like 8× A40 48GB GPUs.

LoRA Benefits:
- Freezes base model → No gradients for 47B parameters
- Only trains adapters → Tiny matrices (~10-100MB)
- Can use Adam → Adapter overhead is negligible
- Better convergence → No need for vanilla SGD
- Faster training → Fewer parameters to update

Memory Usage:
- Full SFT: ~34-41GB per GPU (with vanilla SGD)
- LoRA: ~27-28GB per GPU (with Adam!)

Usage:
    # Fine-tune from converted Megatron checkpoint with LoRA
    torchrun --nproc_per_node=8 examples/mixtral/finetune_mixtral_lora.py \
        --load=/tmp/mixtral_tp2_ep4 \
        --data_path=data/instructions.jsonl \
        --output_path=/tmp/mixtral_lora \
        --tensor_model_parallel_size=2 \
        --expert_model_parallel_size=4 \
        --train_iters=1000 \
        --lora_rank=16 \
        --lora_alpha=32

    # With mock data for testing
    torchrun --nproc_per_node=8 examples/mixtral/finetune_mixtral_lora.py \
        --load=/tmp/mixtral_tp2_ep4 \
        --mock_data \
        --output_path=/tmp/mixtral_lora_test \
        --tensor_model_parallel_size=2 \
        --expert_model_parallel_size=4 \
        --train_iters=10
"""

import argparse
import sys
from pathlib import Path

import torch
from megatron.bridge.models.mixtral import MixtralModelProvider
from megatron.bridge.peft.lora import LoRA
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DistributedDataParallelConfig,
    LoggerConfig,
    MockGPTDatasetConfig,
    OptimizerConfig,
    RNGConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainingConfig,
)
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.gpt_step import forward_step


def create_mock_instruction_dataset(seq_length, num_samples=1000):
    """Create a mock instruction-tuning dataset."""

    class MockInstructionDataset(torch.utils.data.Dataset):
        def __init__(self, seq_length, num_samples, vocab_size=32000):
            self.seq_length = seq_length
            self.num_samples = num_samples
            self.vocab_size = vocab_size

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            # Generate random tokens
            input_ids = torch.randint(0, self.vocab_size, (self.seq_length,), dtype=torch.long)

            # Create labels (mask instruction tokens for SFT)
            # In real SFT, you'd mask the instruction part
            labels = input_ids.clone()
            # For demo, mask first 30% as "instruction"
            instruction_len = int(self.seq_length * 0.3)
            labels[:instruction_len] = -100

            return {"input_ids": input_ids, "labels": labels}

    return MockInstructionDataset(seq_length, num_samples)


def main():
    parser = argparse.ArgumentParser(
        description="Mixtral 8x7B LoRA Fine-Tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model and checkpoint arguments
    parser.add_argument(
        "--load",
        type=str,
        required=True,
        help="Path to pretrained Megatron checkpoint to load for LoRA fine-tuning",
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Directory to save LoRA adapter checkpoints"
    )

    # Data arguments
    parser.add_argument("--data_path", type=str, help="Path to training data (JSONL format)")
    parser.add_argument(
        "--mock_data", action="store_true", help="Use mock data for testing (no real dataset required)"
    )
    parser.add_argument("--seq_length", type=int, default=2048, help="Sequence length")

    # Training arguments
    parser.add_argument("--train_iters", type=int, default=1000, help="Number of training iterations")
    parser.add_argument("--global_batch_size", type=int, default=64, help="Global batch size")
    parser.add_argument("--micro_batch_size", type=int, default=1, help="Micro batch size per GPU")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (lower for fine-tuning)")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--clip_grad", type=float, default=1.0, help="Gradient clipping")

    # LoRA arguments
    parser.add_argument(
        "--lora_rank", type=int, default=16, help="LoRA rank (higher = more capacity, more memory)"
    )
    parser.add_argument(
        "--lora_alpha", type=int, default=32, help="LoRA alpha scaling parameter"
    )
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout rate")
    parser.add_argument(
        "--lora_target_modules",
        nargs="+",
        default=["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"],
        help="Target modules for LoRA",
    )

    # Parallelism arguments
    parser.add_argument("--tensor_model_parallel_size", type=int, default=2, help="Tensor model parallel size")
    parser.add_argument(
        "--pipeline_model_parallel_size", type=int, default=1, help="Pipeline model parallel size"
    )
    parser.add_argument("--expert_model_parallel_size", type=int, default=4, help="Expert model parallel size")
    parser.add_argument("--context_parallel_size", type=int, default=1, help="Context parallel size")

    # Optimization arguments
    parser.add_argument("--recompute_activations", action="store_true", help="Enable activation checkpointing")
    parser.add_argument("--use_distributed_optimizer", action="store_true", help="Use distributed optimizer")

    # Checkpoint arguments
    parser.add_argument("--save_interval", type=int, default=200, help="Checkpoint save interval")
    parser.add_argument("--log_interval", type=int, default=10, help="Logging interval")

    # Other arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--bf16", action="store_true", default=True, help="Use bfloat16")
    parser.add_argument("--fp16", action="store_true", help="Use float16")

    args = parser.parse_args()

    # Validate arguments
    if not args.mock_data and not args.data_path:
        parser.error("Either --data_path or --mock_data must be specified")

    # Create Mixtral model provider
    model_provider = MixtralModelProvider(
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        expert_model_parallel_size=args.expert_model_parallel_size,
        context_parallel_size=args.context_parallel_size,
        seq_length=args.seq_length,
        params_dtype=torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32,
        bf16=args.bf16,
        fp16=args.fp16,
        # Sequence parallelism required when using TP > 1 with MoE
        sequence_parallel=args.tensor_model_parallel_size > 1,
        # Enable activation checkpointing for memory efficiency
        recompute_granularity="selective" if args.recompute_activations else None,
        recompute_modules=["core_attn", "moe"] if args.recompute_activations else None,
        # LoRA-specific: Freeze base model
        perform_initialization=False,  # Don't initialize, will load from checkpoint
    )

    # Create LoRA configuration
    lora_config = LoRA(
        target_modules=args.lora_target_modules,
        dim=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )

    # Create training configuration
    training_config = TrainingConfig(
        train_iters=args.train_iters,
        global_batch_size=args.global_batch_size,
        micro_batch_size=args.micro_batch_size,
        eval_interval=args.train_iters,  # No evaluation for now
    )

    # Create optimizer configuration
    # With LoRA, we can use Adam since adapters are tiny!
    optimizer_config = OptimizerConfig(
        optimizer="adam",
        lr=args.lr,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1e-8,
        clip_grad=args.clip_grad,
        bf16=args.bf16,
        fp16=args.fp16,
        use_distributed_optimizer=args.use_distributed_optimizer,
    )

    # Create scheduler configuration
    scheduler_config = SchedulerConfig(
        lr_decay_style="cosine",
        lr_warmup_iters=min(100, args.train_iters // 10),
        lr_decay_iters=args.train_iters,
        start_weight_decay=args.weight_decay,
        end_weight_decay=args.weight_decay,
    )

    # Create DDP configuration
    ddp_config = DistributedDataParallelConfig(
        grad_reduce_in_fp32=True,
        overlap_grad_reduce=args.use_distributed_optimizer,
        use_distributed_optimizer=args.use_distributed_optimizer,
    )

    # Create dataset configuration
    if args.mock_data:
        print("Using mock instruction data for testing")
        dataset_config = MockGPTDatasetConfig(
            random_seed=args.seed,
            sequence_length=args.seq_length,
            reset_position_ids=False,
            reset_attention_mask=False,
            eod_mask_loss=False,
            num_dataset_builder_threads=1,
            dataloader_type="single",
            num_workers=1,
        )
    else:
        print(f"Loading instruction data from: {args.data_path}")
        # For real data, you'd use HFDatasetConfig with a custom processor
        from megatron.bridge.data.builders.hf_dataset import HFDatasetConfig

        dataset_config = HFDatasetConfig(
            dataset_path=args.data_path,
            seq_length=args.seq_length,
            seed=args.seed,
            dataloader_type="single",
            num_workers=2,
        )

    # Create tokenizer configuration
    # For Mixtral, use the HuggingFace tokenizer
    tokenizer_config = TokenizerConfig(
        tokenizer_type="HuggingFaceTokenizer",
        tokenizer_model="mistralai/Mixtral-8x7B-v0.1",
    )

    # Create checkpoint configuration
    # For PEFT: Only set pretrained_checkpoint to load base model weights
    # Don't set load (that's only for resuming from a PEFT checkpoint)
    checkpoint_config = CheckpointConfig(
        pretrained_checkpoint=args.load,  # Load base model weights from here
        save=args.output_path,
        save_interval=args.save_interval,
        ckpt_format="torch_dist",
    )

    # Create logger configuration
    logger_config = LoggerConfig(
        log_interval=args.log_interval,
        tensorboard_dir=str(Path(args.output_path) / "tensorboard"),
    )

    # Create RNG configuration
    rng_config = RNGConfig(seed=args.seed)

    # Create complete configuration
    config = ConfigContainer(
        model=model_provider,
        train=training_config,
        optimizer=optimizer_config,
        scheduler=scheduler_config,
        ddp=ddp_config,
        dataset=dataset_config,
        tokenizer=tokenizer_config,
        checkpoint=checkpoint_config,
        logger=logger_config,
        rng=rng_config,
        peft=lora_config,  # This enables LoRA!
    )

    # Print configuration summary
    print("\n" + "=" * 80)
    print("Mixtral 8x7B LoRA Fine-Tuning Configuration")
    print("=" * 80)
    print(f"Loading checkpoint from: {args.load}")
    print(f"Output path: {args.output_path}")
    print(f"Training iterations: {args.train_iters}")
    print(f"Global batch size: {args.global_batch_size}")
    print(f"Micro batch size: {args.micro_batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"\nLoRA Configuration:")
    print(f"  Rank: {args.lora_rank}")
    print(f"  Alpha: {args.lora_alpha}")
    print(f"  Dropout: {args.lora_dropout}")
    print(f"  Target modules: {args.lora_target_modules}")
    print(f"\nParallelism:")
    print(f"  TP={args.tensor_model_parallel_size}, PP={args.pipeline_model_parallel_size}")
    print(f"  EP={args.expert_model_parallel_size}, CP={args.context_parallel_size}")
    print(f"\nOptimizer: Adam (can use Adam with LoRA!)")
    print(f"Activation checkpointing: {args.recompute_activations}")
    print("=" * 80 + "\n")

    # Run LoRA fine-tuning
    finetune(config, forward_step)

    print("\n" + "=" * 80)
    print("LoRA Fine-Tuning Complete!")
    print(f"Adapter checkpoints saved to: {args.output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
