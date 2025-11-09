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
Mixtral text generation with Megatron-Bridge.

This script provides both single-prompt and interactive generation modes
for Mixtral models, with support for distributed inference.

Examples:
    # Single prompt generation
    python examples/mixtral/generate_text.py \\
        --hf_model_path="mistralai/Mixtral-8x7B-v0.1" \\
        --prompt="What is quantum computing?" \\
        --max_tokens=100

    # Interactive mode
    python examples/mixtral/generate_text.py \\
        --hf_model_path="mistralai/Mixtral-8x7B-v0.1" \\
        --interactive

    # Multi-GPU with expert parallelism
    torchrun --nproc_per_node=2 examples/mixtral/generate_text.py \\
        --hf_model_path="mistralai/Mixtral-8x7B-v0.1" \\
        --expert_model_parallel_size=2 \\
        --interactive

    # With sampling parameters
    python examples/mixtral/generate_text.py \\
        --hf_model_path="mistralai/Mixtral-8x7B-v0.1" \\
        --prompt="Write a story about:" \\
        --temperature=0.8 \\
        --top_p=0.95 \\
        --max_tokens=500
"""

import argparse
import os
import sys
from typing import Optional

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from transformers import AutoTokenizer

from megatron.bridge import AutoBridge
from megatron.bridge.models import MixtralModelProvider
from megatron.bridge.utils.common_utils import get_last_rank, print_rank_0


class SingleBatchIterator:
    """Iterator that yields a single batch for inference."""

    def __init__(self, input_ids, position_ids):
        self.batch = dict(
            tokens=input_ids,
            position_ids=position_ids,
        )
        self._yielded = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._yielded:
            raise StopIteration
        self._yielded = True
        return self.batch


def forward_step(data_iterator, model, **kwargs) -> torch.Tensor:
    """Forward step for text generation."""
    batch = next(data_iterator)
    forward_args = {
        "input_ids": batch["tokens"],
        "position_ids": batch["position_ids"],
        "attention_mask": batch.get("attention_mask", None),
    }

    def loss_func(output_tensor, non_loss_data=False, **kwargs):
        if non_loss_data:
            # For inference with collect_non_loss_data=True, return the output directly
            return output_tensor
        else:
            # For training, return (output, loss_reduced) tuple
            return output_tensor, torch.tensor(0.0, device=output_tensor.device)

    return model(**forward_args), loss_func


def sample_token(logits: torch.Tensor, temperature: float = 1.0, top_p: float = 1.0, top_k: int = 0) -> torch.Tensor:
    """Sample next token from logits with temperature, top-p, and top-k filtering.

    Args:
        logits: Logits tensor of shape [batch_size, vocab_size]
        temperature: Sampling temperature (higher = more random)
        top_p: Nucleus sampling threshold
        top_k: Top-k sampling threshold (0 = disabled)

    Returns:
        Sampled token IDs
    """
    if temperature != 1.0:
        logits = logits / temperature

    # Top-k filtering
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')

    # Top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Keep at least one token
        sorted_indices_to_remove[..., 0] = False
        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = float('-inf')

    # Sample from the filtered distribution
    probs = torch.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)

    return next_token


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
    do_sample: bool = True,
    show_progress: bool = True,
) -> str:
    """Generate text from a prompt using the model.

    Args:
        model: The Megatron model
        tokenizer: The tokenizer
        prompt: Input prompt text
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
        top_k: Top-k sampling threshold
        do_sample: Whether to sample (vs greedy decoding)
        show_progress: Whether to show generation progress

    Returns:
        Generated text
    """
    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()
    generated_ids = input_ids.clone()

    if show_progress and parallel_state.get_data_parallel_rank() == 0:
        try:
            from tqdm import trange
            iterator = trange(max_tokens, desc="Generating", unit="tok")
        except ImportError:
            iterator = range(max_tokens)
            print_rank_0(f"Generating {max_tokens} tokens...")
    else:
        iterator = range(max_tokens)

    for _ in iterator:
        # Prepare batch
        position_ids = torch.arange(
            generated_ids.shape[1], device=generated_ids.device
        ).unsqueeze(0)

        data_iterator = SingleBatchIterator(generated_ids, position_ids)

        # Forward pass
        forward_backward_func = get_forward_backward_func()
        output = forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=iter(data_iterator),
            model=model,
            num_microbatches=1,
            seq_length=generated_ids.shape[1],
            micro_batch_size=1,
            forward_only=True,
            collect_non_loss_data=True,  # Return logits instead of loss
        )

        # Get logits from last pipeline stage
        if parallel_state.is_pipeline_last_stage():
            logits = output[0]  # output is list, first element is the logits tensor
            # Sample or greedy decode
            if do_sample and (temperature != 0.0):
                next_token = sample_token(
                    logits[:, -1, :],
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )
            else:
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        else:
            next_token = torch.zeros((1, 1), dtype=torch.long, device="cuda")

        # Broadcast next token to all ranks
        if parallel_state.get_pipeline_model_parallel_world_size() > 1:
            dist.broadcast(next_token, src=get_last_rank())

        # Check for EOS
        if next_token.item() == tokenizer.eos_token_id:
            break

        # Append to generated sequence
        generated_ids = torch.cat([generated_ids, next_token], dim=1)

    # Decode generated text
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text


def interactive_mode(model, tokenizer, args):
    """Run interactive generation mode."""
    print_rank_0("\n" + "=" * 80)
    print_rank_0("Interactive Mixtral Generation")
    print_rank_0("=" * 80)
    print_rank_0(f"Model: {args.hf_model_path}")
    print_rank_0(f"Parallelism: TP={args.tensor_model_parallel_size}, "
                 f"PP={args.pipeline_model_parallel_size}, "
                 f"EP={args.expert_model_parallel_size}")
    print_rank_0("\nEnter your prompts below. Type 'quit' or 'exit' to stop.")
    print_rank_0("Type 'help' for available commands.\n")

    while True:
        if parallel_state.get_data_parallel_rank() == 0:
            try:
                prompt = input("\n> ")
            except EOFError:
                break
        else:
            prompt = None

        # Broadcast prompt to all ranks
        if dist.is_initialized():
            prompt_list = [prompt]
            dist.broadcast_object_list(prompt_list, src=0)
            prompt = prompt_list[0]

        if not prompt or prompt.lower() in ["quit", "exit"]:
            print_rank_0("\nExiting interactive mode...")
            break

        if prompt.lower() == "help":
            print_rank_0("\nAvailable commands:")
            print_rank_0("  help                - Show this help message")
            print_rank_0("  quit/exit          - Exit interactive mode")
            print_rank_0("  set temp <value>   - Set temperature (e.g., 'set temp 0.8')")
            print_rank_0("  set topp <value>   - Set top_p (e.g., 'set topp 0.95')")
            print_rank_0("  set maxtokens <n>  - Set max tokens (e.g., 'set maxtokens 200')")
            print_rank_0("  show settings      - Display current generation settings")
            continue

        if prompt.lower().startswith("set "):
            # Handle settings commands
            parts = prompt.split()
            if len(parts) >= 3:
                setting = parts[1].lower()
                try:
                    if setting == "temp":
                        args.temperature = float(parts[2])
                        print_rank_0(f"Temperature set to {args.temperature}")
                    elif setting == "topp":
                        args.top_p = float(parts[2])
                        print_rank_0(f"Top-p set to {args.top_p}")
                    elif setting == "maxtokens":
                        args.max_tokens = int(parts[2])
                        print_rank_0(f"Max tokens set to {args.max_tokens}")
                except ValueError:
                    print_rank_0(f"Invalid value for {setting}")
            continue

        if prompt.lower() == "show settings":
            print_rank_0(f"\nCurrent settings:")
            print_rank_0(f"  Temperature: {args.temperature}")
            print_rank_0(f"  Top-p: {args.top_p}")
            print_rank_0(f"  Top-k: {args.top_k}")
            print_rank_0(f"  Max tokens: {args.max_tokens}")
            continue

        # Generate response
        print_rank_0("\nGenerating response...")
        generated_text = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            do_sample=(args.temperature != 0.0),
            show_progress=False,
        )

        print_rank_0(f"\n{generated_text}\n")


def main(args) -> None:
    """Main function for Mixtral text generation."""

    # Validate arguments
    if not args.hf_model_path and not args.load:
        raise ValueError("Either --hf_model_path or --load must be provided")

    # Initialize distributed environment
    if not dist.is_initialized():
        dist.init_process_group("nccl")

    # Set CUDA device for this rank
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    # Note: provide_distributed_model() handles model parallelism initialization

    if args.load:
        # Load from pre-converted Megatron checkpoint
        print_rank_0(f"Loading Mixtral model from checkpoint: {args.load}...")

        # Create provider directly with Mixtral 8x7B config
        provider = MixtralModelProvider(
            tensor_model_parallel_size=args.tensor_model_parallel_size,
            pipeline_model_parallel_size=args.pipeline_model_parallel_size,
            expert_model_parallel_size=args.expert_model_parallel_size,
        )

        # Skip weight initialization since we're loading from checkpoint
        provider.perform_initialization = False

        print_rank_0(f"\nModel configuration:")
        print_rank_0(f"  Hidden size: {provider.hidden_size}")
        print_rank_0(f"  Num layers: {provider.num_layers}")
        print_rank_0(f"  Num experts: {provider.num_moe_experts}")
        print_rank_0(f"  Router top-k: {provider.moe_router_topk}")
        print_rank_0(f"  Parallelism: TP={args.tensor_model_parallel_size}, "
                     f"PP={args.pipeline_model_parallel_size}, "
                     f"EP={args.expert_model_parallel_size}\n")

        # Finalize provider configuration
        provider.finalize()

        # Create model (provide_distributed_model handles parallelism initialization)
        model = provider.provide_distributed_model(
            ddp_config=None,
            wrap_with_ddp=False,
        )

        # Load checkpoint
        print_rank_0(f"Loading weights from {args.load}...")
        from pathlib import Path

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
                # Maybe it's an old format without iteration dir? Try loading directly
                print_rank_0(f"  No iteration directory found, trying old format...")
                iteration_dir = checkpoint_base

        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        ep_rank = parallel_state.get_expert_model_parallel_rank()

        checkpoint_name = f"mp_rank_{tp_rank:02d}_{pp_rank:03d}_{ep_rank:03d}"
        checkpoint_path = iteration_dir / checkpoint_name / "model_optim_rng.pt"

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model[0].load_state_dict(checkpoint['model'])
        print_rank_0(f"✓ Loaded checkpoint from {checkpoint_path}")

        # Set to evaluation mode for inference
        model[0].eval()

    else:
        # Load from HuggingFace
        print_rank_0(f"Loading Mixtral model from HuggingFace: {args.hf_model_path}...")

        # Load model using AutoBridge
        bridge = AutoBridge.from_hf_pretrained(
            args.hf_model_path,
            trust_remote_code=args.trust_remote_code,
        )

        # Convert to Megatron provider
        provider = bridge.to_megatron_provider()

        # Override parallelism settings
        provider.expert_model_parallel_size = args.expert_model_parallel_size
        provider.tensor_model_parallel_size = args.tensor_model_parallel_size
        provider.pipeline_model_parallel_size = args.pipeline_model_parallel_size

        print_rank_0(f"\nModel configuration:")
        print_rank_0(f"  Hidden size: {provider.hidden_size}")
        print_rank_0(f"  Num layers: {provider.num_layers}")
        print_rank_0(f"  Num experts: {provider.num_moe_experts}")
        print_rank_0(f"  Router top-k: {provider.moe_router_topk}")
        print_rank_0(f"  Parallelism: TP={args.tensor_model_parallel_size}, "
                     f"PP={args.pipeline_model_parallel_size}, "
                     f"EP={args.expert_model_parallel_size}\n")

        # Finalize provider configuration
        provider.finalize()

        # Load distributed model (provide_distributed_model handles parallelism initialization)
        model = provider.provide_distributed_model(
            ddp_config=None,
            wrap_with_ddp=False,
        )

        # Set to evaluation mode for inference
        model[0].eval()

    # Load tokenizer
    tokenizer_path = args.hf_model_path or "mistralai/Mixtral-8x7B-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=args.trust_remote_code,
    )

    # Run interactive or single-prompt mode
    if args.interactive:
        interactive_mode(model, tokenizer, args)
    else:
        if not args.prompt:
            print_rank_0("Error: --prompt required for non-interactive mode")
            sys.exit(1)

        print_rank_0(f"Prompt: {args.prompt}\n")
        print_rank_0("Generating...")

        generated_text = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            do_sample=(args.temperature != 0.0),
            show_progress=True,
        )

        print_rank_0(f"\nGenerated text:\n{generated_text}\n")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Mixtral text generation with Megatron-Bridge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model arguments
    parser.add_argument(
        "--hf_model_path",
        type=str,
        default=None,
        help="HuggingFace model path or identifier (for loading from HF)",
    )
    parser.add_argument(
        "--load",
        type=str,
        default=None,
        help="Path to Megatron checkpoint directory (for loading pre-converted checkpoint)",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Trust remote code when loading from HuggingFace",
    )

    # Generation arguments
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Text prompt for generation (ignored in interactive mode)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (0.0 for greedy decoding)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Nucleus sampling threshold",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=0,
        help="Top-k sampling threshold (0 to disable)",
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

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
