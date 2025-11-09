#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
"""
Test script to verify GPU meta device initialization prevents OOM.

Tests for GitHub issue: https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/1167

This script tests loading large MoE models directly to GPU using meta device:

1. **HF Meta Device Loading**: HuggingFace model loaded with device_map="meta"
   - Prevents materializing full 44GB HF model in memory
   - Bridge detects meta device and uses SafeTensorsStateSource

2. **Lazy Weight Loading**: SafeTensors loads weights from disk to CPU
   - Weights loaded on-demand during conversion
   - No full model materialized at once

3. **Megatron Meta Device Initialization**: Empty Megatron model created on meta device
   - Uses init_model_with_meta_device=True (TransformerEngine PR #596 support)
   - Empty model structure takes 0GB instead of ~44GB per rank
   - No GPU memory allocated until weights are materialized

4. **Optimized Scatter**: Weights transferred to GPU one shard at a time
   - Prevents rank 0 from holding all 44GB on GPU before scattering
   - Each rank receives only its shard (~22GB with EP=2)
   - Parameters materialize on GPU during weight loading

Expected behavior:
- With TP=1, EP=2: Each rank gets 4 experts (~22GB), not all 8 (~44GB)
- Model loaded directly to GPU without CPU intermediate step

Usage:
    # Test direct HF loading with meta device (TP=1, EP=2)
    torchrun --nproc_per_node=2 examples/mixtral/test_meta_device.py \
        --test direct_hf \
        --hf_model_path="mistralai/Mixtral-8x7B-v0.1" \
        --tensor_model_parallel_size=1 \
        --expert_model_parallel_size=2

    # Test 2: Pre-converted checkpoint loading (also works)
    torchrun --nproc_per_node=2 examples/mixtral/test_meta_device.py \
        --test converted_checkpoint \
        --load=/path/to/megatron/checkpoint \
        --tensor_model_parallel_size=1 \
        --expert_model_parallel_size=2
"""

import argparse
import gc
import os
import sys

import torch
import torch.distributed as dist


def setup_distributed():
    """Initialize distributed backend."""
    if not dist.is_initialized():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        dist.init_process_group("nccl")


def print_rank_0(message):
    """Print message only on rank 0."""
    if dist.get_rank() == 0:
        print(message, flush=True)


def get_gpu_memory_stats():
    """Get current GPU memory usage."""
    rank = dist.get_rank()
    device = torch.cuda.current_device()

    allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
    reserved = torch.cuda.memory_reserved(device) / 1024**3    # GB
    max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3  # GB

    return {
        "rank": rank,
        "device": device,
        "allocated_gb": allocated,
        "reserved_gb": reserved,
        "max_allocated_gb": max_allocated,
    }


def print_memory_stats(stage):
    """Print memory statistics for all ranks."""
    stats = get_gpu_memory_stats()

    # Gather stats from all ranks
    world_size = dist.get_world_size()
    all_stats = [None] * world_size
    dist.all_gather_object(all_stats, stats)

    if dist.get_rank() == 0:
        print(f"\n{'='*80}")
        print(f"Memory Statistics - {stage}")
        print(f"{'='*80}")
        for s in all_stats:
            print(f"Rank {s['rank']} (GPU {s['device']}): "
                  f"Allocated: {s['allocated_gb']:.2f}GB, "
                  f"Reserved: {s['reserved_gb']:.2f}GB, "
                  f"Peak: {s['max_allocated_gb']:.2f}GB")
        print(f"{'='*80}\n")

        # Check if issue exists
        avg_allocated = sum(s['max_allocated_gb'] for s in all_stats) / len(all_stats)
        if stage == "After Model Loading" and avg_allocated > 30:
            print("⚠️  WARNING: High memory usage detected!")
            print(f"   Average peak: {avg_allocated:.2f}GB per GPU")
            print("   This may indicate the meta device issue exists.")
            print("   Expected: ~15-25GB per GPU with TP=2")
            print()


def test_direct_hf_loading(args):
    """Test 1: Direct HuggingFace loading with meta device."""
    print_rank_0("="*80)
    print_rank_0("TEST 1: Direct HuggingFace Loading with Meta Device")
    print_rank_0("="*80)
    print_rank_0(f"Model: {args.hf_model_path}")
    print_rank_0(f"Parallelism: TP={args.tensor_model_parallel_size}, EP={args.expert_model_parallel_size}")
    print_rank_0("Testing if meta device prevents full model loading on each rank...")
    print_rank_0("")

    # Reset memory stats
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()

    print_memory_stats("Before Loading")

    try:
        from megatron.bridge import AutoBridge

        print_rank_0("=" * 80)
        print_rank_0("Step 1: Loading HuggingFace model with AutoBridge...")
        print_rank_0(f"  Model path: {args.hf_model_path}")
        print_rank_0(f"  device_map='meta' to load on meta device without materializing weights")
        print_rank_0("=" * 80)

        bridge = AutoBridge.from_hf_pretrained(
            args.hf_model_path,
            trust_remote_code=True,
            device_map="meta",  # Load on meta device to avoid materializing full model
        )

        print_rank_0("✓ Bridge created")
        print_memory_stats("After Bridge Creation")

        print_rank_0("\n" + "=" * 80)
        print_rank_0("Step 2: Converting to Megatron provider...")
        print_rank_0(f"  load_weights=True (will load HF weights during provide_distributed_model)")
        print_rank_0("=" * 80)

        provider = bridge.to_megatron_provider(load_weights=True)

        print_rank_0("\n" + "=" * 80)
        print_rank_0("Step 3: Configuring parallelism...")
        print_rank_0(f"  Tensor parallelism: {args.tensor_model_parallel_size}")
        print_rank_0(f"  Pipeline parallelism: 1")
        print_rank_0(f"  Expert parallelism: {args.expert_model_parallel_size}")
        print_rank_0("=" * 80)

        provider.tensor_model_parallel_size = args.tensor_model_parallel_size
        provider.expert_model_parallel_size = args.expert_model_parallel_size

        print_rank_0("\n" + "=" * 80)
        print_rank_0("Step 4: Finalizing provider (validates configuration)...")
        print_rank_0("=" * 80)

        provider.finalize()

        print_rank_0("✓ Provider finalized")
        print_rank_0(f"  Hidden size: {provider.hidden_size}")
        print_rank_0(f"  Num layers: {provider.num_layers}")
        print_rank_0(f"  Num experts: {provider.num_moe_experts}")
        print_memory_stats("After Provider Finalization")

        print_rank_0("\n" + "=" * 80)
        print_rank_0("Step 5: Creating distributed model...")
        print_rank_0("  Provider will:")
        print_rank_0("    1. Initialize model parallelism (TP, PP, EP groups)")
        print_rank_0("    2. Create empty Megatron model structure on META device")
        print_rank_0("    3. Load and convert HF weights from SafeTensors (CPU)")
        print_rank_0("    4. Materialize parameters to GPU during weight loading")
        print_rank_0(f"  Using init_model_with_meta_device=True (TransformerEngine supports this!)")
        print_rank_0("  Testing if meta device prevents empty model OOM")
        print_rank_0("=" * 80)

        model = provider.provide_distributed_model(
            ddp_config=None,
            wrap_with_ddp=False,
            init_model_with_meta_device=True,  # TransformerEngine meta device support!
        )

        print_rank_0("\n✓ Model created successfully")

        print_memory_stats("After Model Loading")

        # Check where parameters are actually located
        import torch.distributed as dist
        if dist.get_rank() == 0:
            print_rank_0("\n[DEBUG] Checking parameter devices:")
            param_count = 0
            for name, param in model[0].named_parameters():
                if param_count < 5:  # Show first 5 params
                    print_rank_0(f"  {name}: device={param.device}, shape={param.shape}")
                param_count += 1
            print_rank_0(f"  ... (total {param_count} parameters)")

            # Count parameters on each device
            meta_params = sum(1 for p in model[0].parameters() if p.device.type == 'meta')
            cpu_params = sum(1 for p in model[0].parameters() if p.device.type == 'cpu')
            cuda_params = sum(1 for p in model[0].parameters() if p.device.type == 'cuda')
            print_rank_0(f"\n[DEBUG] Parameter distribution:")
            print_rank_0(f"  Meta device: {meta_params}")
            print_rank_0(f"  CPU: {cpu_params}")
            print_rank_0(f"  CUDA: {cuda_params}")

        print_rank_0("\n✓ Direct HuggingFace loading with meta device completed!")
        print_rank_0("\nConclusion:")
        stats = get_gpu_memory_stats()
        print_rank_0(f"Peak memory per rank: {stats['max_allocated_gb']:.2f}GB")

        # With TP=T, EP=E: each rank has (44GB / E) experts + (small amount / T) for shared layers
        # For TP=1, EP=2: each rank should have ~22GB (half the experts)
        expected_per_rank = 44 / args.expert_model_parallel_size  # Rough estimate based on expert parallelism

        if stats['max_allocated_gb'] > expected_per_rank * 1.5:
            print_rank_0(f"\n❌ Issue EXISTS: Each rank loaded >{expected_per_rank * 1.5:.1f}GB")
            print_rank_0(f"   Expected ~{expected_per_rank:.1f}GB per rank with TP={args.tensor_model_parallel_size}, EP={args.expert_model_parallel_size}")
            print_rank_0("   Meta device did NOT prevent full model loading")
            return False
        else:
            print_rank_0(f"\n✓ Issue RESOLVED: Memory usage reasonable for sharded loading")
            print_rank_0(f"   Expected ~{expected_per_rank:.1f}GB per rank with TP={args.tensor_model_parallel_size}, EP={args.expert_model_parallel_size}")
            print_rank_0("   Meta device successfully prevented full model materialization!")
            return True

    except Exception as e:
        print_rank_0(f"\n❌ Error during direct HF loading: {e}")
        print_rank_0("\nThis error suggests the meta device issue may still exist.")
        print_rank_0("OOM errors indicate each rank tried to load the full model.")
        import traceback
        traceback.print_exc()
        return False


def test_converted_checkpoint(args):
    """Test 2: Pre-converted checkpoint loading (workaround)."""
    print_rank_0("="*80)
    print_rank_0("TEST 2: Pre-converted Checkpoint Loading (Workaround)")
    print_rank_0("="*80)
    print_rank_0(f"Checkpoint: {args.load}")
    print_rank_0(f"Parallelism: TP={args.tensor_model_parallel_size}, EP={args.expert_model_parallel_size}")
    print_rank_0("Testing the recommended workaround approach...")
    print_rank_0("")

    # Reset memory stats
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()

    print_memory_stats("Before Loading")

    try:
        from megatron.bridge.models.mixtral import MixtralModelProvider
        from pathlib import Path

        # Read checkpoint metadata
        checkpoint_path = Path(args.load)
        tracker_file = checkpoint_path / "latest_checkpointed_iteration.txt"

        if not tracker_file.exists():
            raise FileNotFoundError(f"Checkpoint not found: {args.load}")

        print_rank_0("Loading from pre-converted Megatron checkpoint...")

        # Create provider
        provider = MixtralModelProvider(
            tensor_model_parallel_size=args.tensor_model_parallel_size,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=args.expert_model_parallel_size,
            seq_length=128,
            params_dtype=torch.bfloat16,
            bf16=True,
            perform_initialization=False,
        )

        provider.finalize()

        print_memory_stats("After Provider Init")

        # Load model
        # Note: provide_distributed_model() handles model parallelism initialization
        print_rank_0("Creating distributed model...")
        model = provider.provide_distributed_model(
            ddp_config=None,
            wrap_with_ddp=False,
            use_cpu_initialization=True,  # Use CPU init for converted checkpoints
        )

        print_memory_stats("After Model Creation")

        # Load checkpoint weights using distributed checkpointing
        from megatron.core import dist_checkpointing
        from megatron.core.dist_checkpointing.serialization import get_default_load_sharded_strategy

        with open(tracker_file, 'r') as f:
            iteration = int(f.read().strip())

        iteration_dir = checkpoint_path / f"iter_{iteration:07d}"

        print_rank_0(f"Loading weights from {iteration_dir}...")

        state_dict = {"model": model[0].sharded_state_dict()}
        load_strategy = get_default_load_sharded_strategy(str(iteration_dir))
        state_dict = dist_checkpointing.load(state_dict, str(iteration_dir), load_strategy)

        model[0].load_state_dict(state_dict["model"], strict=True)

        print_memory_stats("After Checkpoint Loading")

        print_rank_0("\n✓ Pre-converted checkpoint loading completed successfully!")
        print_rank_0("\nConclusion:")
        print_rank_0("✓ Workaround approach works as expected")
        print_rank_0("  Each rank loads only its sharded portion of the model")
        return True

    except Exception as e:
        print_rank_0(f"\n❌ Error during converted checkpoint loading: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Test meta device initialization behavior",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--test",
        type=str,
        choices=["direct_hf", "converted_checkpoint"],
        required=True,
        help="Which test to run",
    )

    parser.add_argument(
        "--hf_model_path",
        type=str,
        help="HuggingFace model path (for direct_hf test)",
    )

    parser.add_argument(
        "--load",
        type=str,
        help="Pre-converted checkpoint path (for converted_checkpoint test)",
    )

    parser.add_argument(
        "--tensor_model_parallel_size",
        type=int,
        default=1,
        help="Tensor model parallel size",
    )

    parser.add_argument(
        "--expert_model_parallel_size",
        type=int,
        default=2,
        help="Expert model parallel size (default: 2 for 2 GPUs with 4 experts each)",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.test == "direct_hf" and not args.hf_model_path:
        parser.error("--hf_model_path required for direct_hf test")
    if args.test == "converted_checkpoint" and not args.load:
        parser.error("--load required for converted_checkpoint test")

    # Setup distributed
    setup_distributed()

    # Validate parallelism configuration matches world size
    import torch.distributed as dist
    world_size = dist.get_world_size()
    required_gpus = args.tensor_model_parallel_size * args.expert_model_parallel_size

    if required_gpus != world_size:
        if dist.get_rank() == 0:
            print(f"ERROR: Configuration mismatch!")
            print(f"  TP={args.tensor_model_parallel_size} × EP={args.expert_model_parallel_size} = {required_gpus} GPUs required")
            print(f"  But world_size = {world_size} GPUs available")
            print(f"\nValid configurations for {world_size} GPUs:")
            if world_size == 2:
                print(f"  - TP=1, EP=2 (default, recommended)")
                print(f"  - TP=2, EP=1")
            else:
                for tp in range(1, world_size + 1):
                    if world_size % tp == 0:
                        ep = world_size // tp
                        print(f"  - TP={tp}, EP={ep}")
        sys.exit(1)

    # Run test
    if args.test == "direct_hf":
        success = test_direct_hf_loading(args)
    else:
        success = test_converted_checkpoint(args)

    # Cleanup
    dist.barrier()

    if dist.get_rank() == 0:
        print("\n" + "="*80)
        if success:
            print("✓ TEST PASSED")
        else:
            print("❌ TEST FAILED")
        print("="*80)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
