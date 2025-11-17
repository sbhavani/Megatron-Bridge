#!/usr/bin/env python3
"""
Checkpoint Conversion with Extra State Fix (Monkey Patch)

This script wraps the standard conversion functionality with a fix for the
"24 ShardedObject are missing" error when converting checkpoints with different
parallelism than training.

Usage - same as the original script:
  python convert_checkpoints_fixed.py export \
    --hf-model Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --megatron-path ./checkpoints/my_model \
    --hf-path ./exports/my_model_hf
"""

import sys
from pathlib import Path

# Apply monkey patch BEFORE importing megatron modules
def apply_extra_state_fix():
    """Monkey patch to remove _extra_state before checkpoint loading."""
    try:
        from megatron.bridge.training import checkpointing
        from megatron.bridge.utils.common_utils import print_rank_0
        
        # Save original function
        original_load_fn = checkpointing._load_model_weights_from_checkpoint
        
        def remove_extra_state_from_sharded_dict(sharded_state_dict):
            """Remove _extra_state keys to avoid resharding validation errors."""
            if not isinstance(sharded_state_dict, dict):
                return sharded_state_dict
            
            total_removed = 0
            
            # Handle both single model and pipeline parallel cases
            if "model" in sharded_state_dict:
                target_dicts = [("model", sharded_state_dict["model"])]
            else:
                target_dicts = [(k, v) for k, v in sharded_state_dict.items() 
                               if k.startswith("model")]
            
            for dict_name, target_dict in target_dicts:
                if not hasattr(target_dict, "keys"):
                    continue
                
                keys_to_remove = [key for key in list(target_dict.keys()) 
                                 if "_extra_state" in key]
                
                for key in keys_to_remove:
                    del target_dict[key]
                
                total_removed += len(keys_to_remove)
            
            if total_removed > 0:
                print_rank_0(f"[PATCH] Removed {total_removed} _extra_state entries to avoid resharding issues")
            
            return sharded_state_dict
        
        def patched_load_model_weights(*args, **kwargs):
            """Patched version that removes _extra_state before loading."""
            import torch
            from megatron.core import dist_checkpointing, mpu
            from megatron.core.dist_checkpointing.serialization import get_default_load_sharded_strategy
            from megatron.core.dist_checkpointing.strategies.fully_parallel import FullyParallelLoadStrategyWrapper
            from megatron.core.utils import unwrap_model
            
            checkpoint_path = args[0]
            model = args[1]
            fully_parallel_load = args[2] if len(args) > 2 else kwargs.get('fully_parallel_load', False)
            return_state_dict = args[3] if len(args) > 3 else kwargs.get('return_state_dict', False)
            dist_ckpt_strictness = args[4] if len(args) > 4 else kwargs.get('dist_ckpt_strictness', 'assume_ok_unexpected')
            strict = args[5] if len(args) > 5 else kwargs.get('strict', True)
            
            # Load common state and metadata (same as original)
            state_dict = dist_checkpointing.load_common_state_dict(checkpoint_path)
            assert state_dict is not None
            
            sharded_sd_metadata = dist_checkpointing.load_content_metadata(preloaded_state_dict=state_dict)
            print_rank_0(f"sharded_state_dict metadata loaded from the checkpoint: {sharded_sd_metadata}")
            model_sd_kwargs = dict(metadata=sharded_sd_metadata)
            
            # Restore modelopt state if needed
            try:
                from modelopt.torch.opt.plugins import restore_modelopt_state
                restore_modelopt_state(model, state_dict)
            except ImportError:
                pass
            
            model = unwrap_model(model)
            sharded_state_dict = checkpointing._generate_model_state_dict(model, model_sd_kwargs)
            
            # *** THIS IS THE KEY FIX: Remove _extra_state BEFORE loading ***
            sharded_state_dict = remove_extra_state_from_sharded_dict(sharded_state_dict)
            
            # Continue with normal loading
            load_strategy = get_default_load_sharded_strategy(checkpoint_path)
            if fully_parallel_load:
                load_strategy = FullyParallelLoadStrategyWrapper(
                    load_strategy, mpu.get_data_parallel_group(with_context_parallel=True)
                )
            
            state_dict = dist_checkpointing.load(
                sharded_state_dict, checkpoint_path, load_strategy, strict=dist_ckpt_strictness
            )
            
            # Clean up any remaining extra state in loaded dict
            checkpointing.delete_extra_state(state_dict)
            
            if return_state_dict:
                return state_dict
            
            # Load into model
            if len(model) == 1:
                checkpointing._load_model_state_dict(model[0], state_dict["model"], strict)
            else:
                for i in range(len(model)):
                    model_key = "model%d" % i
                    if model_key not in state_dict:
                        continue
                    checkpointing._load_model_state_dict(model[i], state_dict[model_key], strict)
            
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
        
        # Apply the monkey patch
        checkpointing._load_model_weights_from_checkpoint = patched_load_model_weights
        print("[PATCH] Successfully applied extra_state fix to checkpoint loading")
        
    except Exception as e:
        print(f"[PATCH] Warning: Could not apply patch: {e}")
        print("[PATCH] Proceeding with original code...")

# Apply patch before any imports
apply_extra_state_fix()

# Now import and run the original conversion script
import argparse
import torch
from megatron.bridge import AutoBridge


def validate_path(path: str, must_exist: bool = False) -> Path:
    """Validate and convert string path to Path object."""
    path_obj = Path(path)
    if must_exist and not path_obj.exists():
        raise ValueError(f"Path does not exist: {path}")
    return path_obj


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert string to torch dtype."""
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_str not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype_str}. Supported: {list(dtype_map.keys())}")
    return dtype_map[dtype_str]


def import_hf_to_megatron(
    hf_model: str,
    megatron_path: str,
    torch_dtype: str = None,
    device_map: str = None,
    trust_remote_code: bool = False,
) -> None:
    """Import a HuggingFace model and save it as a Megatron checkpoint."""
    print(f"🔄 Starting import: {hf_model} -> {megatron_path}")

    kwargs = {}
    if torch_dtype:
        kwargs["torch_dtype"] = get_torch_dtype(torch_dtype)
        print(f"   Using torch_dtype: {torch_dtype}")

    if device_map:
        kwargs["device_map"] = device_map
        print(f"   Using device_map: {device_map}")

    if trust_remote_code:
        kwargs["trust_remote_code"] = trust_remote_code
        print(f"   Trust remote code: {trust_remote_code}")

    print(f"📥 Loading HuggingFace model: {hf_model}")
    AutoBridge.import_ckpt(
        hf_model_id=hf_model,
        megatron_path=megatron_path,
        **kwargs,
    )

    print(f"✅ Successfully imported model to: {megatron_path}")

    checkpoint_path = Path(megatron_path)
    if checkpoint_path.exists():
        print("📁 Checkpoint structure:")
        for item in checkpoint_path.iterdir():
            if item.is_dir():
                print(f"   📂 {item.name}/")
            else:
                print(f"   📄 {item.name}")


def export_megatron_to_hf(
    hf_model: str,
    megatron_path: str,
    hf_path: str,
    show_progress: bool = True,
) -> None:
    """Export a Megatron checkpoint to HuggingFace format."""
    print(f"🔄 Starting export: {megatron_path} -> {hf_path}")

    checkpoint_path = validate_path(megatron_path, must_exist=True)
    print(f"📂 Found Megatron checkpoint: {checkpoint_path}")

    # Look for configuration files
    config_files = list(checkpoint_path.glob("**/run_config.yaml"))
    if not config_files:
        iter_dirs = [d for d in checkpoint_path.iterdir() if d.is_dir() and d.name.startswith("iter_")]
        if iter_dirs:
            latest_iter = max(iter_dirs, key=lambda d: int(d.name.replace("iter_", "")))
            config_files = list(latest_iter.glob("run_config.yaml"))

    if config_files:
        print(f"📋 Found configuration: {config_files[0]}")

    bridge = AutoBridge.from_hf_pretrained(hf_model)

    print("📤 Exporting to HuggingFace format...")
    print("    (The patch will remove _extra_state entries to enable resharding)")
    
    bridge.export_ckpt(
        megatron_path=megatron_path,
        hf_path=hf_path,
        show_progress=show_progress,
    )

    print(f"✅ Successfully exported model to: {hf_path}")

    export_path = Path(hf_path)
    if export_path.exists():
        print("📁 Export structure:")
        for item in export_path.iterdir():
            if item.is_dir():
                print(f"   📂 {item.name}/")
            else:
                print(f"   📄 {item.name}")

    print("🔍 You can now load this model with:")
    print("   from transformers import AutoModelForCausalLM")
    print(f"   model = AutoModelForCausalLM.from_pretrained('{hf_path}')")


def main():
    """Main function to handle command line arguments and execute conversions."""
    parser = argparse.ArgumentParser(
        description="Convert models between HuggingFace and Megatron formats (with extra_state fix)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", help="Conversion direction")

    # Import subcommand
    import_parser = subparsers.add_parser("import", help="Import HuggingFace model to Megatron checkpoint format")
    import_parser.add_argument("--hf-model", required=True, help="HuggingFace model ID or path to model directory")
    import_parser.add_argument("--megatron-path", required=True, help="Directory path where the Megatron checkpoint will be saved")
    import_parser.add_argument("--torch-dtype", choices=["float32", "float16", "bfloat16"], help="Model precision")
    import_parser.add_argument("--device-map", help='Device placement strategy (e.g., "auto", "cuda:0")')
    import_parser.add_argument("--trust-remote-code", action="store_true", help="Allow custom model code execution")

    # Export subcommand
    export_parser = subparsers.add_parser("export", help="Export Megatron checkpoint to HuggingFace format")
    export_parser.add_argument("--hf-model", required=True, help="HuggingFace model ID or path to model directory")
    export_parser.add_argument("--megatron-path", required=True, help="Directory path where the Megatron checkpoint is stored")
    export_parser.add_argument("--hf-path", required=True, help="Directory path where the HuggingFace model will be saved")
    export_parser.add_argument("--no-progress", action="store_true", help="Disable progress bar during export")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "import":
        import_hf_to_megatron(
            hf_model=args.hf_model,
            megatron_path=args.megatron_path,
            torch_dtype=args.torch_dtype,
            device_map=args.device_map,
            trust_remote_code=args.trust_remote_code,
        )

    elif args.command == "export":
        export_megatron_to_hf(
            hf_model=args.hf_model,
            megatron_path=args.megatron_path,
            hf_path=args.hf_path,
            show_progress=not args.no_progress,
        )
    else:
        raise RuntimeError(f"Unknown command: {args.command}")

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())
