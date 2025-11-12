#!/usr/bin/env python
"""
Diagnostic script to examine Megatron checkpoint structure.

This script helps diagnose the checkpoint compatibility issue between
Megatron Bridge and Megatron-LM by inspecting the checkpoint contents.
"""

import argparse
import os
import sys
from pathlib import Path
import torch


def inspect_common_pt(checkpoint_dir):
    """Inspect the common.pt file in a checkpoint directory."""
    common_path = os.path.join(checkpoint_dir, "common.pt")

    if not os.path.exists(common_path):
        print(f"❌ common.pt not found at: {common_path}")
        return None

    print(f"\n{'='*80}")
    print(f"📄 Inspecting common.pt from: {checkpoint_dir}")
    print(f"{'='*80}\n")

    try:
        common_dict = torch.load(common_path, map_location='cpu')

        print(f"✅ Successfully loaded common.pt")
        print(f"\n📊 Top-level keys in common.pt:")
        for key in common_dict.keys():
            value = common_dict[key]
            value_type = type(value).__name__

            if isinstance(value, dict):
                print(f"  - {key}: {value_type} with {len(value)} keys")
                if len(value) > 0 and len(value) < 20:
                    print(f"    Subkeys: {list(value.keys())[:10]}")
            elif isinstance(value, torch.Tensor):
                print(f"  - {key}: Tensor {value.shape}")
            else:
                print(f"  - {key}: {value_type}")

        # Check for 'args' key
        if 'args' in common_dict:
            print(f"\n✅ 'args' key found in checkpoint")
            args = common_dict['args']
            if hasattr(args, 'tensor_model_parallel_size'):
                print(f"   - tensor_model_parallel_size: {args.tensor_model_parallel_size}")
            if hasattr(args, 'pipeline_model_parallel_size'):
                print(f"   - pipeline_model_parallel_size: {args.pipeline_model_parallel_size}")
        else:
            print(f"\n❌ 'args' key NOT found in checkpoint")

        # Inspect model structure
        if 'model' in common_dict:
            print(f"\n📦 Model structure in common.pt:")
            model_dict = common_dict['model']

            if isinstance(model_dict, dict):
                print(f"  Type: dict with {len(model_dict)} keys")

                # Show first level keys
                print(f"\n  First-level keys in model:")
                for k in list(model_dict.keys())[:20]:
                    v = model_dict[k]
                    if isinstance(v, torch.Tensor):
                        print(f"    - {k}: Tensor {v.shape}")
                    elif isinstance(v, dict):
                        print(f"    - {k}: dict with {len(v)} keys")
                        # Show second level for structure understanding
                        if len(v) < 10:
                            for k2 in list(v.keys())[:5]:
                                v2 = v[k2]
                                if isinstance(v2, torch.Tensor):
                                    print(f"        - {k2}: Tensor {v2.shape}")
                                else:
                                    print(f"        - {k2}: {type(v2).__name__}")
                    else:
                        print(f"    - {k}: {type(v).__name__}")

                if len(model_dict) > 20:
                    print(f"    ... and {len(model_dict) - 20} more keys")
            else:
                print(f"  Type: {type(model_dict).__name__}")

        return common_dict

    except Exception as e:
        print(f"❌ Error loading common.pt: {e}")
        import traceback
        traceback.print_exc()
        return None


def inspect_checkpoint_files(checkpoint_dir):
    """List all files in the checkpoint directory."""
    print(f"\n{'='*80}")
    print(f"📁 Files in checkpoint directory: {checkpoint_dir}")
    print(f"{'='*80}\n")

    if not os.path.exists(checkpoint_dir):
        print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        return

    all_files = []
    for root, dirs, files in os.walk(checkpoint_dir):
        for file in files:
            rel_path = os.path.relpath(os.path.join(root, file), checkpoint_dir)
            file_size = os.path.getsize(os.path.join(root, file))
            all_files.append((rel_path, file_size))

    # Sort by name
    all_files.sort()

    print(f"Total files: {len(all_files)}\n")

    # Group by file type
    pt_files = [f for f in all_files if f[0].endswith('.pt')]
    distcp_files = [f for f in all_files if f[0].endswith('.distcp')]
    other_files = [f for f in all_files if not f[0].endswith('.pt') and not f[0].endswith('.distcp')]

    if pt_files:
        print(f"📄 .pt files ({len(pt_files)}):")
        for name, size in pt_files[:10]:
            print(f"  - {name} ({size / 1024:.1f} KB)")
        if len(pt_files) > 10:
            print(f"  ... and {len(pt_files) - 10} more .pt files")

    if distcp_files:
        print(f"\n📄 .distcp files ({len(distcp_files)}):")
        for name, size in distcp_files[:10]:
            print(f"  - {name} ({size / 1024:.1f} KB)")
        if len(distcp_files) > 10:
            print(f"  ... and {len(distcp_files) - 10} more .distcp files")

    if other_files:
        print(f"\n📄 Other files ({len(other_files)}):")
        for name, size in other_files:
            print(f"  - {name} ({size / 1024:.1f} KB)")


def compare_with_expected_structure():
    """Show the expected structure for Megatron-LM compatibility."""
    print(f"\n{'='*80}")
    print(f"✨ Expected Structure for Megatron-LM Compatibility")
    print(f"{'='*80}\n")

    print("""
For a checkpoint to be compatible with Megatron-LM, it should have:

1. common.pt file containing:
   - 'args': SimpleNamespace with training configuration
     * tensor_model_parallel_size
     * pipeline_model_parallel_size
     * world_size
     * data_parallel_size
     * no_save_optim, no_save_rng, ckpt_fully_parallel_save

   - 'checkpoint_version': float (e.g., 3.0)

   - 'iteration': int (training step number)

   - 'model': Can be empty dict {} if all weights are sharded,
              OR should contain the same structure as the sharded files
              for non-sharded weights (embeddings, final norms, etc.)

2. Sharded checkpoint files (*.distcp or organized directories)
   containing ShardedTensor data for model weights

3. Proper metadata files describing the sharding strategy

Common issues:
- Missing 'args' key → Fixed by our recent commit
- Structure mismatch between common.pt and sharded files
- Different sharding strategies between save and load
    """)


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose Megatron checkpoint structure",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "checkpoint_dir",
        type=str,
        help="Path to the checkpoint directory to inspect"
    )
    parser.add_argument(
        "--show-expected",
        action="store_true",
        help="Show expected structure for Megatron-LM compatibility"
    )

    args = parser.parse_args()

    print("\n" + "="*80)
    print("🔍 Megatron Checkpoint Diagnostic Tool")
    print("="*80)

    # Inspect checkpoint files
    inspect_checkpoint_files(args.checkpoint_dir)

    # Inspect common.pt
    common_dict = inspect_common_pt(args.checkpoint_dir)

    # Show expected structure
    if args.show_expected:
        compare_with_expected_structure()

    # Provide recommendations
    print(f"\n{'='*80}")
    print("💡 Recommendations")
    print(f"{'='*80}\n")

    if common_dict is not None:
        has_args = 'args' in common_dict
        has_model = 'model' in common_dict

        if not has_args:
            print("⚠️  Missing 'args' key - this should be fixed by recent commit")
            print("   Make sure you're using the updated checkpointing.py")

        if has_model and isinstance(common_dict.get('model'), dict):
            model_keys = list(common_dict['model'].keys())
            if len(model_keys) > 0:
                print(f"ℹ️  common.pt contains {len(model_keys)} model keys")
                print(f"   This might indicate some weights are not properly sharded")
                print(f"   First few keys: {model_keys[:5]}")

        print("\n✅ Next steps:")
        print("   1. Ensure the latest changes are applied (with 'args' key)")
        print("   2. Re-save the checkpoint from Megatron Bridge")
        print("   3. Try loading in Megatron-LM again")
        print("   4. If issues persist, share the output of this script")


if __name__ == "__main__":
    main()
