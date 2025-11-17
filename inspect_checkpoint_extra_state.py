#!/usr/bin/env python3
"""
Inspect what's actually in the _extra_state of your checkpoint.
This helps understand why it exists even with BF16 training.
"""

import sys
from pathlib import Path

def inspect_checkpoint(checkpoint_path: str):
    """Inspect checkpoint structure and extra_state contents."""
    print(f"Inspecting checkpoint: {checkpoint_path}\n")
    
    try:
        from megatron.core import dist_checkpointing
        
        # Load common state
        print("Loading checkpoint metadata...")
        common_state = dist_checkpointing.load_common_state_dict(checkpoint_path)
        
        print(f"Checkpoint keys: {list(common_state.keys())}\n")
        
        # Look for _extra_state
        extra_state_keys = []
        for key in common_state.keys():
            if '_extra_state' in str(key):
                extra_state_keys.append(key)
        
        if not extra_state_keys:
            print("❌ No _extra_state found in checkpoint")
            return
        
        print(f"✅ Found {len(extra_state_keys)} _extra_state entries:\n")
        
        for i, key in enumerate(extra_state_keys[:5]):  # Show first 5
            print(f"{i+1}. Key: {key}")
            try:
                value = common_state.get(key)
                print(f"   Type: {type(value)}")
                print(f"   Value: {value}")
            except Exception as e:
                print(f"   Could not load value: {e}")
            print()
        
        if len(extra_state_keys) > 5:
            print(f"... and {len(extra_state_keys) - 5} more _extra_state entries\n")
        
        # Load metadata to see sharding info
        print("\nLoading checkpoint metadata...")
        metadata = dist_checkpointing.load_content_metadata(checkpoint_path)
        print(f"Metadata: {metadata}\n")
        
        # Check if this is from TransformerEngine
        print("Checking for TransformerEngine artifacts...")
        te_keys = [k for k in common_state.keys() if 'core_attention' in str(k)]
        print(f"Found {len(te_keys)} core_attention-related keys")
        
        if te_keys:
            print("Sample core_attention keys:")
            for key in te_keys[:3]:
                print(f"  - {key}")
        
    except Exception as e:
        print(f"Error inspecting checkpoint: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_checkpoint_extra_state.py <checkpoint_path>")
        print("\nExample:")
        print("  python inspect_checkpoint_extra_state.py ./checkpoints/my_model/iter_0001000")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    
    if not Path(checkpoint_path).exists():
        print(f"Error: Checkpoint path does not exist: {checkpoint_path}")
        sys.exit(1)
    
    inspect_checkpoint(checkpoint_path)
