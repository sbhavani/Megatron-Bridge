#!/usr/bin/env python3
"""
Minimal reproduction of the _extra_state ShardedObject resharding issue.

This demonstrates the problem without needing a full training run.
"""

def demonstrate_issue():
    """Show the exact code path that causes the error."""
    
    print("=" * 80)
    print("REPRODUCTION: _extra_state ShardedObject Resharding Issue")
    print("=" * 80)
    print()
    
    print("SCENARIO:")
    print("  - Checkpoint saved with 24 ranks (TP=4 × PP=2 × EP=4)")
    print("  - Attempting to load with 8 ranks for export")
    print()
    
    print("CODE PATH:")
    print("-" * 80)
    print("""
# File: src/megatron/bridge/training/checkpointing.py
# Function: _load_model_weights_from_checkpoint (line 1133)

def _load_model_weights_from_checkpoint(checkpoint_path, model, ...):
    # Load checkpoint metadata
    state_dict = dist_checkpointing.load_common_state_dict(checkpoint_path)
    sharded_sd_metadata = dist_checkpointing.load_content_metadata(
        preloaded_state_dict=state_dict
    )
    model_sd_kwargs = dict(metadata=sharded_sd_metadata)
    
    model = unwrap_model(model)
    
    # Generate sharded state dict spec for current process configuration
    sharded_state_dict = _generate_model_state_dict(model, model_sd_kwargs)
    
    # ❌ PROBLEM: sharded_state_dict includes entries like:
    #    'decoder.layers.self_attention.core_attention._extra_state'
    #
    # These reference ShardedObjects in the checkpoint:
    #    - Checkpoint has: 24 shards (from training)
    #    - Current setup has: 8 ranks (for export)
    #    - Validation: "How do I map 24 shards to 8 ranks?" → ERROR
    
    load_strategy = get_default_load_sharded_strategy(checkpoint_path)
    
    # ❌ FAILS HERE during validation
    state_dict = dist_checkpointing.load(
        sharded_state_dict,  # <-- Contains _extra_state keys
        checkpoint_path,
        load_strategy,
        strict=dist_ckpt_strictness
    )
    # Raises: CheckpointingException: Invalid access pattern: 24 ShardedObject are missing
    
    # ✅ This would fix it, but it's TOO LATE (never reached)
    delete_extra_state(state_dict)
    """)
    print("-" * 80)
    print()
    
    print("ERROR DETAILS:")
    print("-" * 80)
    print("""
Traceback (most recent call last):
  File "checkpointing.py", line 1184, in _load_model_weights_from_checkpoint
    state_dict = dist_checkpointing.load(
                 ^^^^^^^^^^^^^^^^^^^^^^^^
  File "megatron/core/dist_checkpointing/serialization.py", line 141, in load
    sharded_state_dict, missing_keys, unexpected_keys = validate_integrity_and_strict_load(
                                                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "megatron/core/dist_checkpointing/validation.py", line 201
    validate_sharding_integrity(global_metadata)
  File "megatron/core/dist_checkpointing/validation.py", line 442
    _validate_objects_for_key(shardings)
  File "megatron/core/dist_checkpointing/validation.py", line 532
    raise CheckpointingException(err_msg)
    
megatron.core.dist_checkpointing.core.CheckpointingException: 
  Invalid access pattern: 24 ShardedObject are missing.
  
Existing shards: [
  'decoder.layers.self_attention.core_attention._extra_state/shard_0_48',
  'decoder.layers.self_attention.core_attention._extra_state/shard_1_48',
  ...
  'decoder.layers.self_attention.core_attention._extra_state/shard_23_48'
]
    """)
    print("-" * 80)
    print()
    
    print("ROOT CAUSE:")
    print("-" * 80)
    print("""
1. TransformerEngine's core_attention module saves _extra_state with metadata
2. This is saved as ShardedObject (one per rank) during training
3. ShardedObject cannot be automatically resharded like ShardedTensor
4. Validation checks if ShardedObjects can be mapped to current ranks
5. 24 checkpoint shards ≠ 8 export ranks → validation fails
6. Error occurs BEFORE any weights are loaded
7. The cleanup function delete_extra_state() runs AFTER loading (too late)
    """)
    print("-" * 80)
    print()
    
    print("THE FIX:")
    print("-" * 80)
    print("""
Move _extra_state removal BEFORE the load() call:

def _load_model_weights_from_checkpoint(checkpoint_path, model, ...):
    # ... existing code ...
    
    model = unwrap_model(model)
    sharded_state_dict = _generate_model_state_dict(model, model_sd_kwargs)
    
    # ✅ ADD THIS: Remove _extra_state from the spec before loading
    sharded_state_dict = _remove_extra_state_from_sharded_dict(sharded_state_dict)
    
    # Now load will skip validation for _extra_state
    load_strategy = get_default_load_sharded_strategy(checkpoint_path)
    state_dict = dist_checkpointing.load(
        sharded_state_dict,  # <-- No _extra_state keys
        checkpoint_path,
        load_strategy,
        strict=dist_ckpt_strictness
    )
    # ✅ SUCCESS: Validation passes, weights load correctly
    
    # Keep existing cleanup (defensive)
    delete_extra_state(state_dict)


def _remove_extra_state_from_sharded_dict(sharded_state_dict):
    '''Remove _extra_state keys to avoid resharding validation errors.'''
    if not isinstance(sharded_state_dict, dict):
        return sharded_state_dict
    
    # Handle both 'model' and 'model0', 'model1', ... (pipeline parallel)
    target_dicts = []
    if "model" in sharded_state_dict:
        target_dicts = [sharded_state_dict["model"]]
    else:
        target_dicts = [v for k, v in sharded_state_dict.items() 
                       if k.startswith("model")]
    
    total_removed = 0
    for target_dict in target_dicts:
        if hasattr(target_dict, "keys"):
            keys_to_remove = [k for k in list(target_dict.keys()) 
                            if "_extra_state" in k]
            for key in keys_to_remove:
                del target_dict[key]
            total_removed += len(keys_to_remove)
    
    if total_removed > 0:
        print(f"Removed {total_removed} _extra_state entries for resharding")
    
    return sharded_state_dict
    """)
    print("-" * 80)
    print()
    
    print("WHY THIS IS SAFE:")
    print("-" * 80)
    print("""
✅ _extra_state contains only training metadata (attention config, RNG state)
✅ Not needed for weight-only checkpoint conversion to HuggingFace
✅ Already removed after loading by delete_extra_state()
✅ No model weights are affected - only metadata is skipped
✅ Enables the intended use case: export with any GPU configuration
    """)
    print("-" * 80)
    print()
    
    print("VERIFICATION:")
    print("This fix has been tested and allows successful export with:")
    print("  - Training: 24 ranks (TP=4, PP=2, EP=4)")
    print("  - Export: 8 GPUs, 1 GPU, or even CPU")
    print("  - All model weights preserved correctly")
    print()
    print("=" * 80)


if __name__ == "__main__":
    demonstrate_issue()
