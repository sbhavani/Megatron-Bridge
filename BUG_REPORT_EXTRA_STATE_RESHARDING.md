# Bug Report: Checkpoint Export Fails with "ShardedObject are missing" Error

## Summary

Exporting Megatron checkpoints to HuggingFace format fails when the checkpoint was saved with a different number of ranks than the export process uses. The error occurs during checkpoint validation before any weights are loaded.

## Error Message

```
megatron.core.dist_checkpointing.core.CheckpointingException: Invalid access pattern: 24 ShardedObject are missing.
Existing shards: ['decoder.layers.self_attention.core_attention._extra_state/shard_0_48', ..., 'shard_23_48']
```

## Root Cause

**Location**: `src/megatron/bridge/training/checkpointing.py:1184-1186`

```python
def _load_model_weights_from_checkpoint(...):
    # ... 
    model = unwrap_model(model)
    sharded_state_dict = _generate_model_state_dict(model, model_sd_kwargs)  # Line 1177
    
    # Problem: sharded_state_dict includes _extra_state keys that reference 
    # ShardedObjects from training (e.g., 24 shards from TP=4*PP=2*EP=4)
    
    load_strategy = get_default_load_sharded_strategy(checkpoint_path)
    if fully_parallel_load:
        load_strategy = FullyParallelLoadStrategyWrapper(...)
    
    state_dict = dist_checkpointing.load(
        sharded_state_dict, checkpoint_path, load_strategy, strict=dist_ckpt_strictness
    )  # Line 1184-1186
    
    # ❌ FAILS HERE during validation before loading
    # ✅ Only AFTER load completes does it remove _extra_state (too late!)
    
    delete_extra_state(state_dict)  # Line 1189 - TOO LATE!
```

The issue is **sequencing**:
1. `_generate_model_state_dict()` creates a spec that includes `_extra_state` keys
2. `dist_checkpointing.load()` validates these can be loaded/resharded
3. Validation fails because `_extra_state` contains `ShardedObject` instances that cannot be resharded
4. Only after successful load would `delete_extra_state()` remove them

## Why This Happens

`_extra_state` is saved by TransformerEngine's `core_attention` module and contains:
- Attention backend configuration
- RNG state for dropout reproducibility  
- Cached module metadata

These are stored as `ShardedObject` (not `ShardedTensor`), which:
- ❌ Cannot be automatically resharded across different parallelism configs
- ❌ Fail validation when rank count doesn't match shard count
- ❌ Are not needed for weight-only checkpoint conversion

## Minimal Reproduction

```python
# Step 1: Train a model with TP=4, PP=2, EP=4 (24 total ranks)
# This creates checkpoints with 24 _extra_state ShardedObject shards

# Step 2: Try to export with different number of ranks
torchrun --nproc_per_node 8 examples/conversion/convert_checkpoints.py export \
  --hf-model Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --megatron-path ./checkpoints/model \
  --hf-path ./exports/model_hf

# Result: CheckpointingException: Invalid access pattern: 24 ShardedObject are missing
```

## Impact

- ❌ Cannot export checkpoints with any GPU count other than exact training config
- ❌ Cannot export on CPU (would need exactly N processes matching training)
- ❌ Blocks model conversion for inference/deployment

## Proposed Fix

Move `_extra_state` removal **before** loading, not after:

```python
def _load_model_weights_from_checkpoint(...):
    # ... existing code ...
    
    model = unwrap_model(model)
    sharded_state_dict = _generate_model_state_dict(model, model_sd_kwargs)
    
    # ✅ NEW: Remove _extra_state BEFORE loading to skip validation
    sharded_state_dict = _remove_extra_state_from_sharded_dict(sharded_state_dict)
    
    load_strategy = get_default_load_sharded_strategy(checkpoint_path)
    if fully_parallel_load:
        load_strategy = FullyParallelLoadStrategyWrapper(...)
    
    state_dict = dist_checkpointing.load(
        sharded_state_dict, checkpoint_path, load_strategy, strict=dist_ckpt_strictness
    )
    
    # Keep existing cleanup for any _extra_state that made it through
    delete_extra_state(state_dict)
    # ... rest of function ...


def _remove_extra_state_from_sharded_dict(sharded_state_dict):
    """Remove _extra_state keys before loading to avoid resharding validation errors."""
    if not isinstance(sharded_state_dict, dict):
        return sharded_state_dict
    
    total_removed = 0
    
    # Handle both single model and pipeline parallel cases
    if "model" in sharded_state_dict:
        target_dicts = [("model", sharded_state_dict["model"])]
    else:
        # Pipeline parallel: model0, model1, etc.
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
        print_rank_0(f"Removed {total_removed} _extra_state entries to enable resharding")
    
    return sharded_state_dict
```

## Why This Fix is Safe

1. **_extra_state is not needed for weight conversion** - It only contains training metadata
2. **Already being removed** - The code already calls `delete_extra_state()` after loading
3. **No weights lost** - Only removes metadata, preserves all model parameters
4. **Enables intended use case** - Export should work with any GPU configuration

## Current Workaround

Users can monkey-patch the function before importing megatron modules:

```python
# Apply before any megatron imports
from megatron.bridge.training import checkpointing

original_fn = checkpointing._load_model_weights_from_checkpoint

def patched_fn(checkpoint_path, model, *args, **kwargs):
    # ... [implementation that removes _extra_state before loading] ...
    
checkpointing._load_model_weights_from_checkpoint = patched_fn
```

## Environment

- **Megatron-Bridge**: Latest from main branch
- **Megatron-Core**: As vendored in megatron-lm
- **Training config**: TP=4, PP=2, EP=4, BF16
- **Export config**: 8 GPUs (any number ≠ 24 fails)
- **Model**: Qwen3-Coder-30B-A3B-Instruct (also affects other models)

## Files Modified (Proposed)

- `src/megatron/bridge/training/checkpointing.py`:
  - Add `_remove_extra_state_from_sharded_dict()` helper function
  - Call it before `dist_checkpointing.load()` in `_load_model_weights_from_checkpoint()`

## Related Code

The existing `delete_extra_state()` function (line 134) shows the intent to remove _extra_state, but it operates on the loaded state_dict, which is too late to prevent validation errors.

---

**Expected behavior**: Export should work with any GPU configuration, as weights are ShardedTensors that support resharding.

**Actual behavior**: Export requires exact same rank count as training due to non-reshardable _extra_state ShardedObjects.
