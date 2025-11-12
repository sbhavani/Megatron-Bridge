# Megatron Bridge ↔ Megatron-LM Checkpoint Compatibility Guide

## Overview

This document explains the checkpoint compatibility between Megatron Bridge and Megatron-LM, common issues, and solutions.

## Issue #1: Missing 'args' Key ✅ FIXED

### Problem
When loading a Megatron Bridge checkpoint in Megatron-LM, you may encounter:
```
KeyError: 'args' at Megatron-LM/megatron/training/checkpointing.py:1382
```

### Root Cause
Megatron-LM's checkpoint loading code expects an `args` object in the checkpoint state_dict with specific attributes like `tensor_model_parallel_size`, `pipeline_model_parallel_size`, etc.

Megatron Bridge uses a `ConfigContainer` instead of args, and wasn't including the `args` key in saved checkpoints.

### Solution
The fix has been implemented in commit `53e350e`:
- Added `_build_megatron_lm_args_from_config()` to convert ConfigContainer to Megatron-LM args format
- Modified `generate_state_dict()` to include the `args` key when saving checkpoints
- All required attributes are now properly mapped

### Required Actions
1. Use the updated `checkpointing.py` with the fix
2. Re-save your checkpoint after applying the fix
3. The newly saved checkpoint will include the `args` key

---

## Issue #2: Weight Key Structure Mismatch 🔍 UNDER INVESTIGATION

### Problem
After fixing the missing 'args' issue, you may encounter:
```
ValueError: Different dict keys encountered in `apply_factory_merges` (
    dict_keys(['embedding.word_embeddings.weight', 'decoder.final_layernorm.weight', ...])
    vs
    dict_keys(['decoder.layers.0.mlp.experts.linear_fc1.weight0', ...])
)
```

### Understanding the Error

This error occurs in Megatron-LM's distributed checkpointing system when loading a checkpoint. The error indicates a mismatch between:
1. **What was saved**: The structure in `common.pt`
2. **What's expected**: The structure from the model's `sharded_state_dict()` during loading

### Checkpoint Structure Basics

Megatron's `torch_dist` checkpoint format separates data into two parts:

#### 1. Common Data (common.pt)
- Non-distributed metadata and configuration
- Contains: `args`, `checkpoint_version`, `iteration`, etc.
- May contain some model weights if they're not ShardedTensors

#### 2. Sharded Data (*.distcp files or directories)
- Distributed model weights stored as ShardedTensors
- Contains: Most layer weights, optimizer states, etc.

The separation happens in `save_preprocess()`:
```python
# Anything that is ShardedBase → goes to sharded files
# Everything else → goes to common.pt
sharded_part, common_state_dict = extract_sharded_base(sharded_state_dict)
```

### Possible Root Causes

1. **Inconsistent Sharding Strategy**
   - Megatron Bridge and Megatron-LM might be using different `sharded_state_dict()` implementations
   - Some weights are ShardedTensors in one but regular tensors in the other

2. **Model Architecture Differences**
   - MoE (Mixture of Experts) models may have different weight naming conventions
   - Expert-specific weights might be structured differently

3. **Checkpoint Conversion Issues**
   - When converting from HuggingFace to Megatron format, the conversion script might not be setting up the sharded state dict correctly

### Diagnostic Steps

We've provided a diagnostic script to help identify the exact issue:

```bash
python diagnose_checkpoint.py /path/to/checkpoint --show-expected
```

This will show you:
- What's actually in `common.pt`
- What files are in the checkpoint directory
- The expected structure for Megatron-LM compatibility

### Debugging Checklist

1. ✅ Run the diagnostic script on your converted checkpoint
2. ✅ Check if `common.pt` has the `args` key (should be fixed now)
3. ✅ Examine which weights are in `common.pt` vs sharded files
4. ✅ Compare with a checkpoint saved directly by Megatron-LM
5. ✅ Check if you're using the same model architecture on both sides

### Potential Solutions (To Be Implemented Based on Findings)

Once we understand the exact structure difference from the diagnostic output, we can implement one of these solutions:

#### Option A: Normalize Checkpoint Structure
Add a post-processing step in Megatron Bridge to ensure:
- The same weights are sharded as in Megatron-LM
- The checkpoint structure matches Megatron-LM's expectations

#### Option B: Custom Checkpoint Converter
Create a conversion utility that transforms Megatron Bridge checkpoints to be fully compatible with Megatron-LM:
```python
# Future utility (not yet implemented)
from megatron.bridge.tools import convert_checkpoint_for_megatron_lm

convert_checkpoint_for_megatron_lm(
    bridge_checkpoint_dir="/path/to/bridge/checkpoint",
    output_dir="/path/to/compatible/checkpoint"
)
```

#### Option C: Modify Model Loading Logic
Update the checkpoint loading code to handle both formats transparently.

---

## How to Proceed

### Step 1: Apply the Current Fix
The missing 'args' fix is already committed. Make sure you have the latest code:
```bash
git pull origin claude/fix-megatron-checkpoint-compat-011CV34vsQtL8vgYx6T6nzj8
```

### Step 2: Diagnose Your Checkpoint
Run the diagnostic script on your converted checkpoint:
```bash
python diagnose_checkpoint.py /path/to/your/checkpoint --show-expected
```

### Step 3: Share Diagnostic Output
If the weight key mismatch persists, share the output of the diagnostic script. This will help us:
- Understand the exact structure difference
- Identify if this is a general issue or specific to your model/conversion
- Implement the appropriate fix

### Step 4: Temporary Workaround
While we implement a permanent fix, you might be able to work around the issue by:
1. Using the same checkpoint format (`ckpt_format=torch_dist`) in both Bridge and LM
2. Ensuring the same tensor/pipeline parallelism settings
3. Using a checkpoint saved directly from Megatron-LM for training continuation

---

## Technical Deep Dive

### Checkpoint Save Flow (Megatron Core)

```python
# 1. Generate state dict from model
state_dict = {
    'args': args,
    'model': model.sharded_state_dict(),  # Returns ShardedStateDict
    'optimizer': optimizer.sharded_state_dict(),
    ...
}

# 2. Preprocess to separate sharded vs common
sharded_part, common_dict = save_preprocess(state_dict)

# 3. Save to respective locations
common_strategy.save_common(common_dict, checkpoint_dir)  # → common.pt
sharded_strategy.save(sharded_part, checkpoint_dir)       # → *.distcp files
```

### Checkpoint Load Flow (Megatron Core)

```python
# 1. Load common data
common_state_dict = load_common(checkpoint_dir)  # ← common.pt

# 2. Generate expected structure from current model
sharded_state_dict = model.sharded_state_dict()

# 3. Preprocess to extract factories
sharded_sd, nonpersistent, sh_ten_factories = load_preprocess(sharded_state_dict)

# 4. Load sharded data
loaded_dict = sharded_strategy.load(sharded_sd, checkpoint_dir)  # ← *.distcp files

# 5. Merge common and loaded data
merge(common_state_dict, loaded_dict)

# 6. Apply factory merges (THIS IS WHERE THE ERROR OCCURS)
# Tries to merge sh_ten_factories with common_state_dict
# Fails if keys don't match
loaded_dict = apply_factory_merges(common_state_dict, sh_ten_factories)
```

The error occurs at step 6 when the keys in `sh_ten_factories` (generated from the loading model) don't match the keys in `common_state_dict` (loaded from the checkpoint).

---

## Related Files

- `/src/megatron/bridge/training/checkpointing.py` - Main checkpointing logic
- `/diagnose_checkpoint.py` - Diagnostic utility script
- `/tmp/Megatron-LM/megatron/core/dist_checkpointing/` - Megatron-LM checkpointing implementation

## References

- [Megatron-LM Checkpoint Documentation](https://github.com/NVIDIA/Megatron-LM)
- [Megatron Core Distributed Checkpointing](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/dist_checkpointing)
