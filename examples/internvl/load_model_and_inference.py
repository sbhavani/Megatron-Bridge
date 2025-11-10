#!/usr/bin/env python
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
PRODUCTION INFERENCE SCRIPT - Uses ONLY Megatron Bridge

✅ This script demonstrates the correct way to use InternVL with Megatron Bridge.
   It ONLY loads the model via AutoBridge - no HuggingFace model loading.

This script:
1. Loads an InternVL model using AutoBridge (automatic HF -> Megatron conversion)
2. Configures distributed parallelism (Tensor, Pipeline, Context Parallel)
3. Runs inference with token generation

Purpose: Production inference and typical usage pattern for InternVL with Megatron.

Requirements:
    - transformers >= 4.37.2 (for InternVL support)
    - For InternVL3.5, transformers >= 4.52.1

Usage:
    torchrun --nproc_per_node=8 load_model_and_inference.py <model_path> \
        --tp_size 2 --pp_size 1 --cp_size 1
"""

import argparse
import os
import sys

import torch
from tqdm import tqdm

try:
    import transformers
    from transformers import AutoTokenizer

    # Check transformers version
    from packaging import version
    transformers_version = version.parse(transformers.__version__)
    if transformers_version < version.parse("4.37.2"):
        print(f"Warning: transformers version {transformers.__version__} is too old.")
        print("InternVL requires transformers >= 4.37.2")
        print("Please upgrade: pip install 'transformers>=4.37.2'")
        sys.exit(1)
except ImportError as e:
    print(f"Error importing required packages: {e}")
    print("Please install: pip install transformers packaging")
    sys.exit(1)

from megatron.core import parallel_state
from megatron.bridge.models.conversion import AutoBridge


def init_distributed(tp_size, pp_size, cp_size):
    """Initialize distributed training environment."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl", world_size=world_size, rank=local_rank)

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=pp_size,
        context_parallel_size=cp_size,
    )


def main():
    parser = argparse.ArgumentParser(description="InternVL Model Inference with Megatron-Bridge")
    parser.add_argument("model_path", type=str, help="Path to the InternVL model")
    parser.add_argument("--tp_size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--pp_size", type=int, default=1, help="Pipeline parallel size")
    parser.add_argument("--cp_size", type=int, default=1, help="Context parallel size")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Maximum number of new tokens to generate")
    parser.add_argument("--prompt", type=str, default="Describe this image in detail.", help="Text prompt")
    parser.add_argument("--image_path", type=str, default=None, help="Path to image file")
    args = parser.parse_args()

    # Initialize distributed environment
    init_distributed(args.tp_size, args.pp_size, args.cp_size)

    # Load model using AutoBridge
    # Note: InternVL models require trust_remote_code=True
    bridge = AutoBridge.from_hf_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    provider = bridge.to_megatron_provider()

    # Set parallelism configuration
    provider.tensor_model_parallel_size = args.tp_size
    provider.pipeline_model_parallel_size = args.pp_size
    provider.context_parallel_size = args.cp_size

    # Provide the distributed model
    model = provider.provide_distributed_model(wrap_with_ddp=False)
    
    # Model is returned as a list when using model parallelism
    if isinstance(model, list):
        model = model[0]
    
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # Prepare input
    if args.image_path:
        from PIL import Image
        image = Image.open(args.image_path).convert("RGB")
        # Prepare inputs with image
        inputs = tokenizer(
            f"<image>\n{args.prompt}",
            return_tensors="pt",
        )
        # TODO: Add image preprocessing and pixel_values
        print("Image input support coming soon!")
    else:
        # Text-only input for testing
        inputs = tokenizer(args.prompt, return_tensors="pt")

    input_ids = inputs["input_ids"].cuda()

    # Generate tokens
    if parallel_state.is_pipeline_first_stage():
        print(f"Generating {args.max_new_tokens} tokens...")

    generated_ids = input_ids
    with torch.no_grad():
        for _ in tqdm(range(args.max_new_tokens), disable=parallel_state.get_tensor_model_parallel_rank() != 0):
            outputs = model(
                input_ids=generated_ids,
                position_ids=torch.arange(generated_ids.shape[1], device=generated_ids.device).unsqueeze(0),
            )

            if parallel_state.is_pipeline_last_stage():
                # Get logits and sample next token
                logits = outputs[:, -1, :]
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

                # Broadcast to all ranks
                torch.distributed.broadcast(next_token, src=parallel_state.get_pipeline_model_parallel_last_rank())
            else:
                next_token = torch.zeros((1, 1), dtype=torch.long, device=generated_ids.device)
                torch.distributed.broadcast(next_token, src=parallel_state.get_pipeline_model_parallel_last_rank())

            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            # Stop if EOS token is generated
            if next_token.item() == tokenizer.eos_token_id:
                break

    # Decode and print result
    if parallel_state.get_tensor_model_parallel_rank() == 0 and parallel_state.get_pipeline_model_parallel_rank() == 0:
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print(f"\nPrompt: {args.prompt}")
        print(f"Generated: {generated_text}")

    # Cleanup
    parallel_state.destroy_model_parallel()


if __name__ == "__main__":
    main()
