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

"""Falcon H1 Hybrid Model Support for Megatron Bridge.

Falcon H1 Architecture
----------------------
Falcon H1 uses a hybrid architecture combining Mamba state-space models with
standard Transformer attention in an alternating pattern:

- Even layers (0, 2, 4, ...): ParallelHybridLayer (Mamba mixer + Self-attention in parallel)
- Odd layers (1, 3, 5, ...): MLP-only layers

The parallel hybrid layers run Mamba and attention components simultaneously,
summing their outputs before applying bias-dropout-add fusion.

Key Components
--------------
1. ParallelHybridLayer: Custom layer combining Mamba + Attention
2. HybridMambaStack: Extended MambaStack supporting 'P' (parallel) layer type
3. FalconH1ModelProvider: Configuration inheriting from MambaModelProvider
4. layer_utils: Helper functions for layer allocation and pattern generation

Integration Points for ModelOpt and Compression
------------------------------------------------
When integrating with NVIDIA ModelOpt or other model compression tools:

1. **Layer Inspection**: Use `LayerType` constants to identify layer types
   - Query hybrid_override_pattern to understand architecture
   - Access layer_utils.allocate_falcon_h1_layers() for pattern analysis

2. **Quantization**: ParallelHybridLayer supports standard quantization hooks
   - Mamba mixer: Quantize conv1d and linear projections
   - Attention: Quantize QKV and output projections via standard methods

3. **Pruning**: Consider hybrid nature when applying structured pruning
   - ParallelHybridLayer has two parallel paths (Mamba + Attention)
   - Use validate_layer_pattern() to verify pruned architectures remain valid

4. **Distillation**: Both components accessible via submodules
   - Access via layer.mamba_mixer and layer.self_attention
   - Maintain parallel summation in distilled student models

Example Usage
-------------
>>> from megatron.bridge.models.falcon_h1 import FalconH1ModelProvider1B
>>> provider = FalconH1ModelProvider1B(vocab_size=32000)
>>> model = provider.provide()

For custom patterns:
>>> from megatron.bridge.models.falcon_h1 import generate_alternating_pattern
>>> pattern = generate_alternating_pattern(num_layers=20)  # "P-P-P-P-P-P-P-P-P-P-"
"""

from megatron.bridge.models.falcon_h1.falcon_h1_provider import (
    FalconH1ModelProvider,
    FalconH1ModelProvider1B,
)
from megatron.bridge.models.falcon_h1.parallel_hybrid_layer import (
    ParallelHybridLayer,
    ParallelHybridLayerSubmodules,
)
from megatron.bridge.models.falcon_h1.hybrid_mamba_stack import (
    HybridMambaStack,
    HybridMambaStackSubmodules,
)
from megatron.bridge.models.falcon_h1.falcon_h1_layer_specs import (
    get_falcon_h1_mamba_stack_spec,
)
from megatron.bridge.models.falcon_h1.layer_utils import (
    LayerType,
    allocate_falcon_h1_layers,
    generate_alternating_pattern,
    validate_layer_pattern,
    build_parallel_hybrid_layer,
)

__all__ = [
    "FalconH1ModelProvider",
    "FalconH1ModelProvider1B",
    "ParallelHybridLayer",
    "ParallelHybridLayerSubmodules",
    "HybridMambaStack",
    "HybridMambaStackSubmodules",
    "get_falcon_h1_mamba_stack_spec",
    "LayerType",
    "allocate_falcon_h1_layers",
    "generate_alternating_pattern",
    "validate_layer_pattern",
    "build_parallel_hybrid_layer",
]
