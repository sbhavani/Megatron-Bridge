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

__all__ = [
    "FalconH1ModelProvider",
    "FalconH1ModelProvider1B",
    "ParallelHybridLayer",
    "ParallelHybridLayerSubmodules",
    "HybridMambaStack",
    "HybridMambaStackSubmodules",
    "get_falcon_h1_mamba_stack_spec",
]
