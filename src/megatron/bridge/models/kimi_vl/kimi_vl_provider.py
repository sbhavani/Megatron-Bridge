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

from dataclasses import dataclass
from typing import Any

from megatron.core.models.gpt import GPTModel as MCoreGPTModel

from megatron.bridge.models.deepseek import DeepSeekV3ModelProvider

from .modeling_kimi_vl import KimiVLModel


# =============================================================================
# Kimi VL Model Providers
# =============================================================================


@dataclass
class KimiVLModelProvider(DeepSeekV3ModelProvider):
    """
    Base model provider for Kimi VL Models.

    Kimi VL is based on DeepSeek V3 architecture with vision-language capabilities.
    """

    # VL models shouldn't scatter embeddings across sequence parallel regions because
    # the vision embeddings are going to be inserted into the language embeddings.
    scatter_embedding_sequence_parallel: bool = False

    # Vision configuration - stored as a dict to allow flexible vision config types
    vision_config: Any = None

    # Token IDs - will be set during bridge conversion
    bos_token_id: int = 0
    eos_token_id: int = 1
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    image_token_id: int = 151655

    # Freeze options
    freeze_language_model: bool = False
    freeze_vision_model: bool = False
    freeze_vision_projection: bool = False

    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> KimiVLModel:
        model = KimiVLModel(self, pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)

        # Apply freeze options if any are enabled
        if self.freeze_language_model or self.freeze_vision_model or self.freeze_vision_projection:
            model.freeze(
                freeze_language_model=self.freeze_language_model,
                freeze_vision_model=self.freeze_vision_model,
                freeze_vision_projection=self.freeze_vision_projection,
            )

        return model

    def provide_language_model(self, pre_process=None, post_process=None, vp_stage=None) -> MCoreGPTModel:
        return super().provide(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
