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

from dataclasses import dataclass, field
from typing import Optional

from megatron.core.models.gpt import GPTModel as MCoreGPTModel
from transformers.models.internvl.configuration_internvl import InternVLConfig, InternVLVisionConfig

from megatron.bridge.models.qwen.qwen_provider import Qwen2ModelProvider
from .modeling_internvl import InternVLModel


# =============================================================================
# InternVL Model Providers
# =============================================================================


@dataclass
class InternVLModelProvider(Qwen2ModelProvider):
    """
    Base model provider for InternVL Models.

    InternVL is a vision-language model that uses:
    - InternViT vision encoder
    - MLP projector
    - Language model (typically Qwen2.5 or InternLM)
    """

    # VL models shouldn't scatter embeddings across sequence parallel regions because
    # the vision embeddings are going to be inserted into the language embeddings.
    scatter_embedding_sequence_parallel: bool = False

    # Vision configuration
    vision_config: InternVLVisionConfig = field(default_factory=InternVLVisionConfig)

    # Token IDs
    bos_token_id: int = 151643
    eos_token_id: int = 151645
    img_context_token_id: int = 151649
    image_token_id: int = 151655

    # Freeze options
    freeze_language_model: bool = False
    freeze_vision_model: bool = False
    freeze_vision_projection: bool = False

    # InternVL specific configurations
    select_layer: int = -1  # Layer to extract features from vision encoder
    ps_version: str = "v2"  # Pixel shuffle version (v1 or v2)
    downsample_ratio: float = 0.5  # Downsample ratio for vision features
    template: Optional[str] = None  # Conversation template
    dynamic_image_size: bool = True  # Whether to use dynamic image sizes
    use_thumbnail: bool = True  # Whether to use thumbnail
    min_dynamic_patch: int = 1  # Minimum number of dynamic patches
    max_dynamic_patch: int = 12  # Maximum number of dynamic patches

    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> InternVLModel:
        model = InternVLModel(self, pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)

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
