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

import types

import torch

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.deepseek.common import get_common_configs, get_common_mapping_list
from megatron.bridge.models.hf_pretrained.vlm import PreTrainedVLM
from megatron.bridge.models.kimi_vl.kimi_vl_provider import KimiVLModelProvider
from megatron.bridge.models.kimi_vl.modeling_kimi_vl import KimiVLModel


@MegatronModelBridge.register_bridge(source="KimiVLForConditionalGeneration", target=KimiVLModel)
class KimiVLBridge(MegatronModelBridge):
    """
    Megatron Bridge for Kimi VL.

    This bridge handles the conversion between HuggingFace Kimi VL models
    and Megatron-Core formats, including weight mappings and configuration
    translation for vision-language models.

    Kimi VL is a multimodal vision-language model from Moonshot AI.

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("moonshotai/Kimi-VL-A3B-Instruct", trust_remote_code=True)
        >>> provider = bridge.to_megatron_provider()
    """

    def provider_bridge(self, hf_pretrained: PreTrainedVLM) -> KimiVLModelProvider:
        hf_config = hf_pretrained.config

        # Get text/language model config
        # Kimi VL wraps a DeepSeek V3-based language model (Moonlight), so extract its config
        text_config = getattr(hf_config, "text_config", hf_config)

        # Create a mock object to pass to get_common_configs
        # This allows us to reuse DeepSeek V3 config extraction for the language model
        class MockTextModel:
            def __init__(self, cfg, gen_cfg):
                self.config = cfg
                self.generation_config = gen_cfg

        mock_text_model = MockTextModel(text_config, hf_pretrained.generation_config)
        configs = get_common_configs(mock_text_model)

        # Add precision configs
        configs["fp16"] = self.dtype_from_hf(hf_config, default=torch.float32) == torch.float16
        configs["bf16"] = self.dtype_from_hf(hf_config, default=torch.float32) == torch.bfloat16
        configs["params_dtype"] = self.dtype_from_hf(hf_config, default=torch.float32)

        # Add VL-specific configs
        configs["vision_config"] = getattr(hf_config, "vision_config", None)

        # VL-specific token IDs
        configs["bos_token_id"] = getattr(hf_config, "bos_token_id", 0)
        configs["eos_token_id"] = getattr(hf_config, "eos_token_id", 1)
        configs["vision_start_token_id"] = getattr(hf_config, "vision_start_token_id", 151652)
        configs["vision_end_token_id"] = getattr(hf_config, "vision_end_token_id", 151653)
        configs["image_token_id"] = getattr(hf_config, "image_token_id", 151655)

        # DeepSeek V3 / Moonlight specific configs
        configs["make_vocab_size_divisible_by"] = 1280
        configs["moe_router_score_function"] = "sigmoid"
        configs["moe_router_enable_expert_bias"] = True
        if hasattr(text_config, "aux_loss_alpha"):
            configs["moe_aux_loss_coeff"] = text_config.aux_loss_alpha

        provider = KimiVLModelProvider(**configs)
        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        # Start with DeepSeek V3 common mappings for the language model
        # Then prefix them with "language_model." since Kimi VL wraps the LM
        mapping_list = get_common_mapping_list()

        # Prefix all language model mappings with "language_model."
        # Kimi VL structure: language_model.{deepseek_param} -> language_model.{hf_param}
        prefixed_mappings = []
        for mapping in mapping_list:
            if hasattr(mapping, 'megatron_param') and hasattr(mapping, 'hf_param'):
                # For AutoMapping
                if isinstance(mapping.hf_param, str):
                    new_mapping = AutoMapping(
                        megatron_param=f"language_model.{mapping.megatron_param}",
                        hf_param=f"language_model.{mapping.hf_param}",
                    )
                    prefixed_mappings.append(new_mapping)
                elif isinstance(mapping.hf_param, dict):
                    # For special mappings like QKV, MLP
                    # Need to prefix each component
                    from megatron.bridge.models.conversion.param_mapping import GatedMLPMapping

                    new_hf_params = {k: f"language_model.{v}" for k, v in mapping.hf_param.items()}
                    new_mapping = type(mapping)(
                        megatron_param=f"language_model.{mapping.megatron_param}",
                        **new_hf_params
                    )
                    prefixed_mappings.append(new_mapping)
            else:
                # Just prefix the params we can
                prefixed_mappings.append(mapping)

        # Add vision-specific mappings
        prefixed_mappings.extend([
            # Vision tower - replicate all parameters
            ReplicatedMapping(
                megatron_param="vision_tower.**",
                hf_param="vision_tower.**",
            ),
            # Multi-modal projector - replicate all parameters
            ReplicatedMapping(
                megatron_param="multi_modal_projector.**",
                hf_param="multi_modal_projector.**",
            ),
        ])

        return MegatronMappingRegistry(*prefixed_mappings)

    def post_conversion_hook(self, megatron_model: KimiVLModel, hf_model):
        """
        Post-conversion hook to bind HuggingFace vision methods to Megatron model.

        This is necessary because Kimi VL uses dynamic module loading for vision processing.
        """
        if hasattr(megatron_model, 'pre_process') and megatron_model.pre_process:
            # Bind vision processing methods from HuggingFace model if available
            if hasattr(hf_model, '_extract_image_features'):
                megatron_model._extract_image_features = types.MethodType(
                    hf_model._extract_image_features.__func__, megatron_model
                )

            if hasattr(hf_model, '_merge_with_image_features'):
                megatron_model._merge_with_image_features = types.MethodType(
                    hf_model._merge_with_image_features.__func__, megatron_model
                )

            # Copy vision tower and projector if they exist in HF model
            if hasattr(hf_model, 'vision_tower'):
                megatron_model.vision_tower = hf_model.vision_tower

            if hasattr(hf_model, 'multi_modal_projector'):
                megatron_model.multi_modal_projector = hf_model.multi_modal_projector
