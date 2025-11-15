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
    GatedMLPMapping,
    QKVMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.deepseek.common import get_common_configs, get_common_mapping_list
from megatron.bridge.models.hf_pretrained.vlm import PreTrainedVLM
from megatron.bridge.models.kimi_vl.kimi_vl_provider import KimiVLModelProvider
from megatron.bridge.models.kimi_vl.modeling_kimi_vl import KimiVLModel


@MegatronModelBridge.register_bridge(source="DeepseekV3ForCausalLM", target=KimiVLModel)
class KimiVLBridge(MegatronModelBridge):
    """
    Megatron Bridge for Kimi VL.

    This bridge handles the conversion between HuggingFace Kimi VL models
    (based on DeepSeek V3 with vision capabilities) and Megatron-Core formats,
    including weight mappings and configuration translation for vision-language models.

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("Kimi/kimi-vl-1.5", trust_remote_code=True)
        >>> provider = bridge.to_megatron_provider()
    """

    def provider_bridge(self, hf_pretrained: PreTrainedVLM) -> KimiVLModelProvider:
        hf_config = hf_pretrained.config

        # Get common DeepSeek V3 configs
        configs = get_common_configs(hf_pretrained)

        # Add precision configs
        configs["fp16"] = self.dtype_from_hf(hf_config, default=torch.float32) == torch.float16
        configs["bf16"] = self.dtype_from_hf(hf_config, default=torch.float32) == torch.bfloat16
        configs["params_dtype"] = self.dtype_from_hf(hf_config, default=torch.float32)

        # Add VL-specific configs
        configs["vision_config"] = getattr(hf_config, "vision_config", None)
        configs["generation_config"] = hf_pretrained.generation_config

        # VL-specific token IDs
        configs["bos_token_id"] = getattr(hf_config, "bos_token_id", 0)
        configs["eos_token_id"] = getattr(hf_config, "eos_token_id", 1)
        configs["vision_start_token_id"] = getattr(hf_config, "vision_start_token_id", 151652)
        configs["vision_end_token_id"] = getattr(hf_config, "vision_end_token_id", 151653)
        configs["image_token_id"] = getattr(hf_config, "image_token_id", 151655)

        # DeepSeek V3 specific configs for Kimi VL
        configs["make_vocab_size_divisible_by"] = 1280
        configs["moe_router_score_function"] = "sigmoid"
        configs["moe_router_enable_expert_bias"] = True
        if hasattr(hf_config, "aux_loss_alpha"):
            configs["moe_aux_loss_coeff"] = hf_config.aux_loss_alpha

        provider = KimiVLModelProvider(**configs)
        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        # Start with common DeepSeek mappings
        mapping_list = get_common_mapping_list()

        # Dictionary maps Megatron parameter names -> HF parameter names
        # for Kimi VL specific structure
        param_mappings = {
            # Language model mappings (Kimi VL structure)
            "language_model.embedding.word_embeddings.weight": "language_model.model.embed_tokens.weight",
            "language_model.output_layer.weight": "language_model.lm_head.weight",
            "language_model.decoder.final_layernorm.weight": "language_model.model.norm.weight",
            # Layer-specific mappings
            "language_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": (
                "language_model.model.layers.*.input_layernorm.weight"
            ),
            "language_model.decoder.layers.*.mlp.linear_fc1.layer_norm_weight": (
                "language_model.model.layers.*.post_attention_layernorm.weight"
            ),
            "language_model.decoder.layers.*.self_attention.linear_proj.weight": (
                "language_model.model.layers.*.self_attn.o_proj.weight"
            ),
            "language_model.decoder.layers.*.mlp.linear_fc2.weight": "language_model.model.layers.*.mlp.down_proj.weight",
        }

        # Add parameter mappings
        for megatron_param, hf_param in param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        # Add special mappings for vision components and complex transformations
        mapping_list.extend(
            [
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
                # QKV: Combine separate Q, K, V matrices into single QKV matrix
                QKVMapping(
                    megatron_param="language_model.decoder.layers.*.self_attention.linear_qkv.weight",
                    q="language_model.model.layers.*.self_attn.q_proj.weight",
                    k="language_model.model.layers.*.self_attn.k_proj.weight",
                    v="language_model.model.layers.*.self_attn.v_proj.weight",
                ),
                # Gated MLP: Combine gate and up projection matrices into single FC1 matrix
                GatedMLPMapping(
                    megatron_param="language_model.decoder.layers.*.mlp.linear_fc1.weight",
                    gate="language_model.model.layers.*.mlp.gate_proj.weight",
                    up="language_model.model.layers.*.mlp.up_proj.weight",
                ),
            ]
        )

        return MegatronMappingRegistry(*mapping_list)

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
