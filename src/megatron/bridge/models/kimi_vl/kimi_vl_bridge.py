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
        # Kimi VL wraps the language model, so we need to extract its config
        text_config = getattr(hf_config, "text_config", hf_config)

        provider = KimiVLModelProvider(
            # Basic transformer config
            num_layers=text_config.num_hidden_layers,
            hidden_size=text_config.hidden_size,
            ffn_hidden_size=text_config.intermediate_size,
            num_attention_heads=text_config.num_attention_heads,
            num_query_groups=getattr(text_config, "num_key_value_heads", text_config.num_attention_heads),
            init_method_std=text_config.initializer_range,
            layernorm_epsilon=getattr(text_config, "rms_norm_eps", 1e-6),
            gated_linear_unit=True,
            make_vocab_size_divisible_by=self.make_vocab_size_divisible_by(text_config.vocab_size),
            rotary_base=getattr(text_config, "rope_theta", 10000.0),
            share_embeddings_and_output_weights=getattr(text_config, "tie_word_embeddings", False),
            vocab_size=text_config.vocab_size,
            seq_length=getattr(text_config, "max_position_embeddings", 4096),
            # Precision config
            fp16=(self.dtype_from_hf(hf_config, default=torch.float32) == torch.float16),
            bf16=(self.dtype_from_hf(hf_config, default=torch.float32) == torch.bfloat16),
            params_dtype=self.dtype_from_hf(hf_config, default=torch.float32),
            generation_config=hf_pretrained.generation_config,
            # VL-specific configs
            vision_config=getattr(hf_config, "vision_config", None),
            # VL-specific token IDs
            bos_token_id=getattr(hf_config, "bos_token_id", 0),
            eos_token_id=getattr(hf_config, "eos_token_id", 1),
            vision_start_token_id=getattr(hf_config, "vision_start_token_id", 151652),
            vision_end_token_id=getattr(hf_config, "vision_end_token_id", 151653),
            image_token_id=getattr(hf_config, "image_token_id", 151655),
        )

        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        # Dictionary maps Megatron parameter names -> HF parameter names
        # Kimi VL wraps the language model with vision components
        param_mappings = {
            # Language model embeddings and output
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

        mapping_list = []
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
