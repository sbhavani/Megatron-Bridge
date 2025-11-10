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

import torch

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    QKVMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.internvl.internvl_provider import InternVLModelProvider
from megatron.bridge.models.internvl.modeling_internvl import InternVLModel
from megatron.bridge.models.hf_pretrained.vlm import PreTrainedVLM


@MegatronModelBridge.register_bridge(source="InternVLChatModel", target=InternVLModel)
class InternVLBridge(MegatronModelBridge):
    """Megatron Bridge for InternVL vision-language models."""

    def provider_bridge(self, hf_pretrained: PreTrainedVLM) -> InternVLModelProvider:
        hf_config = hf_pretrained.config
        llm_config = getattr(hf_config, 'llm_config', hf_config)

        if hasattr(hf_config, 'get_vision_config'):
            vision_config = hf_config.get_vision_config()
        else:
            vision_config = hf_config.vision_config

        # Set vision config defaults for missing attributes
        if not hasattr(vision_config, 'use_mask_token'):
            vision_config.use_mask_token = False
        if not hasattr(vision_config, 'mask_token_id'):
            vision_config.mask_token_id = None
        if not hasattr(vision_config, 'use_absolute_position_embeddings'):
            vision_config.use_absolute_position_embeddings = True
        if not hasattr(vision_config, 'use_mean_pooling'):
            vision_config.use_mean_pooling = False
        if not hasattr(vision_config, 'hidden_dropout_prob'):
            vision_config.hidden_dropout_prob = 0.0
        if not hasattr(vision_config, 'attention_dropout'):
            vision_config.attention_dropout = 0.0
        if not hasattr(vision_config, 'projection_dropout'):
            vision_config.projection_dropout = 0.0
        if not hasattr(vision_config, 'use_qk_norm'):
            vision_config.use_qk_norm = False
        if not hasattr(vision_config, 'attention_bias'):
            vision_config.attention_bias = True
        if not hasattr(vision_config, '_attn_implementation'):
            vision_config._attn_implementation = 'eager'
        if not hasattr(vision_config, 'chunk_size_feed_forward'):
            vision_config.chunk_size_feed_forward = 0
        if not hasattr(vision_config, 'norm_type'):
            vision_config.norm_type = 'layer_norm'
        if not hasattr(vision_config, 'layer_scale_init_value'):
            vision_config.layer_scale_init_value = 1.0
        if not hasattr(vision_config, 'layer_norm_eps'):
            vision_config.layer_norm_eps = 1e-6
        if not hasattr(vision_config, 'num_channels'):
            vision_config.num_channels = 3

        # Convert int to tuple for image/patch size
        if hasattr(vision_config, 'image_size') and isinstance(vision_config.image_size, int):
            vision_config.image_size = (vision_config.image_size, vision_config.image_size)
        if hasattr(vision_config, 'patch_size') and isinstance(vision_config.patch_size, int):
            vision_config.patch_size = (vision_config.patch_size, vision_config.patch_size)

        vocab_size = getattr(llm_config, 'vocab_size', 151674)
        
        provider = InternVLModelProvider(
            init_method_std=getattr(llm_config, 'initializer_range', 0.02),
            hidden_size=getattr(llm_config, 'hidden_size', 896),
            ffn_hidden_size=getattr(llm_config, 'intermediate_size', 4864),
            kv_channels=getattr(llm_config, 'hidden_size', 896) // getattr(llm_config, 'num_attention_heads', 14),
            seq_length=getattr(llm_config, 'max_position_embeddings', 32768),
            num_attention_heads=getattr(llm_config, 'num_attention_heads', 14),
            num_layers=getattr(llm_config, 'num_hidden_layers', 24),
            num_query_groups=getattr(llm_config, 'num_key_value_heads', 2),
            rotary_base=getattr(llm_config, 'rope_theta', 1000000.0),
            layernorm_epsilon=getattr(llm_config, 'rms_norm_eps', 1e-6),
            vocab_size=vocab_size,
            make_vocab_size_divisible_by=128,
            vision_config=vision_config,
            bos_token_id=getattr(hf_config, "bos_token_id", 151643),
            eos_token_id=getattr(hf_config, "eos_token_id", 151645),
            img_context_token_id=getattr(hf_config, "img_context_token_id", 151649),
            image_token_id=getattr(hf_config, "image_token_id", 151655),
            select_layer=getattr(hf_config, "select_layer", -1),
            ps_version=getattr(hf_config, "ps_version", "v2"),
            downsample_ratio=getattr(hf_config, "downsample_ratio", 0.5),
            dynamic_image_size=getattr(hf_config, "dynamic_image_size", True),
            use_thumbnail=getattr(hf_config, "use_thumbnail", True),
            min_dynamic_patch=getattr(hf_config, "min_dynamic_patch", 1),
            max_dynamic_patch=getattr(hf_config, "max_dynamic_patch", 12),
            fp16=(self.dtype_from_hf(hf_config, default=torch.float32) == torch.float16),
            bf16=(self.dtype_from_hf(hf_config, default=torch.float32) == torch.bfloat16),
            params_dtype=self.dtype_from_hf(hf_config, default=torch.float32),
        )
        
        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        param_mappings = {
            # Language model embeddings
            "language_model.model.embed_tokens.weight": "language_model.embedding.word_embeddings.weight",
            # Language model layers
            "language_model.model.layers.*.input_layernorm.weight": "language_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight",
            "language_model.model.layers.*.self_attn.o_proj.weight": "language_model.decoder.layers.*.self_attention.linear_proj.weight",
            "language_model.model.layers.*.post_attention_layernorm.weight": "language_model.decoder.layers.*.mlp.linear_fc1.layer_norm_weight",
            "language_model.model.layers.*.mlp.down_proj.weight": "language_model.decoder.layers.*.mlp.linear_fc2.weight",
            # Final layer norm
            "language_model.model.norm.weight": "language_model.decoder.final_layernorm.weight",
        }

        mapping_list = [AutoMapping(megatron_param=m, hf_param=h) for h, m in param_mappings.items()]
        
        mapping_list.extend([
            ReplicatedMapping(megatron_param="vision_model.**", hf_param="vision_model.**"),
            ReplicatedMapping(megatron_param="mlp1.**", hf_param="mlp1.**"),
            QKVMapping(
                megatron_param="language_model.decoder.layers.*.self_attention.linear_qkv.weight",
                q="language_model.model.layers.*.self_attn.q_proj.weight",
                k="language_model.model.layers.*.self_attn.k_proj.weight",
                v="language_model.model.layers.*.self_attn.v_proj.weight",
            ),
            GatedMLPMapping(
                megatron_param="language_model.decoder.layers.*.mlp.linear_fc1.weight",
                gate="language_model.model.layers.*.mlp.gate_proj.weight",
                up="language_model.model.layers.*.mlp.up_proj.weight",
            ),
        ])

        return MegatronMappingRegistry(*mapping_list)

