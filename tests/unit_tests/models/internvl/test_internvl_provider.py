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
from transformers.models.internvl.configuration_internvl import InternVLVisionConfig

from megatron.bridge.models.internvl.internvl_provider import InternVLModelProvider


class TestInternVLModelProvider:
    """Test cases for InternVLModelProvider class."""

    def test_internvl_model_provider_initialization(self):
        """Test InternVLModelProvider can be initialized with default values."""
        provider = InternVLModelProvider(
            num_layers=24,
            hidden_size=896,
            num_attention_heads=14,
        )

        # Check required transformer config fields
        assert provider.num_layers == 24
        assert provider.hidden_size == 896
        assert provider.num_attention_heads == 14

        # Check Qwen2-inherited defaults
        assert provider.normalization == "RMSNorm"
        assert provider.gated_linear_unit is True
        assert provider.position_embedding_type == "rope"
        assert provider.add_bias_linear is False
        assert provider.attention_dropout == 0.0
        assert provider.hidden_dropout == 0.0
        assert provider.share_embeddings_and_output_weights is False

    def test_internvl_vl_specific_defaults(self):
        """Test InternVLModelProvider VL-specific default configuration."""
        provider = InternVLModelProvider(
            num_layers=24,
            hidden_size=896,
            num_attention_heads=14,
        )

        # Check VL-specific defaults
        assert provider.scatter_embedding_sequence_parallel is False
        assert isinstance(provider.vision_config, InternVLVisionConfig)

        # Check token IDs
        assert provider.bos_token_id == 151643
        assert provider.eos_token_id == 151645
        assert provider.img_context_token_id == 151649
        assert provider.image_token_id == 151655

        # Check freeze options defaults
        assert provider.freeze_language_model is False
        assert provider.freeze_vision_model is False
        assert provider.freeze_vision_projection is False

        # Check InternVL-specific defaults
        assert provider.select_layer == -1
        assert provider.ps_version == "v2"
        assert provider.downsample_ratio == 0.5
        assert provider.dynamic_image_size is True
        assert provider.use_thumbnail is True
        assert provider.min_dynamic_patch == 1
        assert provider.max_dynamic_patch == 12

    def test_internvl_custom_vision_config(self):
        """Test InternVLModelProvider with custom vision configuration."""
        custom_vision_config = InternVLVisionConfig(
            hidden_size=1024,
            intermediate_size=4096,
            num_hidden_layers=24,
            num_attention_heads=16,
            image_size=448,
            patch_size=14,
        )

        provider = InternVLModelProvider(
            num_layers=24,
            hidden_size=896,
            num_attention_heads=14,
            vision_config=custom_vision_config,
        )

        assert provider.vision_config.hidden_size == 1024
        assert provider.vision_config.intermediate_size == 4096
        assert provider.vision_config.num_hidden_layers == 24
        assert provider.vision_config.num_attention_heads == 16

    def test_internvl_custom_token_ids(self):
        """Test InternVLModelProvider with custom token IDs."""
        provider = InternVLModelProvider(
            num_layers=24,
            hidden_size=896,
            num_attention_heads=14,
            bos_token_id=100,
            eos_token_id=101,
            img_context_token_id=102,
            image_token_id=103,
        )

        assert provider.bos_token_id == 100
        assert provider.eos_token_id == 101
        assert provider.img_context_token_id == 102
        assert provider.image_token_id == 103

    def test_internvl_custom_freeze_options(self):
        """Test InternVLModelProvider with custom freeze options."""
        provider = InternVLModelProvider(
            num_layers=24,
            hidden_size=896,
            num_attention_heads=14,
            freeze_language_model=True,
            freeze_vision_model=False,
            freeze_vision_projection=True,
        )

        assert provider.freeze_language_model is True
        assert provider.freeze_vision_model is False
        assert provider.freeze_vision_projection is True

    def test_internvl_custom_internvl_specific_configs(self):
        """Test InternVLModelProvider with custom InternVL-specific configurations."""
        provider = InternVLModelProvider(
            num_layers=24,
            hidden_size=896,
            num_attention_heads=14,
            select_layer=-2,
            ps_version="v1",
            downsample_ratio=0.25,
            dynamic_image_size=False,
            use_thumbnail=False,
            min_dynamic_patch=2,
            max_dynamic_patch=6,
        )

        assert provider.select_layer == -2
        assert provider.ps_version == "v1"
        assert provider.downsample_ratio == 0.25
        assert provider.dynamic_image_size is False
        assert provider.use_thumbnail is False
        assert provider.min_dynamic_patch == 2
        assert provider.max_dynamic_patch == 6

    def test_internvl_dtype_configuration(self):
        """Test InternVLModelProvider dtype configuration."""
        provider_bf16 = InternVLModelProvider(
            num_layers=24,
            hidden_size=896,
            num_attention_heads=14,
            bf16=True,
            params_dtype=torch.bfloat16,
        )

        assert provider_bf16.bf16 is True
        assert provider_bf16.params_dtype == torch.bfloat16

        provider_fp16 = InternVLModelProvider(
            num_layers=24,
            hidden_size=896,
            num_attention_heads=14,
            fp16=True,
            params_dtype=torch.float16,
        )

        assert provider_fp16.fp16 is True
        assert provider_fp16.params_dtype == torch.float16

    def test_internvl_inherit_from_qwen2_provider(self):
        """Test that InternVLModelProvider inherits Qwen2 configurations correctly."""
        provider = InternVLModelProvider(
            num_layers=24,
            hidden_size=896,
            num_attention_heads=14,
            seq_length=32768,
            vocab_size=151674,
            rotary_base=1000000.0,
        )

        # Check that inherited configurations work
        assert provider.seq_length == 32768
        assert provider.vocab_size == 151674
        assert provider.rotary_base == 1000000.0

        # VL-specific overrides should still work
        assert provider.scatter_embedding_sequence_parallel is False
        assert isinstance(provider.vision_config, InternVLVisionConfig)

    def test_internvl_provide_method_exists(self):
        """Test that provide method exists and is callable."""
        provider = InternVLModelProvider(
            num_layers=24,
            hidden_size=896,
            num_attention_heads=14,
        )

        assert hasattr(provider, "provide")
        assert callable(provider.provide)

    def test_internvl_provide_language_model_method_exists(self):
        """Test that provide_language_model method exists and is callable."""
        provider = InternVLModelProvider(
            num_layers=24,
            hidden_size=896,
            num_attention_heads=14,
        )

        assert hasattr(provider, "provide_language_model")
        assert callable(provider.provide_language_model)

    def test_internvl_model_provider_kv_channels(self):
        """Test InternVLModelProvider kv_channels calculation."""
        provider = InternVLModelProvider(
            num_layers=24,
            hidden_size=896,
            num_attention_heads=14,
            num_query_groups=2,
        )

        # kv_channels should be hidden_size // num_attention_heads
        expected_kv_channels = 896 // 14  # 64
        assert provider.kv_channels == expected_kv_channels

    def test_internvl_model_provider_parallelism_config(self):
        """Test InternVLModelProvider with parallelism configurations."""
        provider = InternVLModelProvider(
            num_layers=24,
            hidden_size=896,
            num_attention_heads=14,
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=2,
            context_parallel_size=1,
            sequence_parallel=True,
        )

        assert provider.tensor_model_parallel_size == 2
        assert provider.pipeline_model_parallel_size == 2
        assert provider.context_parallel_size == 1
        assert provider.sequence_parallel is True


class TestInternVLModelProviderIntegration:
    """Integration tests for InternVL model provider."""

    def test_internvl_provider_complete_configuration(self):
        """Test InternVLModelProvider with complete configuration."""
        custom_vision_config = InternVLVisionConfig(
            hidden_size=1024,
            intermediate_size=4096,
            num_hidden_layers=24,
            num_attention_heads=16,
            image_size=448,
            patch_size=14,
        )

        provider = InternVLModelProvider(
            # Language model config
            num_layers=24,
            hidden_size=896,
            num_attention_heads=14,
            num_query_groups=2,
            ffn_hidden_size=4864,
            seq_length=32768,
            vocab_size=151674,
            rotary_base=1000000.0,
            layernorm_epsilon=1e-6,
            # Vision config
            vision_config=custom_vision_config,
            # Token IDs
            bos_token_id=151643,
            eos_token_id=151645,
            img_context_token_id=151649,
            image_token_id=151655,
            # InternVL-specific
            select_layer=-1,
            ps_version="v2",
            downsample_ratio=0.5,
            dynamic_image_size=True,
            use_thumbnail=True,
            min_dynamic_patch=1,
            max_dynamic_patch=12,
            # Freeze options
            freeze_language_model=False,
            freeze_vision_model=False,
            freeze_vision_projection=False,
            # Dtype
            bf16=True,
            params_dtype=torch.bfloat16,
        )

        # Verify all configurations
        assert provider.num_layers == 24
        assert provider.hidden_size == 896
        assert provider.vision_config.hidden_size == 1024
        assert provider.select_layer == -1
        assert provider.bf16 is True

