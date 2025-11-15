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

from unittest.mock import Mock, patch

import pytest
import torch
from transformers import GenerationConfig

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.hf_pretrained.vlm import PreTrainedVLM
from megatron.bridge.models.kimi_vl.kimi_vl_bridge import KimiVLBridge
from megatron.bridge.models.kimi_vl.kimi_vl_provider import KimiVLModelProvider


@pytest.fixture
def mock_vision_config():
    """Create a mock vision config."""
    config = Mock()
    config.hidden_size = 1280
    config.intermediate_size = 3420
    config.num_hidden_layers = 24
    return config


@pytest.fixture
def mock_hf_config(mock_vision_config):
    """Create a mock HF config for Kimi VL."""
    config = Mock()
    # DeepSeek V3 base configuration
    config.num_hidden_layers = 32
    config.hidden_size = 4096
    config.intermediate_size = 11008
    config.num_attention_heads = 32
    config.num_key_value_heads = 32
    config.initializer_range = 0.02
    config.rms_norm_eps = 1e-6
    config.vocab_size = 151936
    config.max_position_embeddings = 4096
    config.rope_theta = 1000000.0
    config.tie_word_embeddings = False

    # VL-specific configuration
    config.vision_config = mock_vision_config
    config.bos_token_id = 0
    config.eos_token_id = 1
    config.vision_start_token_id = 151652
    config.vision_end_token_id = 151653
    config.image_token_id = 151655

    # DeepSeek V3 specific (for MoE)
    config.aux_loss_alpha = 0.001

    return config


@pytest.fixture
def mock_hf_pretrained(mock_hf_config):
    """Create a mock HF pretrained VLM."""
    pretrained = Mock(spec=PreTrainedVLM)
    pretrained.config = mock_hf_config
    pretrained.generation_config = GenerationConfig()
    return pretrained


@pytest.fixture
def kimi_vl_bridge():
    """Create a KimiVLBridge instance."""
    return KimiVLBridge()


class TestKimiVLBridgeInitialization:
    """Test KimiVLBridge initialization and basic functionality."""

    def test_bridge_initialization(self, kimi_vl_bridge):
        """Test that bridge can be initialized."""
        assert isinstance(kimi_vl_bridge, KimiVLBridge)

    def test_bridge_has_required_methods(self, kimi_vl_bridge):
        """Test that bridge has required methods."""
        assert hasattr(kimi_vl_bridge, "provider_bridge")
        assert callable(kimi_vl_bridge.provider_bridge)

        assert hasattr(kimi_vl_bridge, "mapping_registry")
        assert callable(kimi_vl_bridge.mapping_registry)


class TestKimiVLBridgeProviderBridge:
    """Test provider_bridge method functionality."""

    @patch("megatron.bridge.models.kimi_vl.kimi_vl_bridge.get_common_configs")
    def test_provider_bridge_basic_config(self, mock_get_configs, kimi_vl_bridge, mock_hf_pretrained):
        """Test provider_bridge creates correct provider with basic config."""
        # Mock get_common_configs to return minimal config
        mock_get_configs.return_value = {
            "num_layers": 32,
            "hidden_size": 4096,
            "ffn_hidden_size": 11008,
            "num_attention_heads": 32,
            "num_query_groups": 32,
            "init_method_std": 0.02,
            "layernorm_epsilon": 1e-6,
            "vocab_size": 151936,
            "seq_length": 4096,
            "rotary_base": 1000000.0,
            "share_embeddings_and_output_weights": False,
        }

        provider = kimi_vl_bridge.provider_bridge(mock_hf_pretrained)

        assert isinstance(provider, KimiVLModelProvider)

    @patch("megatron.bridge.models.kimi_vl.kimi_vl_bridge.get_common_configs")
    def test_provider_bridge_vl_specific_config(self, mock_get_configs, kimi_vl_bridge, mock_hf_pretrained):
        """Test provider_bridge creates correct VL-specific configuration."""
        mock_get_configs.return_value = {
            "num_layers": 32,
            "hidden_size": 4096,
        }

        provider = kimi_vl_bridge.provider_bridge(mock_hf_pretrained)

        # Check VL-specific token IDs
        assert provider.bos_token_id == 0
        assert provider.eos_token_id == 1
        assert provider.vision_start_token_id == 151652
        assert provider.vision_end_token_id == 151653
        assert provider.image_token_id == 151655

        # Check vision config
        assert provider.vision_config is not None

    @patch("megatron.bridge.models.kimi_vl.kimi_vl_bridge.get_common_configs")
    def test_provider_bridge_with_custom_token_ids(self, mock_get_configs, kimi_vl_bridge, mock_hf_pretrained):
        """Test provider_bridge with custom token IDs."""
        mock_get_configs.return_value = {"num_layers": 32}

        # Modify mock config with custom token IDs
        mock_hf_pretrained.config.bos_token_id = 100
        mock_hf_pretrained.config.eos_token_id = 101
        mock_hf_pretrained.config.vision_start_token_id = 102

        provider = kimi_vl_bridge.provider_bridge(mock_hf_pretrained)

        assert provider.bos_token_id == 100
        assert provider.eos_token_id == 101
        assert provider.vision_start_token_id == 102

    @patch("megatron.bridge.models.kimi_vl.kimi_vl_bridge.get_common_configs")
    def test_provider_bridge_with_missing_token_ids(self, mock_get_configs, kimi_vl_bridge, mock_hf_pretrained):
        """Test provider_bridge with missing token IDs uses defaults."""
        mock_get_configs.return_value = {"num_layers": 32}

        # Remove some token IDs from config
        delattr(mock_hf_pretrained.config, "vision_start_token_id")
        delattr(mock_hf_pretrained.config, "image_token_id")

        provider = kimi_vl_bridge.provider_bridge(mock_hf_pretrained)

        # Should use defaults
        assert provider.vision_start_token_id == 151652
        assert provider.image_token_id == 151655

    @patch("megatron.bridge.models.kimi_vl.kimi_vl_bridge.get_common_configs")
    @patch.object(KimiVLBridge, "dtype_from_hf")
    def test_provider_bridge_dtype_handling(
        self, mock_dtype_from_hf, mock_get_configs, kimi_vl_bridge, mock_hf_pretrained
    ):
        """Test provider_bridge handles dtype correctly."""
        mock_get_configs.return_value = {"num_layers": 32}
        mock_dtype_from_hf.return_value = torch.float16

        provider = kimi_vl_bridge.provider_bridge(mock_hf_pretrained)

        assert provider.fp16 is True
        assert provider.bf16 is False
        assert provider.params_dtype == torch.float16

    @patch("megatron.bridge.models.kimi_vl.kimi_vl_bridge.get_common_configs")
    @patch.object(KimiVLBridge, "dtype_from_hf")
    def test_provider_bridge_bfloat16_handling(
        self, mock_dtype_from_hf, mock_get_configs, kimi_vl_bridge, mock_hf_pretrained
    ):
        """Test provider_bridge handles bfloat16 correctly."""
        mock_get_configs.return_value = {"num_layers": 32}
        mock_dtype_from_hf.return_value = torch.bfloat16

        provider = kimi_vl_bridge.provider_bridge(mock_hf_pretrained)

        assert provider.fp16 is False
        assert provider.bf16 is True
        assert provider.params_dtype == torch.bfloat16

    @patch("megatron.bridge.models.kimi_vl.kimi_vl_bridge.get_common_configs")
    def test_provider_bridge_generation_config(self, mock_get_configs, kimi_vl_bridge, mock_hf_pretrained):
        """Test provider_bridge includes generation config."""
        mock_get_configs.return_value = {"num_layers": 32}

        custom_gen_config = GenerationConfig(max_length=2048, temperature=0.8)
        mock_hf_pretrained.generation_config = custom_gen_config

        provider = kimi_vl_bridge.provider_bridge(mock_hf_pretrained)

        assert provider.generation_config is custom_gen_config

    @patch("megatron.bridge.models.kimi_vl.kimi_vl_bridge.get_common_configs")
    def test_provider_bridge_moe_config(self, mock_get_configs, kimi_vl_bridge, mock_hf_pretrained):
        """Test provider_bridge handles MoE configuration correctly."""
        mock_get_configs.return_value = {"num_layers": 32}

        provider = kimi_vl_bridge.provider_bridge(mock_hf_pretrained)

        # Check DeepSeek V3 specific MoE configs
        assert provider.make_vocab_size_divisible_by == 1280
        assert provider.moe_router_score_function == "sigmoid"
        assert provider.moe_router_enable_expert_bias is True
        assert provider.moe_aux_loss_coeff == 0.001


class TestKimiVLBridgeMappingRegistry:
    """Test mapping_registry method functionality."""

    def test_mapping_registry_returns_correct_type(self, kimi_vl_bridge):
        """Test mapping_registry returns MegatronMappingRegistry."""
        registry = kimi_vl_bridge.mapping_registry()

        assert isinstance(registry, MegatronMappingRegistry)

    def test_mapping_registry_contains_required_mappings(self, kimi_vl_bridge):
        """Test mapping_registry contains all required parameter mappings."""
        registry = kimi_vl_bridge.mapping_registry()

        # Extract mappings - registry should contain mappings for common parameters
        mappings = registry.mappings
        assert len(mappings) > 0

        # Check that we have mappings for embeddings, output layer, layernorms
        mapping_names = []
        for mapping in mappings:
            # Collect Megatron param pattern
            if hasattr(mapping, "megatron_param"):
                mapping_names.append(str(getattr(mapping, "megatron_param")))
            # Collect HF param pattern(s)
            hf = getattr(mapping, "hf_param", None)
            if isinstance(hf, dict):
                mapping_names.extend([str(v) for v in hf.values()])
            elif isinstance(hf, str):
                mapping_names.append(hf)

        # Should contain word embeddings mapping
        has_embeddings = any("embed_tokens" in name or "word_embeddings" in name for name in mapping_names)
        assert has_embeddings, "Should contain embeddings mapping"

        # Should contain output layer mapping
        has_output = any("lm_head" in name or "output_layer" in name for name in mapping_names)
        assert has_output, "Should contain output layer mapping"

    def test_mapping_registry_visual_params(self, kimi_vl_bridge):
        """Test mapping_registry handles visual parameters correctly."""
        registry = kimi_vl_bridge.mapping_registry()

        # Should contain visual parameter mappings
        mappings = registry.mappings
        mapping_names = []
        for mapping in mappings:
            if hasattr(mapping, "megatron_param"):
                mapping_names.append(str(getattr(mapping, "megatron_param")))
            hf = getattr(mapping, "hf_param", None)
            if isinstance(hf, dict):
                mapping_names.extend([str(v) for v in hf.values()])
            elif isinstance(hf, str):
                mapping_names.append(hf)

        # Should have vision tower and multi_modal_projector mappings
        has_vision_tower = any("vision_tower" in name for name in mapping_names)
        has_projector = any("multi_modal_projector" in name for name in mapping_names)
        assert has_vision_tower or has_projector, "Should contain vision parameter mappings"

    def test_mapping_registry_qkv_mappings(self, kimi_vl_bridge):
        """Test mapping_registry contains QKV parameter mappings."""
        registry = kimi_vl_bridge.mapping_registry()

        mappings = registry.mappings
        mapping_names = []
        for mapping in mappings:
            if hasattr(mapping, "megatron_param"):
                mapping_names.append(str(getattr(mapping, "megatron_param")))

        # Should contain QKV mappings
        has_qkv = any("linear_qkv" in name for name in mapping_names)
        assert has_qkv, "Should contain QKV mappings"

    def test_mapping_registry_mlp_mappings(self, kimi_vl_bridge):
        """Test mapping_registry contains MLP parameter mappings."""
        registry = kimi_vl_bridge.mapping_registry()

        mappings = registry.mappings
        mapping_names = []
        for mapping in mappings:
            if hasattr(mapping, "megatron_param"):
                mapping_names.append(str(getattr(mapping, "megatron_param")))

        # Should contain MLP mappings
        has_mlp = any("mlp" in name for name in mapping_names)
        assert has_mlp, "Should contain MLP mappings"


class TestKimiVLBridgeEdgeCases:
    """Test edge cases and error conditions."""

    @patch("megatron.bridge.models.kimi_vl.kimi_vl_bridge.get_common_configs")
    def test_provider_bridge_with_minimal_config(self, mock_get_configs, kimi_vl_bridge):
        """Test provider_bridge with minimal HF config."""
        minimal_pretrained = Mock(spec=PreTrainedVLM)
        minimal_config = Mock()

        # Set only required fields
        minimal_config.num_hidden_layers = 24
        minimal_config.hidden_size = 2048
        minimal_config.vision_config = None

        minimal_pretrained.config = minimal_config
        minimal_pretrained.generation_config = GenerationConfig()

        mock_get_configs.return_value = {
            "num_layers": 24,
            "hidden_size": 2048,
        }

        provider = kimi_vl_bridge.provider_bridge(minimal_pretrained)

        assert isinstance(provider, KimiVLModelProvider)


class TestKimiVLBridgePostConversionHook:
    """Test post_conversion_hook functionality."""

    def test_post_conversion_hook_binds_methods(self, kimi_vl_bridge):
        """Test that post_conversion_hook binds vision methods correctly."""
        # Create mock models
        megatron_model = Mock()
        megatron_model.pre_process = True

        hf_model = Mock()
        hf_model._extract_image_features = Mock()
        hf_model._merge_with_image_features = Mock()
        hf_model.vision_tower = Mock()
        hf_model.multi_modal_projector = Mock()

        # Call post_conversion_hook
        kimi_vl_bridge.post_conversion_hook(megatron_model, hf_model)

        # Verify methods were bound
        assert hasattr(megatron_model, "_extract_image_features")
        assert hasattr(megatron_model, "_merge_with_image_features")
        assert hasattr(megatron_model, "vision_tower")
        assert hasattr(megatron_model, "multi_modal_projector")
