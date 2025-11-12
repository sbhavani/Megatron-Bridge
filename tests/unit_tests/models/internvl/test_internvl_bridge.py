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
from transformers.models.internvl.configuration_internvl import InternVLVisionConfig

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.hf_pretrained.vlm import PreTrainedVLM
from megatron.bridge.models.internvl.internvl_bridge import InternVLBridge
from megatron.bridge.models.internvl.internvl_provider import InternVLModelProvider


@pytest.fixture
def mock_hf_llm_config():
    """Create a mock HF LLM config for InternVL."""
    config = Mock()
    config.num_hidden_layers = 24
    config.hidden_size = 896
    config.intermediate_size = 4864
    config.num_attention_heads = 14
    config.num_key_value_heads = 2
    config.initializer_range = 0.02
    config.rms_norm_eps = 1e-6
    config.vocab_size = 151674
    config.max_position_embeddings = 32768
    config.rope_theta = 1000000.0
    config.tie_word_embeddings = False
    return config


@pytest.fixture
def mock_hf_config(mock_hf_llm_config):
    """Create a mock HF config for InternVL."""
    config = Mock()
    config.llm_config = mock_hf_llm_config
    config.vision_config = InternVLVisionConfig()
    config.bos_token_id = 151643
    config.eos_token_id = 151645
    config.img_context_token_id = 151649
    config.image_token_id = 151655
    config.select_layer = -1
    config.ps_version = "v2"
    config.downsample_ratio = 0.5
    config.dynamic_image_size = True
    config.use_thumbnail = True
    config.min_dynamic_patch = 1
    config.max_dynamic_patch = 12

    # Add get_vision_config method
    def get_vision_config():
        return config.vision_config

    config.get_vision_config = get_vision_config
    return config


@pytest.fixture
def mock_hf_pretrained(mock_hf_config):
    """Create a mock HF pretrained VLM."""
    pretrained = Mock(spec=PreTrainedVLM)
    pretrained.config = mock_hf_config
    pretrained.generation_config = GenerationConfig()
    return pretrained


@pytest.fixture
def internvl_bridge():
    """Create an InternVLBridge instance."""
    return InternVLBridge()


class TestInternVLBridgeInitialization:
    """Test InternVLBridge initialization and basic functionality."""

    def test_bridge_initialization(self, internvl_bridge):
        """Test that bridge can be initialized."""
        assert isinstance(internvl_bridge, InternVLBridge)

    def test_bridge_has_required_methods(self, internvl_bridge):
        """Test that bridge has required methods."""
        assert hasattr(internvl_bridge, "provider_bridge")
        assert callable(internvl_bridge.provider_bridge)

        assert hasattr(internvl_bridge, "mapping_registry")
        assert callable(internvl_bridge.mapping_registry)


class TestInternVLBridgeProviderBridge:
    """Test provider_bridge method functionality."""

    def test_provider_bridge_basic_config(self, internvl_bridge, mock_hf_pretrained):
        """Test provider_bridge creates correct provider with basic config."""
        provider = internvl_bridge.provider_bridge(mock_hf_pretrained)

        assert isinstance(provider, InternVLModelProvider)

        # Check basic transformer config
        assert provider.num_layers == 24
        assert provider.hidden_size == 896
        assert provider.ffn_hidden_size == 4864
        assert provider.num_attention_heads == 14
        assert provider.num_query_groups == 2

    def test_provider_bridge_vision_config(self, internvl_bridge, mock_hf_pretrained):
        """Test provider_bridge correctly handles vision configuration."""
        provider = internvl_bridge.provider_bridge(mock_hf_pretrained)

        assert isinstance(provider.vision_config, InternVLVisionConfig)

    def test_provider_bridge_token_ids(self, internvl_bridge, mock_hf_pretrained):
        """Test provider_bridge correctly sets token IDs."""
        provider = internvl_bridge.provider_bridge(mock_hf_pretrained)

        assert provider.bos_token_id == 151643
        assert provider.eos_token_id == 151645
        assert provider.img_context_token_id == 151649
        assert provider.image_token_id == 151655

    def test_provider_bridge_internvl_specific_configs(self, internvl_bridge, mock_hf_pretrained):
        """Test provider_bridge correctly sets InternVL-specific configurations."""
        provider = internvl_bridge.provider_bridge(mock_hf_pretrained)

        assert provider.select_layer == -1
        assert provider.ps_version == "v2"
        assert provider.downsample_ratio == 0.5
        assert provider.dynamic_image_size is True
        assert provider.use_thumbnail is True
        assert provider.min_dynamic_patch == 1
        assert provider.max_dynamic_patch == 12

    def test_provider_bridge_dtype_handling(self, internvl_bridge, mock_hf_pretrained):
        """Test provider_bridge correctly handles dtype conversion."""
        # Mock torch_dtype attribute
        mock_hf_pretrained.config.torch_dtype = torch.bfloat16

        provider = internvl_bridge.provider_bridge(mock_hf_pretrained)

        assert provider.bf16 is True
        assert provider.params_dtype == torch.bfloat16

    def test_provider_bridge_vision_config_defaults(self, internvl_bridge, mock_hf_pretrained):
        """Test provider_bridge sets vision config defaults for missing attributes."""
        # Remove some attributes to test defaults
        vision_config = InternVLVisionConfig()
        if hasattr(vision_config, 'use_mask_token'):
            delattr(vision_config, 'use_mask_token')
        if hasattr(vision_config, 'use_qk_norm'):
            delattr(vision_config, 'use_qk_norm')

        mock_hf_pretrained.config.vision_config = vision_config

        provider = internvl_bridge.provider_bridge(mock_hf_pretrained)

        # Verify defaults were set
        assert hasattr(provider.vision_config, 'use_mask_token')
        assert hasattr(provider.vision_config, 'use_qk_norm')

    def test_provider_bridge_image_size_conversion(self, internvl_bridge, mock_hf_pretrained):
        """Test provider_bridge converts int image_size to tuple."""
        # Set image_size as int
        mock_hf_pretrained.config.vision_config.image_size = 448

        provider = internvl_bridge.provider_bridge(mock_hf_pretrained)

        # Should be converted to tuple
        assert isinstance(provider.vision_config.image_size, tuple)
        assert provider.vision_config.image_size == (448, 448)

    def test_provider_bridge_patch_size_conversion(self, internvl_bridge, mock_hf_pretrained):
        """Test provider_bridge converts int patch_size to tuple."""
        # Set patch_size as int
        mock_hf_pretrained.config.vision_config.patch_size = 14

        provider = internvl_bridge.provider_bridge(mock_hf_pretrained)

        # Should be converted to tuple
        assert isinstance(provider.vision_config.patch_size, tuple)
        assert provider.vision_config.patch_size == (14, 14)

    def test_provider_bridge_without_llm_config(self, internvl_bridge, mock_hf_pretrained):
        """Test provider_bridge handles configs without separate llm_config."""
        # Remove llm_config to test fallback
        delattr(mock_hf_pretrained.config, 'llm_config')
        
        # Add LLM attributes directly to main config
        mock_hf_pretrained.config.num_hidden_layers = 24
        mock_hf_pretrained.config.hidden_size = 896
        mock_hf_pretrained.config.intermediate_size = 4864
        mock_hf_pretrained.config.num_attention_heads = 14
        mock_hf_pretrained.config.num_key_value_heads = 2
        mock_hf_pretrained.config.initializer_range = 0.02
        mock_hf_pretrained.config.rms_norm_eps = 1e-6
        mock_hf_pretrained.config.vocab_size = 151674
        mock_hf_pretrained.config.max_position_embeddings = 32768
        mock_hf_pretrained.config.rope_theta = 1000000.0

        provider = internvl_bridge.provider_bridge(mock_hf_pretrained)

        # Should still work with fallback
        assert provider.num_layers == 24
        assert provider.hidden_size == 896


class TestInternVLBridgeMappingRegistry:
    """Test mapping_registry method functionality."""

    def test_mapping_registry_returns_registry(self, internvl_bridge):
        """Test mapping_registry returns a MegatronMappingRegistry."""
        registry = internvl_bridge.mapping_registry()

        assert isinstance(registry, MegatronMappingRegistry)

    def test_mapping_registry_has_embeddings_mapping(self, internvl_bridge):
        """Test mapping_registry includes embeddings mapping."""
        registry = internvl_bridge.mapping_registry()
        
        # Check that registry has mappings
        assert len(registry.mappings) > 0

    def test_mapping_registry_has_attention_mappings(self, internvl_bridge):
        """Test mapping_registry includes attention-related mappings."""
        registry = internvl_bridge.mapping_registry()
        
        # Registry should contain QKV and projection mappings
        mapping_types = [type(m).__name__ for m in registry.mappings]
        assert "QKVMapping" in mapping_types
        assert "AutoMapping" in mapping_types

    def test_mapping_registry_has_mlp_mappings(self, internvl_bridge):
        """Test mapping_registry includes MLP-related mappings."""
        registry = internvl_bridge.mapping_registry()
        
        # Registry should contain gated MLP mappings
        mapping_types = [type(m).__name__ for m in registry.mappings]
        assert "GatedMLPMapping" in mapping_types

    def test_mapping_registry_has_vision_mappings(self, internvl_bridge):
        """Test mapping_registry includes vision model mappings."""
        registry = internvl_bridge.mapping_registry()
        
        # Registry should contain replicated mappings for vision components
        mapping_types = [type(m).__name__ for m in registry.mappings]
        assert "ReplicatedMapping" in mapping_types


class TestInternVLBridgeIntegration:
    """Integration tests for InternVL bridge."""

    def test_bridge_full_pipeline(self, internvl_bridge, mock_hf_pretrained):
        """Test complete bridge pipeline from config to mappings."""
        # Get provider
        provider = internvl_bridge.provider_bridge(mock_hf_pretrained)
        assert isinstance(provider, InternVLModelProvider)

        # Get mapping registry
        registry = internvl_bridge.mapping_registry()
        assert isinstance(registry, MegatronMappingRegistry)

        # Verify provider and registry are compatible
        assert provider.num_layers > 0
        assert len(registry.mappings) > 0

    def test_bridge_with_custom_vision_config(self, internvl_bridge, mock_hf_pretrained):
        """Test bridge with custom vision configuration."""
        custom_vision_config = InternVLVisionConfig(
            hidden_size=1024,
            intermediate_size=4096,
            num_hidden_layers=24,
            num_attention_heads=16,
            image_size=448,
            patch_size=14,
        )
        mock_hf_pretrained.config.vision_config = custom_vision_config

        provider = internvl_bridge.provider_bridge(mock_hf_pretrained)

        assert provider.vision_config.hidden_size == 1024
        assert provider.vision_config.num_hidden_layers == 24

