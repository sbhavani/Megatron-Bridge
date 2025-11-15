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

from unittest.mock import Mock

import pytest

from megatron.bridge.models.kimi_vl.kimi_vl_provider import KimiVLModelProvider
from megatron.bridge.models.kimi_vl.modeling_kimi_vl import KimiVLModel


@pytest.fixture
def kimi_vl_provider():
    """Create a minimal KimiVLModelProvider instance."""
    return KimiVLModelProvider(
        num_layers=2,
        hidden_size=256,
        num_attention_heads=4,
    )


class TestKimiVLModelProviderInitialization:
    """Test KimiVLModelProvider initialization."""

    def test_provider_initialization(self, kimi_vl_provider):
        """Test that provider can be initialized with minimal config."""
        assert isinstance(kimi_vl_provider, KimiVLModelProvider)
        assert kimi_vl_provider.num_layers == 2
        assert kimi_vl_provider.hidden_size == 256
        assert kimi_vl_provider.num_attention_heads == 4

    def test_provider_vl_defaults(self, kimi_vl_provider):
        """Test that provider has correct VL-specific defaults."""
        # VL models should not scatter embeddings for sequence parallelism
        assert kimi_vl_provider.scatter_embedding_sequence_parallel is False

        # Check default token IDs
        assert kimi_vl_provider.bos_token_id == 0
        assert kimi_vl_provider.eos_token_id == 1
        assert kimi_vl_provider.vision_start_token_id == 151652
        assert kimi_vl_provider.vision_end_token_id == 151653
        assert kimi_vl_provider.image_token_id == 151655

    def test_provider_freeze_defaults(self, kimi_vl_provider):
        """Test that provider has correct freeze option defaults."""
        assert kimi_vl_provider.freeze_language_model is False
        assert kimi_vl_provider.freeze_vision_model is False
        assert kimi_vl_provider.freeze_vision_projection is False


class TestKimiVLModelProviderProvide:
    """Test KimiVLModelProvider provide method."""

    def test_provide_returns_kimi_vl_model(self, kimi_vl_provider):
        """Test that provide method returns KimiVLModel instance."""
        model = kimi_vl_provider.provide()
        assert isinstance(model, KimiVLModel)

    def test_provide_with_freeze_options(self):
        """Test that provide method applies freeze options."""
        provider = KimiVLModelProvider(
            num_layers=2,
            hidden_size=256,
            num_attention_heads=4,
            freeze_language_model=True,
        )

        # Mock the freeze method to verify it's called
        with pytest.mock.patch.object(KimiVLModel, "freeze") as mock_freeze:
            model = provider.provide()
            # freeze should be called with the provider's freeze settings
            mock_freeze.assert_called_once_with(
                freeze_language_model=True, freeze_vision_model=False, freeze_vision_projection=False
            )


class TestKimiVLModelProviderLanguageModel:
    """Test KimiVLModelProvider provide_language_model method."""

    def test_provide_language_model(self, kimi_vl_provider):
        """Test that provide_language_model returns GPT model."""
        # This should return the base language model without VL wrapper
        from megatron.core.models.gpt import GPTModel

        language_model = kimi_vl_provider.provide_language_model()
        assert isinstance(language_model, GPTModel)


class TestKimiVLModelProviderCustomization:
    """Test KimiVLModelProvider with custom configurations."""

    def test_provider_with_custom_vision_config(self):
        """Test provider with custom vision configuration."""
        vision_config = Mock()
        vision_config.hidden_size = 1024

        provider = KimiVLModelProvider(
            num_layers=2, hidden_size=256, num_attention_heads=4, vision_config=vision_config
        )

        assert provider.vision_config.hidden_size == 1024

    def test_provider_with_custom_token_ids(self):
        """Test provider with custom token IDs."""
        provider = KimiVLModelProvider(
            num_layers=2,
            hidden_size=256,
            num_attention_heads=4,
            bos_token_id=100,
            eos_token_id=101,
            image_token_id=200,
        )

        assert provider.bos_token_id == 100
        assert provider.eos_token_id == 101
        assert provider.image_token_id == 200

    def test_provider_inherits_deepseek_v3_properties(self):
        """Test that provider inherits DeepSeek V3 properties."""
        provider = KimiVLModelProvider(num_layers=2, hidden_size=256, num_attention_heads=4)

        # Check DeepSeek V3 specific properties
        assert hasattr(provider, "moe_router_score_function")
        assert hasattr(provider, "q_lora_rank")
        assert hasattr(provider, "kv_lora_rank")
        assert provider.normalization == "RMSNorm"
        assert provider.gated_linear_unit is True
