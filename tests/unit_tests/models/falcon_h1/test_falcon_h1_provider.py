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

import torch

from megatron.bridge.models.falcon_h1.falcon_h1_provider import (
    FalconH1ModelProvider,
    FalconH1ModelProvider1B,
)


class TestFalconH1ModelProvider:
    """Test cases for FalconH1ModelProvider class."""

    def test_falcon_h1_provider_initialization(self):
        """Test FalconH1ModelProvider can be initialized with default values."""
        provider = FalconH1ModelProvider(
            num_layers=12,
            hidden_size=768,
            num_attention_heads=8,
        )

        # Check required transformer config fields
        assert provider.num_layers == 12
        assert provider.hidden_size == 768
        assert provider.num_attention_heads == 8

        # Check Falcon H1-specific defaults
        assert provider.fp16_lm_cross_entropy is False
        assert provider.parallel_output is True
        assert provider.share_embeddings_and_output_weights is False
        assert provider.params_dtype == torch.bfloat16
        assert provider.fp16 is False
        assert provider.bf16 is True
        assert provider.mamba_num_groups == 1
        assert provider.mamba_state_dim == 128
        assert provider.mamba_head_dim == 64
        assert provider.num_query_groups == 1
        assert provider.hybrid_attention_ratio == 0.0
        assert provider.hybrid_mlp_ratio == 0.5
        assert provider.parallel_hybrid_ratio == 0.5
        assert provider.hybrid_override_pattern is None
        assert provider.seq_length == 8192
        assert provider.position_embedding_type == "rope"
        assert provider.rotary_percent == 1.0
        assert provider.rotary_base == 10000
        assert provider.seq_len_interpolation_factor is None
        assert provider.apply_rope_fusion is True
        assert provider.make_vocab_size_divisible_by == 128
        assert provider.gated_linear_unit is True
        assert provider.normalization == "RMSNorm"
        assert provider.add_bias_linear is False
        assert provider.hidden_dropout == 0.0
        assert provider.attention_dropout == 0.0
        assert provider.layernorm_epsilon == 1e-5
        assert provider.deallocate_pipeline_outputs is True
        assert provider.bias_dropout_fusion is True
        assert provider.cross_entropy_loss_fusion is True
        assert provider.vocab_size is None

    def test_falcon_h1_provider_with_hybrid_configuration(self):
        """Test FalconH1ModelProvider with custom hybrid configuration."""
        provider = FalconH1ModelProvider(
            num_layers=12,
            hidden_size=768,
            num_attention_heads=8,
            hybrid_mlp_ratio=0.5,
            parallel_hybrid_ratio=0.5,
            hybrid_override_pattern="P-P-P-P-P-P-",
        )

        assert provider.hybrid_mlp_ratio == 0.5
        assert provider.parallel_hybrid_ratio == 0.5
        assert provider.hybrid_override_pattern == "P-P-P-P-P-P-"

    def test_provide_method_basic(self):
        """Test the provide method creates a Mamba model."""
        provider = FalconH1ModelProvider(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=8,
            vocab_size=1000,
            tensor_model_parallel_size=1,
            make_vocab_size_divisible_by=128,
        )

        # Mock dependencies
        with patch("megatron.bridge.models.falcon_h1.falcon_h1_provider.parallel_state") as mock_ps:
            with patch("megatron.bridge.models.falcon_h1.falcon_h1_provider.calculate_padded_vocab_size", return_value=1024):
                with patch("megatron.bridge.models.falcon_h1.falcon_h1_provider.MCoreMambaModel") as mock_model:
                    mock_ps.is_pipeline_first_stage.return_value = True
                    mock_ps.is_pipeline_last_stage.return_value = True
                    mock_instance = Mock()
                    mock_model.return_value = mock_instance

                    result = provider.provide()

                    assert result == mock_instance
                    mock_model.assert_called_once()

    def test_provide_method_with_vocab_padding(self):
        """Test provide method calculates padded vocab size when padding is enabled."""
        provider = FalconH1ModelProvider(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=8,
            vocab_size=50000,
            tensor_model_parallel_size=8,
            make_vocab_size_divisible_by=128,
            should_pad_vocab=True,  # Enable padding
        )

        with patch("megatron.bridge.models.falcon_h1.falcon_h1_provider.parallel_state") as mock_ps:
            with patch(
                "megatron.bridge.models.falcon_h1.falcon_h1_provider.calculate_padded_vocab_size", return_value=50176
            ) as mock_calc_vocab:
                with patch("megatron.bridge.models.falcon_h1.falcon_h1_provider.MCoreMambaModel") as mock_model:
                    mock_ps.is_pipeline_first_stage.return_value = True
                    mock_ps.is_pipeline_last_stage.return_value = True
                    mock_instance = Mock()
                    mock_model.return_value = mock_instance

                    _ = provider.provide()

                    # Verify calculate_padded_vocab_size was called with correct parameters
                    mock_calc_vocab.assert_called_once_with(50000, 128, 8)
                    # Verify model was created with padded vocab size
                    call_kwargs = mock_model.call_args.kwargs
                    assert call_kwargs["vocab_size"] == 50176

    def test_provide_method_no_vocab_padding(self):
        """Test provide method uses original vocab size when padding is disabled."""
        provider = FalconH1ModelProvider(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=8,
            vocab_size=50000,
            tensor_model_parallel_size=8,
            make_vocab_size_divisible_by=128,
            should_pad_vocab=False,  # Disable padding
        )

        with patch("megatron.bridge.models.falcon_h1.falcon_h1_provider.parallel_state") as mock_ps:
            with patch("megatron.bridge.models.falcon_h1.falcon_h1_provider.calculate_padded_vocab_size") as mock_calc_vocab:
                with patch("megatron.bridge.models.falcon_h1.falcon_h1_provider.MCoreMambaModel") as mock_model:
                    mock_ps.is_pipeline_first_stage.return_value = True
                    mock_ps.is_pipeline_last_stage.return_value = True
                    mock_instance = Mock()
                    mock_model.return_value = mock_instance

                    _ = provider.provide()

                    # Verify calculate_padded_vocab_size was NOT called
                    mock_calc_vocab.assert_not_called()
                    # Verify model was created with original vocab size
                    call_kwargs = mock_model.call_args.kwargs
                    assert call_kwargs["vocab_size"] == 50000

    def test_provide_method_pipeline_stages(self):
        """Test provide method respects pipeline stage arguments."""
        provider = FalconH1ModelProvider(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=8,
            vocab_size=1000,
            tensor_model_parallel_size=1,
            make_vocab_size_divisible_by=128,
        )

        with patch("megatron.bridge.models.falcon_h1.falcon_h1_provider.parallel_state") as mock_ps:
            with patch("megatron.bridge.models.falcon_h1.falcon_h1_provider.calculate_padded_vocab_size", return_value=1024):
                with patch("megatron.bridge.models.falcon_h1.falcon_h1_provider.MCoreMambaModel") as mock_mamba:
                    # Test default behavior - uses parallel_state
                    mock_ps.is_pipeline_first_stage.return_value = False
                    mock_ps.is_pipeline_last_stage.return_value = True
                    mock_instance = Mock()
                    mock_mamba.return_value = mock_instance

                    provider.provide()

                    # Check the model was called with pipeline stages from parallel_state
                    call_kwargs = mock_mamba.call_args.kwargs
                    assert call_kwargs["pre_process"] is False
                    assert call_kwargs["post_process"] is True

    def test_provide_method_virtual_pipeline_error(self):
        """Test provide method raises error for virtual pipeline."""
        provider = FalconH1ModelProvider(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=8,
            vocab_size=1000,
        )
        provider.virtual_pipeline_model_parallel_size = 2  # Set virtual pipeline

        with patch("megatron.bridge.models.falcon_h1.falcon_h1_provider.parallel_state"):
            with patch("megatron.bridge.models.falcon_h1.falcon_h1_provider.MCoreMambaModel"):
                # Should raise AssertionError for virtual pipeline
                try:
                    provider.provide(vp_stage=0)
                    assert False, "Expected AssertionError for virtual pipeline"
                except AssertionError as e:
                    assert "Virtual pipeline model parallelism is temporarily unsupported" in str(e)

    def test_mamba_stack_spec_callable(self):
        """Test that mamba_stack_spec can be a callable."""

        def custom_stack_spec():
            spec = Mock()
            spec.info = "custom spec"
            return spec

        provider = FalconH1ModelProvider(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=8,
            vocab_size=1000,
            tensor_model_parallel_size=1,
            make_vocab_size_divisible_by=128,
            mamba_stack_spec=custom_stack_spec,
        )

        with patch("megatron.bridge.models.falcon_h1.falcon_h1_provider.parallel_state"):
            with patch("megatron.bridge.models.falcon_h1.falcon_h1_provider.calculate_padded_vocab_size", return_value=1024):
                with patch("megatron.bridge.models.falcon_h1.falcon_h1_provider.MCoreMambaModel") as mock_mamba:
                    mock_instance = Mock()
                    mock_mamba.return_value = mock_instance

                    provider.provide()

                    # The custom_stack_spec should have been called
                    assert provider.mamba_stack_spec == custom_stack_spec
                    spec_call_kwarg = mock_mamba.call_args.kwargs["mamba_stack_spec"]
                    assert isinstance(spec_call_kwarg, Mock)
                    assert spec_call_kwarg.info == "custom spec"

    def test_minimal_configuration(self):
        """Test that minimal configuration works."""
        # FalconH1ModelProvider should work with minimal required fields
        provider = FalconH1ModelProvider(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=8,
        )
        assert provider.num_layers == 2
        assert provider.hidden_size == 128
        assert provider.num_attention_heads == 8

    def test_falcon_h1_specific_configuration(self):
        """Test Falcon H1-specific configuration parameters."""
        provider = FalconH1ModelProvider(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=8,
            mamba_num_groups=8,
            mamba_state_dim=256,
            mamba_head_dim=128,
            num_query_groups=2,
            gated_linear_unit=True,
            normalization="RMSNorm",
            add_bias_linear=False,
        )

        assert provider.mamba_num_groups == 8
        assert provider.mamba_state_dim == 256
        assert provider.mamba_head_dim == 128
        assert provider.num_query_groups == 2
        assert provider.gated_linear_unit is True
        assert provider.normalization == "RMSNorm"
        assert provider.add_bias_linear is False

    def test_dropout_configuration(self):
        """Test dropout configuration."""
        provider = FalconH1ModelProvider(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=8,
            hidden_dropout=0.1,
            attention_dropout=0.2,
            layernorm_epsilon=1e-6,
        )

        assert provider.hidden_dropout == 0.1
        assert provider.attention_dropout == 0.2
        assert provider.layernorm_epsilon == 1e-6


class TestFalconH1ModelProvider1B:
    """Test cases for FalconH1ModelProvider1B class."""

    def test_falcon_h1_1b_default_configuration(self):
        """Test Falcon H1 1B model has correct default configuration."""
        provider = FalconH1ModelProvider1B()

        # Check Falcon H1 1B specific configuration
        assert provider.num_layers == 20
        assert provider.hidden_size == 1024
        assert provider.ffn_hidden_size == 4096
        assert provider.num_attention_heads == 8
        assert provider.num_query_groups == 2
        assert provider.mamba_num_groups == 1
        assert provider.mamba_state_dim == 128
        assert provider.mamba_head_dim == 64
        assert provider.seq_length == 8192
        assert provider.rotary_base == 10000
        assert provider.make_vocab_size_divisible_by == 128
        assert provider.hybrid_override_pattern == "P-" * 10  # 20 layers

    def test_falcon_h1_1b_override_configuration(self):
        """Test Falcon H1 1B model with overridden configuration."""
        provider = FalconH1ModelProvider1B(
            seq_length=4096,
            hidden_dropout=0.1,
        )

        # Check overridden values
        assert provider.seq_length == 4096
        assert provider.hidden_dropout == 0.1

        # Check defaults remain
        assert provider.num_layers == 20
        assert provider.hidden_size == 1024
        assert provider.num_attention_heads == 8

    def test_falcon_h1_1b_hybrid_pattern(self):
        """Test Falcon H1 1B hybrid pattern configuration."""
        provider = FalconH1ModelProvider1B()

        # Check that the hybrid pattern contains both Parallel and MLP layers
        pattern = provider.hybrid_override_pattern
        assert "P" in pattern  # Parallel hybrid layers
        assert "-" in pattern  # MLP layers
        assert len(pattern) == 20  # 20 layers total
        assert pattern.count("P") == 10  # 10 parallel hybrid layers
        assert pattern.count("-") == 10  # 10 MLP layers


class TestFalconH1ModelProviderInheritance:
    """Test inheritance relationships between Falcon H1 providers."""

    def test_all_providers_inherit_from_base(self):
        """Test all Falcon H1 model providers inherit from FalconH1ModelProvider."""
        providers = [
            FalconH1ModelProvider1B,
        ]

        for provider_class in providers:
            assert issubclass(provider_class, FalconH1ModelProvider)

    def test_provide_method_inherited(self):
        """Test that provide method works correctly in inherited classes."""
        # Test with Falcon H1 1B
        provider = FalconH1ModelProvider1B()

        # The provide method should be inherited from FalconH1ModelProvider
        assert hasattr(provider, "provide")
        assert callable(provider.provide)

    def test_hybrid_patterns_consistency(self):
        """Test that hybrid patterns are consistent across providers."""
        # Falcon H1 models should have both "P" (parallel) and "-" (MLP) in their pattern
        provider = FalconH1ModelProvider1B()
        pattern = provider.hybrid_override_pattern
        assert "P" in pattern  # Parallel hybrid layers
        assert "-" in pattern  # MLP layers
