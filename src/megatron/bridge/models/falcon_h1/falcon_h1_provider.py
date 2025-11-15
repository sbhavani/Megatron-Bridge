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

import logging
from dataclasses import dataclass
from typing import Callable, Optional, Union

from megatron.core.transformer import ModuleSpec

from megatron.bridge.models.mamba.mamba_provider import MambaModelProvider
from megatron.bridge.models.falcon_h1.falcon_h1_layer_specs import get_falcon_h1_mamba_stack_spec


logger = logging.getLogger(__name__)


def get_default_falcon_h1_stack_spec():
    """Return the Falcon H1-specific mamba stack spec.

    This is a named function (not a lambda) to allow proper serialization
    and reconstruction from checkpoints. Named functions can be imported
    via their module path, unlike lambdas.

    Returns:
        Falcon H1 mamba stack specification with parallel hybrid support
    """
    return get_falcon_h1_mamba_stack_spec()


@dataclass
class FalconH1ModelProvider(MambaModelProvider):
    """Configuration and provider for Falcon H1 hybrid models.

    Falcon H1 uses a hybrid architecture with alternating layers:
    - Even layers (0, 2, 4, ...): ParallelHybridLayer (Mamba mixer + Self-attention in parallel)
    - Odd layers (1, 3, 5, ...): MLP-only layers

    This provider inherits from MambaModelProvider and overrides Falcon H1-specific defaults.
    """

    # Override Mamba defaults for Falcon H1
    mamba_num_groups: int = 1
    mamba_state_dim: int = 128
    mamba_head_dim: int = 64

    # Attention parameters (for hybrid layers) - Falcon H1 uses attention unlike pure Mamba
    num_attention_heads: int = 8
    num_query_groups: int = 1  # GQA

    # Hybrid architecture ratios - Falcon H1 specific pattern
    hybrid_attention_ratio: float = 0.0  # No standalone attention layers
    hybrid_mlp_ratio: float = 0.5  # Half of layers are MLP-only
    parallel_hybrid_ratio: float = 0.5  # Half of layers are parallel hybrid (Mamba+Attention)
    hybrid_override_pattern: Optional[str] = None

    # Position embeddings - Falcon H1 uses RoPE unlike pure Mamba
    position_embedding_type: str = "rope"
    rotary_base: int = 10000

    # Use Falcon H1-specific stack spec instead of default Mamba spec
    mamba_stack_spec: Union[ModuleSpec, Callable[[], ModuleSpec]] = get_default_falcon_h1_stack_spec

    # Note: All other fields (gated_linear_unit, normalization, etc.) are inherited from MambaModelProvider
    # with their appropriate defaults

    def provide(self, pre_process=None, post_process=None, vp_stage=None):
        """Configure and instantiate a Mamba model with Falcon H1 architecture.

        Overrides parent to pass parallel_hybrid_ratio to HybridMambaStack.

        Args:
            pre_process: Whether to include pre-processing in the model
            post_process: Whether to include post-processing in the model
            vp_stage: Virtual pipeline stage

        Returns:
            MCoreMambaModel: Configured Megatron Core Mamba model with HybridMambaStack
        """
        from megatron.core import parallel_state
        from megatron.core.models.mamba import MambaModel as MCoreMambaModel
        from megatron.bridge.utils.vocab_utils import calculate_padded_vocab_size

        mamba_stack_spec = self.mamba_stack_spec
        if not isinstance(mamba_stack_spec, ModuleSpec):
            mamba_stack_spec = mamba_stack_spec()

        assert getattr(self, "virtual_pipeline_model_parallel_size", None) is None and vp_stage is None, (
            "Virtual pipeline model parallelism is temporarily unsupported in SSM/Mamba "
            "models due to upstream MCore MambaModel API dependency"
        )

        assert self.vocab_size is not None, "vocab_size must be configured before calling provide()"
        if self.should_pad_vocab:
            padded_vocab_size = calculate_padded_vocab_size(
                self.vocab_size, self.make_vocab_size_divisible_by, self.tensor_model_parallel_size
            )
        else:
            padded_vocab_size = self.vocab_size

        # Call with parallel_hybrid_ratio for HybridMambaStack
        return MCoreMambaModel(
            self,
            mamba_stack_spec=mamba_stack_spec,
            vocab_size=padded_vocab_size,
            max_sequence_length=self.seq_length,
            hybrid_attention_ratio=self.hybrid_attention_ratio,
            hybrid_mlp_ratio=self.hybrid_mlp_ratio,
            parallel_hybrid_ratio=self.parallel_hybrid_ratio,  # Falcon H1 specific
            hybrid_override_pattern=self.hybrid_override_pattern,
            fp16_lm_cross_entropy=self.fp16_lm_cross_entropy,
            parallel_output=self.parallel_output,
            share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
            position_embedding_type=self.position_embedding_type,
            rotary_percent=self.rotary_percent,
            rotary_base=self.rotary_base,
            seq_len_interpolation_factor=self.seq_len_interpolation_factor,
            pre_process=pre_process or parallel_state.is_pipeline_first_stage(),
            post_process=post_process or parallel_state.is_pipeline_last_stage(),
        )


@dataclass
class FalconH1ModelProvider1B(FalconH1ModelProvider):
    """Configuration for a 1B parameter Falcon H1 model.

    Based on Falcon3-1B-Base architecture with hybrid Mamba+Attention layers.
    """

    num_layers: int = 20
    hidden_size: int = 1024
    ffn_hidden_size: int = 4096
    num_attention_heads: int = 8
    num_query_groups: int = 2  # GQA with 2 KV heads
    mamba_num_groups: int = 1
    mamba_state_dim: int = 128
    mamba_head_dim: int = 64
    seq_length: int = 8192
    rotary_base: int = 10000
    make_vocab_size_divisible_by: int = 128

    # Falcon H1 hybrid pattern: alternating Parallel (P) and MLP (-)
    # P = ParallelHybridLayer (Mamba + Attention)
    # - = MLP layer
    hybrid_override_pattern: str = "P-" * 10  # 20 layers: P-P-P-P-P-P-P-P-P-P-
