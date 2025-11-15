# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Falcon H1 specific layer specifications with parallel hybrid layers."""

from megatron.core.extensions.transformer_engine import (
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    TERowParallelLinear,
)
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.ssm.mamba_mixer import MambaMixer, MambaMixerSubmodules
from megatron.core.ssm.mlp_layer import MLPLayer
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayerSubmodules

from megatron.bridge.models.falcon_h1.parallel_hybrid_layer import ParallelHybridLayer, ParallelHybridLayerSubmodules
from megatron.bridge.models.falcon_h1.hybrid_mamba_stack import HybridMambaStack, HybridMambaStackSubmodules


def get_falcon_h1_mamba_stack_spec():
    """Get the Falcon H1-specific mamba stack spec with parallel hybrid layers.

    Falcon H1 only uses two layer types:
    - MLP layers (odd layers: 1, 3, 5, ...)
    - Parallel hybrid layers (even layers: 0, 2, 4, ...)

    Returns:
        ModuleSpec for HybridMambaStack configured for Falcon H1 architecture
    """
    return ModuleSpec(
        module=HybridMambaStack,
        submodules=HybridMambaStackSubmodules(
            # MLP layer for Falcon H1 odd layers
            mlp_layer=ModuleSpec(
                module=MLPLayer,
                submodules=TransformerLayerSubmodules(
                    mlp=ModuleSpec(
                        module=MLP,
                        submodules=MLPSubmodules(
                            linear_fc1=TELayerNormColumnParallelLinear,
                            linear_fc2=TERowParallelLinear
                        ),
                    ),
                    mlp_bda=get_bias_dropout_add,
                ),
            ),
            # Parallel hybrid layer for Falcon H1 even layers (Mamba + Attention)
            parallel_hybrid_layer=ModuleSpec(
                module=ParallelHybridLayer,
                submodules=ParallelHybridLayerSubmodules(
                    mamba_mixer=ModuleSpec(
                        module=MambaMixer,
                        submodules=MambaMixerSubmodules(
                            in_proj=TELayerNormColumnParallelLinear,
                            out_proj=TERowParallelLinear
                        ),
                    ),
                    parallel_hybrid_bda=get_bias_dropout_add,
                    self_attention=ModuleSpec(
                        module=ModuleSpec(
                            module=SelfAttention,
                            params={"attn_mask_type": AttnMaskType.causal},
                            submodules=SelfAttentionSubmodules(
                                linear_qkv=TELayerNormColumnParallelLinear,
                                core_attention=TEDotProductAttention,
                                linear_proj=TERowParallelLinear,
                            ),
                        ),
                    ),
                    input_layernorm=TELayerNormColumnParallelLinear,
                ),
            ),
        ),
    )
