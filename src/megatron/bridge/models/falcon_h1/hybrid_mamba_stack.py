# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Extended MambaStack with support for parallel hybrid layers (Falcon H1)."""

from dataclasses import dataclass
from typing import Union, Optional

import torch

from megatron.core.ssm.mamba_block import MambaStack, MambaStackSubmodules
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.process_groups_config import ProcessGroupCollection

from megatron.bridge.models.falcon_h1.parallel_hybrid_layer import ParallelHybridLayer
from megatron.bridge.models.falcon_h1.layer_utils import LayerType


def allocate_hybrid_layers(
    total_layers_count: int,
    target_attention_ratio: float,
    target_mlp_ratio: float,
    target_parallel_hybrid_ratio: float,
    override_pattern: str = None,
) -> list:
    """Allocate layer types for hybrid architecture.

    Args:
        total_layers_count: Total number of layers
        target_attention_ratio: Target ratio of attention layers (0.0-1.0)
        target_mlp_ratio: Target ratio of MLP layers (0.0-1.0)
        target_parallel_hybrid_ratio: Target ratio of parallel hybrid layers (0.0-1.0)
        override_pattern: Override pattern string (e.g., "P-P-P-")

    Returns:
        List of layer type symbols
    """
    assert total_layers_count > 0
    assert target_attention_ratio >= 0.0 and target_attention_ratio <= 1.0
    assert target_mlp_ratio >= 0.0 and target_mlp_ratio <= 1.0
    assert target_parallel_hybrid_ratio >= 0.0 and target_parallel_hybrid_ratio <= 1.0
    assert target_attention_ratio + target_mlp_ratio + target_parallel_hybrid_ratio <= 1.0

    if override_pattern is not None:
        layer_type_list = list(override_pattern)
        if len(layer_type_list) != total_layers_count:
            raise ValueError(
                f"Override pattern length {len(layer_type_list)} does not match "
                f"total layers {total_layers_count}"
            )
        for symbol in layer_type_list:
            if symbol not in LayerType.VALID:
                raise ValueError(f"Invalid symbol '{symbol}' in override pattern")
        return layer_type_list

    # Auto-allocate based on ratios
    layer_type_list = [LayerType.MAMBA] * total_layers_count

    # Allocate attention layers
    attention_count = round(total_layers_count * target_attention_ratio)
    if attention_count > 0:
        for i in range(attention_count):
            idx = int(i * total_layers_count / attention_count)
            layer_type_list[idx] = LayerType.ATTENTION

    # Allocate MLP layers
    mlp_count = round(total_layers_count * target_mlp_ratio)
    if mlp_count > 0:
        available_indices = [i for i, t in enumerate(layer_type_list) if t == LayerType.MAMBA]
        for i in range(min(mlp_count, len(available_indices))):
            idx = available_indices[i]
            layer_type_list[idx] = LayerType.MLP

    # Allocate parallel hybrid layers
    parallel_count = round(total_layers_count * target_parallel_hybrid_ratio)
    if parallel_count > 0:
        available_indices = [i for i, t in enumerate(layer_type_list) if t == LayerType.MAMBA]
        for i in range(min(parallel_count, len(available_indices))):
            idx = available_indices[i]
            layer_type_list[idx] = LayerType.PARALLEL

    return layer_type_list


@dataclass
class HybridMambaStackSubmodules(MambaStackSubmodules):
    """Extended MambaStackSubmodules with parallel_hybrid_layer."""
    parallel_hybrid_layer: Union[ModuleSpec, type] = IdentityOp


class HybridMambaStack(MambaStack):
    """Extended MambaStack with support for parallel hybrid layers.

    This extends the standard MambaStack to support the 'P' (parallel hybrid)
    layer type needed for Falcon H1.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: HybridMambaStackSubmodules,
        vocab_size: int,
        max_sequence_length: int,
        pre_process: bool = True,
        hybrid_attention_ratio: float = 0.0,
        hybrid_mlp_ratio: float = 0.0,
        parallel_hybrid_ratio: float = 0.0,
        hybrid_override_pattern: str = None,
        post_layer_norm: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        position_embedding_type: str = 'learned_absolute',
        rotary_percent: float = 1.0,
        rotary_base: int = 10000,
        seq_len_interpolation_factor: Optional[float] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        """Initialize HybridMambaStack.

        Args:
            config: Transformer configuration
            submodules: HybridMambaStackSubmodules with parallel_hybrid_layer spec
            vocab_size: Vocabulary size
            max_sequence_length: Maximum sequence length
            pre_process: Whether to include embedding layer
            hybrid_attention_ratio: Ratio of attention layers
            hybrid_mlp_ratio: Ratio of MLP layers
            parallel_hybrid_ratio: Ratio of parallel hybrid layers
            hybrid_override_pattern: Override pattern (e.g., "P-P-P-")
            post_layer_norm: Whether to include final layer norm
            post_process: Whether to include output layer
            fp16_lm_cross_entropy: Use FP16 for cross entropy
            parallel_output: Parallel output
            share_embeddings_and_output_weights: Share embeddings
            position_embedding_type: Type of position embedding
            rotary_percent: Rotary embedding percent
            rotary_base: Rotary base
            seq_len_interpolation_factor: Sequence length interpolation factor
            pg_collection: Process group collection
        """
        # Store parallel_hybrid_ratio before calling super().__init__()
        self.parallel_hybrid_ratio = parallel_hybrid_ratio
        self.parallel_hybrid_layer_spec = submodules.parallel_hybrid_layer

        # Call parent __init__ but it won't know about parallel hybrid
        # We need to build layers ourselves
        # First, skip MambaStack.__init__ and go straight to MegatronModule
        from megatron.core.transformer.module import MegatronModule
        MegatronModule.__init__(self, config)

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.pg_collection = pg_collection

        self.config = config
        self.pre_process = pre_process
        self.post_process = post_process
        self.post_layer_norm = post_layer_norm
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.fp16_lm_cross_entropy = fp16_lm_cross_entropy
        self.parallel_output = parallel_output
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.position_embedding_type = position_embedding_type
        self.rotary_percent = rotary_percent
        self.rotary_base = rotary_base
        self.seq_len_interpolation_factor = seq_len_interpolation_factor

        self.hybrid_attention_ratio = hybrid_attention_ratio
        self.hybrid_mlp_ratio = hybrid_mlp_ratio
        self.hybrid_override_pattern = hybrid_override_pattern

        # Allocate layers including parallel hybrid
        layer_type_list = allocate_hybrid_layers(
            self.config.num_layers,
            self.hybrid_attention_ratio,
            self.hybrid_mlp_ratio,
            self.parallel_hybrid_ratio,
            self.hybrid_override_pattern,
        )

        # Build layers
        self.layers = torch.nn.ModuleList()
        pp_layer_offset = 0

        for i, layer_type in enumerate(layer_type_list):
            if layer_type == LayerType.MAMBA:
                layer = build_module(
                    submodules.mamba_layer,
                    config=self.config,
                    layer_number=i + 1 + pp_layer_offset,
                    pg_collection=pg_collection,
                )
            elif layer_type == LayerType.ATTENTION:
                layer = build_module(
                    submodules.attention_layer,
                    config=self.config,
                    layer_number=i + 1 + pp_layer_offset,
                    pg_collection=pg_collection,
                )
            elif layer_type == LayerType.MLP:
                layer = build_module(
                    submodules.mlp_layer,
                    config=self.config,
                    layer_number=i + 1 + pp_layer_offset,
                    pg_collection=pg_collection,
                )
            elif layer_type == LayerType.PARALLEL:
                # Build parallel hybrid layer
                layer = build_module(
                    self.parallel_hybrid_layer_spec,
                    config=self.config,
                    layer_number=i + 1 + pp_layer_offset,
                    pg_collection=pg_collection,
                )
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")

            self.layers.append(layer)

        # Final layer norm
        if self.post_layer_norm:
            self.final_norm = build_module(
                submodules.mamba_layer.submodules.mixer.submodules.in_proj,
                config=self.config,
                hidden_size=self.config.hidden_size,
                eps=self.config.layernorm_epsilon,
            )

        # Embedding
        if self.pre_process:
            from megatron.core.transformer.token_embedding import TokenEmbedding
            self.embedding = TokenEmbedding(
                self.config,
                self.vocab_size,
                self.max_sequence_length,
                embedding_dropout_prob=self.config.hidden_dropout,
                position_embedding_type=self.position_embedding_type,
                pg_collection=pg_collection,
            )

        # Output layer
        if self.post_process:
            if self.share_embeddings_and_output_weights:
                self.output_layer = None
            else:
                from megatron.core.transformer.linear_fc import LinearFC
                self.output_layer = LinearFC(
                    config,
                    self.config.hidden_size,
                    self.vocab_size,
                    bias=False,
                    skip_bias_add=False,
                    skip_weight_param_allocation=self.pre_process and self.share_embeddings_and_output_weights,
                    pg_collection=pg_collection,
                )

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        inference_context: Optional = None,
        **kwargs
    ):
        """Forward pass with support for parallel hybrid layers."""
        # Get embeddings
        if self.pre_process:
            hidden_states = self.embedding(input_ids, position_ids)
        else:
            hidden_states = input_ids

        # Prepare rotary embeddings if using RoPE
        rotary_pos_emb = None
        if self.position_embedding_type == 'rope':
            from megatron.core.models.common.embeddings.rotary_pos_embedding import apply_rotary_pos_emb
            from megatron.core.models.common.embeddings.rope_utils import get_pos_emb_on_this_cp_rank
            # This is simplified - in real implementation would handle CP ranks properly
            rotary_seq_len = self.max_sequence_length
            rotary_pos_emb = self.embedding.rotary_pos_emb(rotary_seq_len) if hasattr(self.embedding, 'rotary_pos_emb') else None

        # Process through layers
        for layer in self.layers:
            if isinstance(layer, ParallelHybridLayer):
                hidden_states = layer(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    inference_context=inference_context,
                    rotary_pos_emb=rotary_pos_emb,
                )
            elif isinstance(layer, TransformerLayer):
                hidden_states, _ = layer(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    inference_context=inference_context,
                    rotary_pos_emb=rotary_pos_emb,
                )
            else:
                # Mamba or MLP layer
                hidden_states = layer(
                    hidden_states=hidden_states,
                    inference_context=inference_context,
                )

        # Final norm
        if self.post_layer_norm:
            hidden_states = self.final_norm(hidden_states)

        # Output layer
        if self.post_process:
            if self.output_layer is None:
                # Share embeddings
                logits = torch.matmul(hidden_states, self.embedding.word_embeddings.weight.t())
            else:
                logits, _ = self.output_layer(hidden_states)
            return logits

        return hidden_states
