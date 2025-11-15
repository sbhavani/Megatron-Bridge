# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Utility functions for Falcon H1 hybrid layer allocation and building."""

from typing import Optional

from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.process_groups_config import ProcessGroupCollection

from megatron.bridge.models.falcon_h1.parallel_hybrid_layer import ParallelHybridLayer


# Layer type symbols for Falcon H1
class LayerType:
    """Layer type constants for hybrid architecture."""
    MAMBA = "M"
    ATTENTION = "*"
    MLP = "-"
    PARALLEL = "P"  # Parallel hybrid (Mamba + Attention)
    VALID = {MAMBA, ATTENTION, MLP, PARALLEL}


def generate_alternating_pattern(
    num_layers: int,
    even_type: str = LayerType.PARALLEL,
    odd_type: str = LayerType.MLP,
) -> str:
    """Generate alternating pattern string for Falcon H1.

    Args:
        num_layers: Total number of layers
        even_type: Layer type for even indices (0, 2, 4, ...)
        odd_type: Layer type for odd indices (1, 3, 5, ...)

    Returns:
        Pattern string (e.g., "P-P-P-" for 8 layers)
    """
    pattern = ""
    for i in range(num_layers):
        pattern += even_type if i % 2 == 0 else odd_type
    return pattern


def validate_layer_pattern(pattern: str, num_layers: Optional[int] = None) -> bool:
    """Validate a layer pattern string.

    Args:
        pattern: Pattern string to validate (e.g., "P-P-P-")
        num_layers: If provided, check that pattern length matches

    Returns:
        True if valid, raises ValueError if invalid
    """
    if not pattern:
        raise ValueError("Pattern cannot be empty")

    for i, symbol in enumerate(pattern):
        if symbol not in LayerType.VALID:
            raise ValueError(
                f"Invalid symbol '{symbol}' at position {i} in pattern. "
                f"Valid symbols: {LayerType.VALID}"
            )

    if num_layers is not None and len(pattern) != num_layers:
        raise ValueError(
            f"Pattern length {len(pattern)} does not match "
            f"expected num_layers {num_layers}"
        )

    return True


def allocate_falcon_h1_layers(
    num_layers: int,
    override_pattern: Optional[str] = None,
) -> list:
    """Allocate layer types for Falcon H1 architecture.

    Falcon H1 uses a simple alternating pattern:
    - Even layers: Parallel (Mamba + Attention)
    - Odd layers: MLP

    Args:
        num_layers: Total number of layers
        override_pattern: Optional pattern override (e.g., "P-P-P-")

    Returns:
        List of layer type symbols
    """
    if override_pattern is not None:
        validate_layer_pattern(override_pattern, num_layers)
        return list(override_pattern)

    # Default Falcon H1 pattern: alternating P and -
    return list(generate_alternating_pattern(num_layers))


def build_parallel_hybrid_layer(
    spec: ModuleSpec,
    config: TransformerConfig,
    layer_number: int,
    pg_collection: Optional[ProcessGroupCollection] = None,
) -> ParallelHybridLayer:
    """Build a parallel hybrid layer from spec.

    Args:
        spec: Module specification for ParallelHybridLayer
        config: Transformer configuration
        layer_number: Layer index
        pg_collection: Process group collection

    Returns:
        Configured ParallelHybridLayer instance
    """
    return build_module(
        spec,
        config=config,
        layer_number=layer_number,
        pg_collection=pg_collection,
    )
