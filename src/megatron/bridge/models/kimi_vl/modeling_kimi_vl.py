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

import types
from typing import Optional

import torch
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel import scatter_to_sequence_parallel_region
from megatron.core.transformer.module import MegatronModule
from torch import Tensor

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.utils.common_utils import hook_hf_module_setattr_for_tp_grad_sync


class KimiVLModel(MegatronModule):
    """
    Kimi VL Model. (Based on DeepSeek V3 language model with vision-language capabilities.)

    Args:
        config (GPTModelProvider):
            language model provider.
        pre_process (bool, optional):
            Include embedding layer (used with pipeline parallelism). Defaults to True.
        post_process (bool, optional):
            Include an output layer (used with pipeline parallelism). Defaults to True.
        vp_stage (int, optional):
            Virtual pipeline stage number. Defaults to None.
    """

    def __init__(
        self,
        config: GPTModelProvider,
        pre_process: bool = True,
        post_process: bool = True,
        vp_stage: Optional[int] = None,
    ) -> None:
        super().__init__(config=config)

        self.pre_process = pre_process
        self.post_process = post_process
        self.vp_stage = vp_stage

        # Initialize vision tower and multi-modal projector if in pre_process stage
        if pre_process:
            # Import the vision tower class dynamically based on config
            # For now, we'll use a placeholder approach - the actual vision tower
            # will be loaded during bridge conversion from HuggingFace model
            if hasattr(config, 'vision_config') and config.vision_config is not None:
                try:
                    # Try to load the vision model from the config
                    from transformers import AutoModel

                    # Create vision tower - this will be properly initialized during conversion
                    # For now, create a simple module that will be replaced
                    self.vision_tower = torch.nn.Module()
                    self.multi_modal_projector = torch.nn.Module()
                except ImportError:
                    # If transformers is not available, create placeholder
                    self.vision_tower = torch.nn.Module()
                    self.multi_modal_projector = torch.nn.Module()
            else:
                self.vision_tower = torch.nn.Module()
                self.multi_modal_projector = torch.nn.Module()

            # Ensure HF visual tower params are marked for TP grad sync
            if hasattr(self, 'vision_tower'):
                hook_hf_module_setattr_for_tp_grad_sync(self.vision_tower)
            if hasattr(self, 'multi_modal_projector'):
                hook_hf_module_setattr_for_tp_grad_sync(self.multi_modal_projector)

        # Initialize language model
        self.language_model = self.config.provide_language_model(
            pre_process=pre_process, post_process=post_process, vp_stage=vp_stage
        )

        # Finalize grad will need these to be bind with module
        self.share_embeddings_and_output_weights = config.share_embeddings_and_output_weights
        self.shared_embedding_or_output_weight = self.language_model.shared_embedding_or_output_weight

    def set_input_tensor(self, input_tensor) -> None:
        """Set model chunk input tensor."""
        self.language_model.set_input_tensor(input_tensor)

    def _extract_image_features(self, pixel_values, image_grid_hws):
        """
        Extract image features using the vision tower.

        This method will be bound from the HuggingFace model during conversion.
        """
        # Placeholder implementation - will be replaced with actual HF method
        raise NotImplementedError(
            "This method should be bound from the HuggingFace model during conversion"
        )

    def _merge_with_image_features(self, inputs_embeds, input_ids, image_features):
        """
        Merge text embeddings with image features.

        This method will be bound from the HuggingFace model during conversion.
        """
        # Placeholder implementation - will be replaced with actual HF method
        raise NotImplementedError(
            "This method should be bound from the HuggingFace model during conversion"
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_hws: Optional[torch.LongTensor] = None,
        labels: Tensor = None,
        inference_context: BaseInferenceContext = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        runtime_gather_output: Optional[bool] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        loss_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass for Kimi VL model.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            position_ids: Position IDs
            inputs_embeds: Pre-computed input embeddings (optional)
            pixel_values: Image pixel values
            image_grid_hws: Image grid heights and widths
            labels: Labels for language modeling
            inference_context: Inference context
            packed_seq_params: Packed sequence parameters
            extra_block_kwargs: Extra block keyword arguments
            runtime_gather_output: Runtime gather output flag
            inference_params: Inference parameters (alias for inference_context)
            loss_mask: Loss mask

        Returns:
            Model outputs (logits or loss)
        """
        if self.pre_process:
            # Get text embeddings
            if inputs_embeds is None:
                inputs_embeds = self.language_model.embedding(
                    input_ids=input_ids, position_ids=None
                )  # [decoder_seq_len, b, h_language]

                inputs_embeds = inputs_embeds.transpose(1, 0).contiguous()  # [b, decoder_seq_len, h_language]

            # Process and merge image features if provided
            if pixel_values is not None and pixel_values.size(0) > 0:
                # Extract image features using vision tower
                if hasattr(self, '_extract_image_features'):
                    pixel_values = pixel_values.to(self.vision_tower.dtype if hasattr(self.vision_tower, 'dtype') else inputs_embeds.dtype)
                    image_features = self._extract_image_features(pixel_values, image_grid_hws)

                    # Merge image features with text embeddings
                    if hasattr(self, '_merge_with_image_features'):
                        inputs_embeds = inputs_embeds.to(image_features.dtype if isinstance(image_features, torch.Tensor) else inputs_embeds.dtype).clone()
                        inputs_embeds = self._merge_with_image_features(inputs_embeds, input_ids, image_features)
            else:
                # Handle case with no images - add dummy image features for stability
                # This matches the ms-swift implementation to ensure vision tower gets gradient flow
                # even for text-only batches, preventing unused parameter issues in distributed training
                if hasattr(self, 'vision_tower') and hasattr(self, '_extract_image_features'):
                    try:
                        # Create a dummy black image and process it
                        # The result is added with zero weight to maintain gradient flow
                        dummy_pixel_values = torch.zeros(
                            (1, 3, 32, 32),
                            dtype=inputs_embeds.dtype,
                            device=inputs_embeds.device
                        )
                        dummy_grid_hws = torch.tensor([[32, 32]], dtype=torch.long, device=inputs_embeds.device)

                        # Extract dummy features (vision tower forward pass)
                        dummy_pixel_values = dummy_pixel_values.to(
                            self.vision_tower.dtype if hasattr(self.vision_tower, 'dtype') else inputs_embeds.dtype
                        )
                        dummy_features = self._extract_image_features(dummy_pixel_values, dummy_grid_hws)

                        # Add dummy features with zero weight: inputs_embeds + dummy_features.mean() * 0.0
                        # This ensures vision tower parameters get gradients without affecting the output
                        if isinstance(dummy_features, torch.Tensor):
                            inputs_embeds = inputs_embeds + dummy_features.mean() * 0.0
                        elif isinstance(dummy_features, (list, tuple)) and len(dummy_features) > 0:
                            inputs_embeds = inputs_embeds + dummy_features[0].mean() * 0.0
                    except Exception:
                        # Silently skip if dummy image processing fails
                        # This can happen if vision tower isn't properly initialized yet
                        pass

            # Convert back to [decoder_seq_len, b, h_language]
            inputs_embeds = inputs_embeds.transpose(1, 0)

            # Apply sequence parallelism if enabled
            if self.config.sequence_parallel:
                inputs_embeds = scatter_to_sequence_parallel_region(inputs_embeds)

        # Forward through language model
        outputs = self.language_model.forward(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=inputs_embeds if self.pre_process else None,
            labels=labels,
            loss_mask=loss_mask,
            runtime_gather_output=runtime_gather_output,
        )
        return outputs

    def freeze(self, freeze_language_model: bool, freeze_vision_model: bool, freeze_vision_projection: bool):
        """Freeze model modules.

        Make specific modules non-trainable by setting requires_grad to False.

        Args:
            freeze_language_model (bool): Freeze the language model module.
            freeze_vision_model (bool): Freeze the vision model module.
            freeze_vision_projection (bool): Freeze the vision projection module (multi_modal_projector).
        """
        modules = []

        if freeze_language_model and hasattr(self, "language_model") and self.language_model is not None:
            modules.append(self.language_model)

        if freeze_vision_model and hasattr(self, "vision_tower") and self.vision_tower is not None:
            modules.append(self.vision_tower)

        if freeze_vision_projection and hasattr(self, "multi_modal_projector") and self.multi_modal_projector is not None:
            modules.append(self.multi_modal_projector)

        for module in modules:
            for param in module.parameters():
                param.requires_grad = False
