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
import torch.nn as nn
from megatron.core.tensor_parallel import scatter_to_sequence_parallel_region
from megatron.core.transformer.module import MegatronModule
from torch import Tensor
from transformers.models.internvl.modeling_internvl import (
    InternVLModel as HFInternVLModel,
    InternVLVisionModel,
)

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.utils.common_utils import hook_hf_module_setattr_for_tp_grad_sync


class InternVLModel(MegatronModule):
    """
    InternVL Vision-Language (VL) model wrapper for Megatron.

    InternVL combines a vision encoder (InternViT), an MLP projector, and a language model
    to enable multimodal understanding and generation.

    Args:
        config (GPTModelProvider): Model provider containing configuration for language and vision modules.
        pre_process (bool, optional): Whether to construct the vision tower and projector. Default: True.
        post_process (bool, optional): Whether to apply post-processing. Default: True.
        vp_stage (Optional[int], optional): Pipeline stage for model parallelism. Default: None.

    Attributes:
        pre_process (bool): If True, enables vision and multimodal components.
        post_process (bool): If True, enables post-processing.
        vp_stage (Optional[int]): Pipeline stage for model parallelism.
        vision_model (nn.Module): Vision encoder (InternViT).
        mlp1 (nn.Module): First MLP projection layer.
        language_model (nn.Module): The underlying language model.
        get_image_features (callable): Method to extract image features.

    Forward Inputs:
        input_ids (torch.LongTensor, optional): Tokenized input ids for the language model.
        attention_mask (torch.Tensor, optional): Attention mask for the language model.
        position_ids (torch.LongTensor, optional): Position ids for the language model.
        inputs_embeds (torch.FloatTensor, optional): Precomputed input embeddings.
        pixel_values (torch.Tensor, optional): Image tensor(s) for the vision tower.
        labels (torch.Tensor, optional): Target labels for supervised training.
        runtime_gather_output (bool, optional): If True, gather outputs across pipeline stages.
        loss_mask (Tensor, optional): Mask for loss computation.

    Returns:
        Tensor: Model output (e.g., logits or loss, depending on mode).

    Note:
        - If `pre_process` is False, only the language model is constructed.
        - The vision tower and projector are only active if `pre_process` is True.
        - This class is intended for use within the Megatron-LM framework.
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

        if pre_process:
            # Initialize vision model
            self.vision_model = InternVLVisionModel._from_config(config.vision_config)

            # Initialize MLP projector
            # InternVL uses a simple linear projection from vision to language space
            self.mlp1 = nn.Sequential(
                nn.LayerNorm(config.vision_config.hidden_size * int(1 / config.downsample_ratio) ** 2),
                nn.Linear(
                    config.vision_config.hidden_size * int(1 / config.downsample_ratio) ** 2,
                    config.hidden_size,
                ),
                nn.GELU(),
                nn.Linear(config.hidden_size, config.hidden_size),
            )

            # Ensure HF visual tower params are marked for TP grad sync and future assignments are hooked.
            hook_hf_module_setattr_for_tp_grad_sync(self.vision_model)

        self.language_model = self.config.provide_language_model(
            pre_process=pre_process, post_process=post_process, vp_stage=vp_stage
        )

        # Finalize grad requires these to be bound with module
        self.share_embeddings_and_output_weights = config.share_embeddings_and_output_weights
        self.shared_embedding_or_output_weight = self.language_model.shared_embedding_or_output_weight

        # Bind methods from HF's InternVLModel to this instance
        self.get_image_features = types.MethodType(HFInternVLModel.get_image_features, self)

    def set_input_tensor(self, input_tensor) -> None:
        """Set model chunk input tensor."""
        self.language_model.set_input_tensor(input_tensor)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        runtime_gather_output: Optional[bool] = None,
        *,
        loss_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass for InternVL model.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask (not used, included for compatibility)
            position_ids: Position IDs for the language model
            inputs_embeds: Pre-computed input embeddings (optional)
            pixel_values: Image pixel values for the vision encoder
            labels: Target labels for training
            runtime_gather_output: Whether to gather outputs across pipeline stages
            loss_mask: Mask for loss computation

        Returns:
            Model outputs (logits or loss)
        """
        if self.pre_process:
            if inputs_embeds is None:
                inputs_embeds = self.language_model.embedding(
                    input_ids=input_ids, position_ids=None
                )  # [decoder_seq_len, b, h_language]

                inputs_embeds = inputs_embeds.transpose(1, 0).contiguous()  # [b, decoder_seq_len, h_language]

            if pixel_values is not None:
                # Extract image features using the vision model
                image_features = self.get_image_features(pixel_values)

                # Check that we have the right number of image tokens
                assert input_ids is not None
                special_image_mask = (
                    (input_ids == self.config.img_context_token_id) |
                    (input_ids == self.config.image_token_id)
                ).unsqueeze(-1)
                special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)

                # Verify the number of image tokens matches the image features
                if inputs_embeds[special_image_mask].numel() != image_features.numel():
                    image_tokens_in_text = special_image_mask.sum()
                    raise ValueError(
                        f"Number of images does not match number of special image tokens in the input text. "
                        f"Got {image_tokens_in_text} image tokens in the text but {image_features.numel()} "
                        "elements from image embeddings."
                    )

                # Cast image features to the same dtype and device as input embeddings
                image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)

                # Insert image features into the input embeddings
                inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

            inputs_embeds = inputs_embeds.transpose(1, 0).contiguous()  # (B, T, D) -> (T, B, D)

            if self.config.sequence_parallel:
                inputs_embeds = scatter_to_sequence_parallel_region(inputs_embeds)

        # Compute causal attention mask
        attention_mask = self._compute_attention_mask(input_ids)

        outputs = self.language_model.forward(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,  # (B, 1, T, T)
            decoder_input=inputs_embeds,  # (T, B, D)
            labels=labels,  # (B, T)
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
            freeze_vision_projection (bool): Freeze the vision projection module (mlp1).
        """
        modules = []

        if freeze_language_model and hasattr(self, "language_model") and self.language_model is not None:
            modules.append(self.language_model)

        if freeze_vision_model and hasattr(self, "vision_model") and self.vision_model is not None:
            modules.append(self.vision_model)

        if freeze_vision_projection and hasattr(self, "mlp1") and self.mlp1 is not None:
            modules.append(self.mlp1)

        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

    def _compute_attention_mask(
        self,
        input_ids: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Compute causal attention mask."""
        if not self.pre_process:
            return None

        if input_ids is None:
            return None

        batch_size, seq_len = input_ids.shape
        # Create causal mask (lower triangular)
        causal_mask = torch.tril(torch.ones((batch_size, 1, seq_len, seq_len))).to(input_ids.device)

        # For InternVL, we just use a standard causal mask
        # The image tokens are treated like regular tokens in the sequence
        return causal_mask
