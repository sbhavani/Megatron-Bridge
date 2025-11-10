#!/usr/bin/env python3
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

"""
Mixtral 8x7B Finetuning Script with YAML and CLI Configuration Overrides.

This script provides a flexible way to finetune Mixtral 8x7B models using Megatron-Bridge with support for
both YAML configuration files and command-line overrides using Hydra-style syntax.

Examples:
    Loading pretrained weights (recommended for finetune):
        1) Import HF checkpoint to Megatron format:
           $ python examples/conversion/convert_checkpoints.py import \\
               --hf-model mistralai/Mixtral-8x7B-Instruct-v0.1 \\
               --megatron-path /path/to/megatron_ckpt

        2) Run finetune using the imported checkpoint:
           $ torchrun --nproc_per_node=8 examples/recipes/mixtral/finetune_mixtral_8x7b.py \\
               --pretrained-checkpoint /path/to/megatron_ckpt

    Using a custom YAML config file:
        $ torchrun --nproc_per_node=8 finetune_mixtral_8x7b.py --config-file conf/mixtral_8x7b_finetune.yaml

    CLI overrides:
        $ torchrun --nproc_per_node=8 finetune_mixtral_8x7b.py model.tensor_model_parallel_size=4 train.train_iters=1000

    Using mock data for testing:
        $ torchrun --nproc_per_node=8 finetune_mixtral_8x7b.py --mock-data

    Selecting a different model variant:
        $ torchrun --nproc_per_node=8 finetune_mixtral_8x7b.py --recipe mixtral_8x22b_finetune_config

Configuration Precedence:
    1. Base configuration from mixtral_8x7b_finetune_config() recipe
    2. YAML overrides from --config-file (if provided)
    3. CLI overrides (highest precedence)
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Tuple

import torch
from omegaconf import OmegaConf

from megatron.bridge.recipes.mixtral import mixtral as mixtral_recipes
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.training.utils.omegaconf_utils import (
    apply_overrides,
    create_omegaconf_dict_config,
    parse_hydra_overrides,
)
from megatron.bridge.utils.common_utils import get_rank_safe


logger: logging.Logger = logging.getLogger(__name__)


SCRIPT_DIR: Path = Path(__file__).parent.resolve()
DEFAULT_CONFIG_FILENAME: str = "mixtral_8x7b_finetune_override_example.yaml"
DEFAULT_CONFIG_FILE_PATH: Path = SCRIPT_DIR / "conf" / DEFAULT_CONFIG_FILENAME


def parse_cli_args() -> Tuple[argparse.Namespace, list[str]]:
    """Parse known script args and return remaining as Hydra-style overrides."""
    parser = argparse.ArgumentParser(
        description="Finetune Mixtral 8x7B with YAML and CLI overrides",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default=str(DEFAULT_CONFIG_FILE_PATH) if DEFAULT_CONFIG_FILE_PATH.exists() else None,
        help="Path to the YAML OmegaConf override file. Default: conf/mixtral_8x7b_finetune_override_example.yaml",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to training dataset (e.g., JSONL file with instruction-following examples).",
    )
    parser.add_argument(
        "--recipe",
        type=str,
        default="mixtral_8x7b_finetune_config",
        help=(
            "Name of the recipe function in megatron.bridge.recipes.mixtral.mixtral to use, "
            "e.g., mixtral_8x7b_finetune_config, mixtral_8x22b_finetune_config."
        ),
    )
    parser.add_argument(
        "--pretrained-checkpoint",
        type=str,
        default=None,
        help=(
            "Path to imported Megatron checkpoint directory to load before finetuning. "
            "Generate it with examples/conversion/convert_checkpoints.py."
        ),
    )
    parser.add_argument(
        "--mock-data",
        action="store_true",
        help="Use mock data for testing (overrides --data-path)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args, cli_dotlist_overrides = parser.parse_known_args()
    return args, cli_dotlist_overrides


def main() -> None:
    """
    Entry point for the Mixtral 8x7B finetuning script.

    This function orchestrates the complete configuration workflow:
    1. Loads the base configuration from the selected recipe
    2. Applies YAML overrides from --config-file (if exists)
    3. Applies CLI overrides using Hydra-style syntax
    4. Starts Megatron training with the final merged configuration

    Examples of CLI usage:
        # Use default 8x7B config with custom learning rate
        torchrun --nproc_per_node=8 finetune_mixtral_8x7b.py optimizer.lr=1e-5

        # Use 8x22B config instead
        torchrun --nproc_per_node=16 finetune_mixtral_8x7b.py --recipe mixtral_8x22b_finetune_config

        # Multiple MoE-specific overrides
        torchrun --nproc_per_node=8 finetune_mixtral_8x7b.py \\
            model.tensor_model_parallel_size=2 \\
            model.expert_model_parallel_size=4 \\
            train.train_iters=1000
    """
    args, cli_overrides = parse_cli_args()

    logger.info("Megatron-Bridge Mixtral 8x7B Finetuning Script with YAML & CLI Overrides")
    logger.info("-------------------------------------------------------------------------")

    # Resolve the recipe function from the provided name
    recipe_name = getattr(args, "recipe", "mixtral_8x7b_finetune_config")
    available_recipes = [name for name in dir(mixtral_recipes) if name.endswith("_finetune_config")]
    if not hasattr(mixtral_recipes, recipe_name):
        logger.error(
            "Unknown recipe '%s'. Available recipes: %s",
            recipe_name,
            ", ".join(sorted(available_recipes)),
        )
        sys.exit(2)
    finetune_config_func = getattr(mixtral_recipes, recipe_name)

    # Build recipe kwargs based on CLI arguments
    recipe_kwargs = {
        "mock": args.mock_data,
    }

    # Add data paths if provided
    if args.data_path and not args.mock_data:
        recipe_kwargs["data_paths"] = [args.data_path]

    # Load configuration from the selected recipe
    cfg: ConfigContainer = finetune_config_func(**recipe_kwargs)

    # Override checkpoint load path if pretrained checkpoint is provided
    if args.pretrained_checkpoint:
        cfg.checkpoint.load = args.pretrained_checkpoint
        logger.info(f"Will load pretrained checkpoint from: {args.pretrained_checkpoint}")

    logger.info("Loaded base configuration")

    if get_rank_safe() == 0:
        cfg.print_yaml()

    merged_omega_conf, excluded_fields = create_omegaconf_dict_config(cfg)

    if args.config_file:
        logger.debug(f"Loading YAML overrides from: {args.config_file}")
        if not os.path.exists(args.config_file):
            logger.error(f"Override YAML file not found: {args.config_file}")
            sys.exit(1)
        yaml_overrides_omega = OmegaConf.load(args.config_file)
        merged_omega_conf = OmegaConf.merge(merged_omega_conf, yaml_overrides_omega)

    if cli_overrides:
        logger.debug(f"Applying Hydra-style command-line overrides: {cli_overrides}")
        merged_omega_conf = parse_hydra_overrides(merged_omega_conf, cli_overrides)

    final_overrides_as_dict = OmegaConf.to_container(merged_omega_conf, resolve=True)
    apply_overrides(cfg, final_overrides_as_dict, excluded_fields)

    if get_rank_safe() == 0:
        logger.info("--- Final Merged Configuration ---")
        cfg.print_yaml()
        logger.info("----------------------------------")

    # Start training (uses same pretrain function - it handles both pretrain and finetune)
    logger.debug("Starting finetuning...")
    pretrain(config=cfg, forward_step_func=forward_step)

    # Cleanup process group
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
