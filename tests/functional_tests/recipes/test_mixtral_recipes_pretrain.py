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

"""Functional smoke tests for Mixtral recipe configurations."""

import pytest

from megatron.bridge.recipes.mixtral import (
    mixtral_8x7b_pretrain_config,
    mixtral_8x22b_pretrain_config,
)
from tests.functional_tests.recipes.utils import run_pretrain_recipe_test


MIXTRAL_PRETRAIN_RECIPES = [
    # (config_func, name, parallelism_overrides, model_overrides)
    (
        mixtral_8x7b_pretrain_config,
        "mixtral_8x7b",
        {"expert_model_parallel_size": 2},  # Reduce EP for test
        {"num_layers": 2},  # Use minimal layers for testing
    ),
    (
        mixtral_8x22b_pretrain_config,
        "mixtral_8x22b",
        {
            "tensor_model_parallel_size": 1,  # Reduce from 4 to 1 for 2 GPUs
            "pipeline_model_parallel_size": 1,  # Reduce from 2 to 1 for 2 GPUs
            "expert_model_parallel_size": 2,  # Keep EP=2 to test MoE
            "sequence_parallel": False,  # Disable SP when TP=1
        },
        {"num_layers": 2},  # Use minimal layers for testing
    ),
]


class TestMixtralRecipes:
    """Test class for Mixtral recipe functional tests."""

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize("config_func,recipe_name,parallelism_overrides,model_overrides", MIXTRAL_PRETRAIN_RECIPES)
    def test_mixtral_pretrain_recipes(self, config_func, recipe_name, parallelism_overrides, model_overrides, tmp_path):
        """Functional test for Mixtral recipes with appropriate parallelism configurations."""
        run_pretrain_recipe_test(
            config_func,
            recipe_name,
            tmp_path,
            model_overrides=model_overrides,
            **parallelism_overrides,
        )

