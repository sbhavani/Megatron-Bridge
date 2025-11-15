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

import json
import subprocess
from pathlib import Path

import pytest
import torch


# Minimal configuration for a Kimi VL toy model
# Based on DeepSeek V3 architecture with vision capabilities
HF_KIMI_VL_TOY_MODEL_CONFIG = {
    "architectures": ["DeepseekV3ForCausalLM"],
    "attention_dropout": 0.0,
    "bos_token_id": 0,
    "eos_token_id": 1,
    "vision_start_token_id": 151652,
    "vision_end_token_id": 151653,
    "image_token_id": 151655,
    "hidden_act": "silu",
    "hidden_size": 2048,
    "initializer_range": 0.02,
    "intermediate_size": 5504,
    "max_position_embeddings": 4096,
    "model_type": "deepseek_v3",
    "num_attention_heads": 16,
    "num_hidden_layers": 2,
    "num_key_value_heads": 16,
    "rms_norm_eps": 1e-06,
    "rope_theta": 1000000.0,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "use_cache": True,
    "vocab_size": 151936,
    # Vision config
    "vision_config": {
        "hidden_size": 1024,
        "intermediate_size": 2048,
        "num_hidden_layers": 12,
        "num_attention_heads": 16,
        "image_size": 224,
        "patch_size": 14,
    },
}


class TestKimiVLConversion:
    """
    Test Kimi VL model conversion from local HuggingFace model with different parallelism configurations.
    """

    @pytest.fixture(scope="class")
    def kimi_vl_toy_model_path(self, tmp_path_factory):
        """
        Create and save a minimal Kimi VL toy model configuration to a temporary directory.

        Since Kimi VL uses a dynamic module, we'll create a minimal config that can be
        used for testing the bridge and provider logic.

        Args:
            tmp_path_factory: Pytest temporary path factory for class-scoped fixtures

        Returns:
            str: Path to the saved model directory
        """
        # Create a temporary directory for this test class
        temp_dir = tmp_path_factory.mktemp("kimi_vl_toy_model")
        model_dir = temp_dir / "kimi_vl_toy"
        model_dir.mkdir()

        # Save config.json
        config_path = model_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(HF_KIMI_VL_TOY_MODEL_CONFIG, f, indent=2)

        # Create minimal tokenizer files
        tokenizer_config = {
            "tokenizer_class": "PreTrainedTokenizer",
            "vocab_size": 151936,
            "bos_token": "<|begin_of_text|>",
            "eos_token": "<|end_of_text|>",
            "pad_token": "<|end_of_text|>",
        }

        with open(model_dir / "tokenizer_config.json", "w") as f:
            json.dump(tokenizer_config, f, indent=2)

        # Note: For actual conversion testing, a full model with weights would be needed
        # This is just for testing the configuration bridge logic
        return str(model_dir)

    def test_toy_model_creation(self, kimi_vl_toy_model_path):
        """
        Test that the toy model configuration is created correctly.

        Args:
            kimi_vl_toy_model_path: Path to the toy Kimi VL model (from fixture)
        """
        # Verify the model directory exists
        model_path = Path(kimi_vl_toy_model_path)
        assert model_path.exists(), f"Model directory not found at {model_path}"

        # Check essential files exist
        config_file = model_path / "config.json"
        assert config_file.exists(), f"config.json not found at {config_file}"

        # Check for tokenizer files
        tokenizer_config_file = model_path / "tokenizer_config.json"
        assert tokenizer_config_file.exists(), f"tokenizer_config.json not found at {tokenizer_config_file}"

        # Load and verify config
        with open(config_file) as f:
            config_data = json.load(f)

        assert config_data["model_type"] == "deepseek_v3"
        assert config_data["hidden_size"] == 2048
        assert config_data["num_hidden_layers"] == 2
        assert config_data["num_attention_heads"] == 16
        assert config_data["vocab_size"] == 151936
        assert "vision_config" in config_data

        print(f"SUCCESS: Toy model configuration created and validated at {kimi_vl_toy_model_path}")

    @pytest.mark.skip(reason="Requires full Kimi VL model with weights and dynamic module support")
    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize(
        "tp,pp,test_name",
        [
            (2, 1, "TP"),
            (1, 2, "PP"),
        ],
    )
    def test_kimi_vl_conversion_parallelism(self, kimi_vl_toy_model_path, tmp_path, tp, pp, test_name):
        """
        Test Kimi VL model conversion with different parallelism configurations.

        This test is skipped by default as it requires a full model with weights
        and HuggingFace's dynamic module support for Kimi VL.

        Args:
            kimi_vl_toy_model_path: Path to the toy Kimi VL model (from fixture)
            tmp_path: Pytest temporary path fixture
            tp: Tensor parallelism size
            pp: Pipeline parallelism size
            test_name: Name of the test for identification
        """
        # Create temporary output directory for conversion results
        test_output_dir = tmp_path / f"kimi_vl_{test_name}"
        test_output_dir.mkdir(exist_ok=True)

        # Run hf_megatron_roundtrip_multi_gpu.py with specified parallelism configuration
        cmd = [
            "python",
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=2",
            "--nnodes=1",
            "-m",
            "coverage",
            "run",
            "--data-file=/opt/Megatron-Bridge/.coverage",
            "--source=/opt/Megatron-Bridge/",
            "--parallel-mode",
            "examples/conversion/hf_megatron_roundtrip_multi_gpu.py",
            "--hf-model-id",
            kimi_vl_toy_model_path,
            "--output-dir",
            str(test_output_dir),
            "--tp",
            str(tp),
            "--pp",
            str(pp),
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent.parent.parent
            )
            print(cmd)

            # Check that the conversion completed successfully
            if result.returncode != 0:
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                assert False, f"Kimi VL {test_name} conversion failed with return code {result.returncode}"

            # Verify that the converted model was saved
            model_name = Path(kimi_vl_toy_model_path).name
            converted_model_dir = test_output_dir / model_name
            assert converted_model_dir.exists(), f"Converted model directory not found at {converted_model_dir}"

            # Check that essential model files exist
            config_file = converted_model_dir / "config.json"
            assert config_file.exists(), f"config.json not found in converted model at {config_file}"

            # Verify the config contains Kimi VL-specific parameters
            with open(config_file) as f:
                saved_config = json.load(f)

            assert saved_config["model_type"] == "deepseek_v3", "Model type should be deepseek_v3"
            assert saved_config["hidden_size"] == 2048, "Hidden size should match toy config"
            assert "vision_config" in saved_config, "VL model should have vision_config"

            print(f"SUCCESS: Kimi VL {test_name} conversion test completed successfully")
            print(f"Converted model saved at: {converted_model_dir}")

        except Exception as e:
            print(f"Error during Kimi VL {test_name} conversion test: {e}")
            raise
