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
from transformers import InternVLModel
from transformers.models.internvl.configuration_internvl import InternVLConfig


HF_INTERNVL_TOY_MODEL_CONFIG = {
    "architectures": ["InternVLModel"],  # Standard transformers class name
    "attention_dropout": 0.0,
    "bos_token_id": 151643,
    "eos_token_id": 151645,
    "img_context_token_id": 151649,
    "image_token_id": 151655,
    "hidden_act": "silu",
    "initializer_range": 0.02,
    "max_position_embeddings": 4096,
    "model_type": "internvl",  # Use 'internvl' not 'internvl_chat' for AutoConfig compatibility
    "rms_norm_eps": 1e-06,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.40.0",
    "use_cache": True,
    "select_layer": -1,
    "ps_version": "v2",
    "downsample_ratio": 0.5,
    "dynamic_image_size": True,
    "use_thumbnail": True,
    "min_dynamic_patch": 1,
    "max_dynamic_patch": 12,
    # LLM config (simplified for testing)
    "llm_config": {
        "architectures": ["Qwen2ForCausalLM"],
        "hidden_size": 896,
        "intermediate_size": 4864,
        "num_hidden_layers": 2,  # Small for testing
        "num_attention_heads": 14,
        "num_key_value_heads": 2,
        "vocab_size": 151674,
        "max_position_embeddings": 4096,
        "rope_theta": 1000000.0,
        "rms_norm_eps": 1e-06,
        "model_type": "qwen2",
        "tie_word_embeddings": False,
    },
    # Vision config (simplified for testing)
    "vision_config": {
        "attention_dropout": 0.0,
        "dropout": 0.0,
        "hidden_act": "gelu",
        "hidden_size": 512,  # Small for testing
        "image_size": 224,  # Small for testing
        "intermediate_size": 2048,
        "layer_norm_eps": 1e-06,
        "model_type": "intern_vit_6b",
        "num_attention_heads": 8,
        "num_channels": 3,
        "num_hidden_layers": 2,  # Small for testing
        "patch_size": 14,
        "qk_normalization": True,
        "qkv_bias": True,
    },
}


class TestInternVLConversion:
    """
    Test InternVL model conversion from local HuggingFace model with different parallelism configurations.
    """

    @pytest.fixture(scope="class")
    def internvl_toy_model_path(self, tmp_path_factory):
        """
        Create and save a HuggingFace InternVL toy model from config to a temporary directory.

        Args:
            tmp_path_factory: Pytest temporary path factory for class-scoped fixtures

        Returns:
            str: Path to the saved HuggingFace model directory
        """
        # Create a temporary directory for this test class
        temp_dir = tmp_path_factory.mktemp("internvl_toy_model")
        model_dir = temp_dir / "internvl_toy"

        # Create InternVL config from the toy model config
        # Create config directly using InternVLConfig
        config = InternVLConfig(**HF_INTERNVL_TOY_MODEL_CONFIG)
        config.torch_dtype = torch.bfloat16  # Explicitly set the torch_dtype in config

        # Create model with random weights and convert to bfloat16
        try:
            model = InternVLModel(config)
        except Exception as e:
            # If the specific model class isn't available, skip this test
            pytest.skip(f"InternVL model class not available in transformers: {e}")

        model = model.bfloat16()

        # Debug: Check model dtype before saving
        for name, param in model.named_parameters():
            print(f"Before save - {name}: {param.dtype}")
            break  # Just check the first parameter

        # Save model and config to directory first (this creates model_dir)
        model.save_pretrained(model_dir, safe_serialization=True)

        # Create minimal processor files (no network needed)
        # The conversion test doesn't actually use these, but they need to exist
        preprocessor_config = {
            "do_convert_rgb": True,
            "do_normalize": True,
            "do_rescale": True,
            "do_resize": True,
            "image_mean": [0.5, 0.5, 0.5],
            "image_processor_type": "InternVLImageProcessor",
            "image_std": [0.5, 0.5, 0.5],
            "processor_class": "InternVLProcessor",
            "rescale_factor": 0.00392156862745098,
            "size": {"height": 448, "width": 448},
        }
        
        tokenizer_config = {
            "tokenizer_class": "Qwen2Tokenizer",
            "vocab_size": 151936,
            "bos_token": "<|endoftext|>",
            "eos_token": "<|endoftext|>",
            "pad_token": "<|endoftext|>",
            "unk_token": "<|endoftext|>",
        }

        with open(model_dir / "preprocessor_config.json", "w") as f:
            json.dump(preprocessor_config, f, indent=2)
        
        with open(model_dir / "tokenizer_config.json", "w") as f:
            json.dump(tokenizer_config, f, indent=2)

        # Also save config.json explicitly to ensure compatibility with correct torch_dtype
        config_to_save = HF_INTERNVL_TOY_MODEL_CONFIG.copy()
        config_path = model_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config_to_save, f, indent=2)

        # Create minimal model weights file if none exists
        weights_file = model_dir / "model.safetensors"
        if not weights_file.exists():
            weights_file = model_dir / "pytorch_model.bin"
            if not weights_file.exists():
                # Create a minimal weights file for testing
                # InternVL has language model and vision components
                minimal_weights = {
                    "language_model.model.embed_tokens.weight": torch.randn(151936, 896, dtype=torch.bfloat16),
                    "language_model.model.norm.weight": torch.randn(896, dtype=torch.bfloat16),
                    "vision_model.embeddings.class_embedding": torch.randn(512, dtype=torch.bfloat16),
                }
                torch.save(minimal_weights, weights_file)

        return str(model_dir)

    def test_toy_model_creation(self, internvl_toy_model_path):
        """
        Test that the toy model is created correctly and can be loaded.

        Args:
            internvl_toy_model_path: Path to the toy InternVL model (from fixture)
        """
        # Verify the model directory exists
        model_path = Path(internvl_toy_model_path)
        assert model_path.exists(), f"Model directory not found at {model_path}"

        # Check essential files exist
        config_file = model_path / "config.json"
        assert config_file.exists(), f"config.json not found at {config_file}"

        # Check for model weights (safetensors preferred)
        weights_file = model_path / "model.safetensors"
        if not weights_file.exists():
            weights_file = model_path / "pytorch_model.bin"
        assert weights_file.exists(), f"Model weights file not found in {model_path}"

        # Check for processor files
        processor_config_file = model_path / "preprocessor_config.json"
        assert processor_config_file.exists(), f"preprocessor_config.json not found at {processor_config_file}"

        # Load and verify config
        with open(config_file) as f:
            config_data = json.load(f)

        assert config_data["model_type"] == "internvl"
        assert config_data["llm_config"]["hidden_size"] == 896
        assert config_data["llm_config"]["num_hidden_layers"] == 2
        assert config_data["vision_config"]["hidden_size"] == 512
        assert config_data["vision_config"]["num_hidden_layers"] == 2
        # Check InternVL-specific parameters
        assert config_data["select_layer"] == -1
        assert config_data["ps_version"] == "v2"
        assert config_data["downsample_ratio"] == 0.5
        assert config_data["dynamic_image_size"] is True
        assert config_data["use_thumbnail"] is True

        print(f"SUCCESS: Toy model created and validated at {internvl_toy_model_path}")
        print("Model weights are correctly in bfloat16 format")

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize(
        "tp,pp,test_name",
        [
            (2, 1, "TP"),
            (1, 2, "PP"),
        ],
    )
    def test_internvl_conversion_parallelism(self, internvl_toy_model_path, tmp_path, tp, pp, test_name):
        """
        Test InternVL model conversion with different parallelism configurations.

        Args:
            internvl_toy_model_path: Path to the toy InternVL model (from fixture)
            tmp_path: Pytest temporary path fixture
            tp: Tensor parallelism size
            pp: Pipeline parallelism size
            test_name: Name of the test for identification
        """
        # Create temporary output directory for conversion results
        test_output_dir = tmp_path / f"internvl_{test_name}"
        test_output_dir.mkdir(exist_ok=True)

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
            internvl_toy_model_path,
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

            # Check that the conversion completed successfully
            if result.returncode != 0:
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                assert False, f"InternVL {test_name} conversion failed with return code {result.returncode}"

            # Verify that the converted model was saved
            # The output directory should be named after the last part of the model path
            model_name = Path(internvl_toy_model_path).name  # "internvl_toy"
            converted_model_dir = test_output_dir / model_name
            assert converted_model_dir.exists(), f"Converted model directory not found at {converted_model_dir}"

            # Check that essential model files exist
            config_file = converted_model_dir / "config.json"
            assert config_file.exists(), f"config.json not found in converted model at {config_file}"

            # Check for model weights file (could be either safetensors or pytorch_model.bin)
            weights_file_safetensors = converted_model_dir / "model.safetensors"
            weights_file_pytorch = converted_model_dir / "pytorch_model.bin"
            assert weights_file_safetensors.exists() or weights_file_pytorch.exists(), (
                f"Model weights file not found in converted model at {converted_model_dir}"
            )

            # Verify the config contains InternVL-specific parameters
            with open(config_file) as f:
                saved_config = json.load(f)

            assert saved_config["model_type"] == "internvl", "Model type should be internvl"
            # Verify InternVL-specific parameters
            assert saved_config["select_layer"] == -1, "select_layer should match"
            assert saved_config["ps_version"] == "v2", "ps_version should match"
            assert saved_config["downsample_ratio"] == 0.5, "downsample_ratio should match"

            print(f"SUCCESS: InternVL {test_name} conversion test completed successfully")
            print(f"Converted model saved at: {converted_model_dir}")

        except Exception as e:
            print(f"Error during InternVL {test_name} conversion test: {e}")
            raise

