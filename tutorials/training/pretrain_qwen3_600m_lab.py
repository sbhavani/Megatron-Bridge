from megatron.bridge.recipes.qwen import qwen3_600m_pretrain_config

from _pretrain_qwen3_lab_common import run_pretrain


if __name__ == "__main__":
    run_pretrain(
        config_factory=qwen3_600m_pretrain_config,
        default_results_dir="/workspace/megatron-results/qwen3_600m",
    )
