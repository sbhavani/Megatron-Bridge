import argparse

from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain


def build_parser(default_results_dir: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=default_results_dir)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--virtual-pipeline-parallel-size", type=int, default=None)
    parser.add_argument("--context-parallel-size", type=int, default=1)
    parser.add_argument("--sequence-parallel", action="store_true")
    parser.add_argument("--use-megatron-fsdp", action="store_true")
    parser.add_argument("--train-iters", type=int, default=10)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--global-batch-size", type=int, default=4)
    parser.add_argument("--seq-length", type=int, default=2048)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--recompute-granularity", choices=["none", "selective", "full"], default="none")
    parser.add_argument("--recompute-method", choices=["uniform", "block"], default=None)
    parser.add_argument("--recompute-num-layers", type=int, default=None)
    parser.add_argument("--recompute-modules", nargs="*", default=None)
    parser.add_argument("--use-distributed-optimizer", action="store_true")
    parser.add_argument("--tp-comm-overlap", action="store_true")
    parser.add_argument("--overlap-grad-reduce", action="store_true")
    parser.add_argument("--overlap-param-gather", action="store_true")
    parser.add_argument("--profile", choices=["none", "pytorch"], default="none")
    parser.add_argument("--profile-step-start", type=int, default=9)
    parser.add_argument("--profile-step-end", type=int, default=10)
    parser.add_argument("--profile-ranks", type=int, nargs="*", default=[0])
    parser.add_argument("--save", default=None)
    parser.add_argument("--save-interval", type=int, default=None)
    parser.add_argument("--load", default=None)
    parser.add_argument("--async-save", action="store_true")
    return parser


def run_pretrain(config_factory, default_results_dir: str) -> None:
    args = build_parser(default_results_dir).parse_args()
    cfg = config_factory()

    cfg.train.train_iters = args.train_iters
    cfg.train.micro_batch_size = args.micro_batch_size
    cfg.train.global_batch_size = args.global_batch_size
    cfg.scheduler.lr_warmup_iters = 0
    cfg.scheduler.lr_warmup_steps = 0
    cfg.logger.log_interval = args.log_interval

    cfg.model.tensor_model_parallel_size = args.tensor_parallel_size
    cfg.model.pipeline_model_parallel_size = args.pipeline_parallel_size
    cfg.model.virtual_pipeline_model_parallel_size = args.virtual_pipeline_parallel_size
    cfg.model.context_parallel_size = args.context_parallel_size
    cfg.model.sequence_parallel = args.sequence_parallel
    cfg.model.seq_length = args.seq_length
    cfg.dataset.seq_length = args.seq_length

    cfg.ddp.use_megatron_fsdp = args.use_megatron_fsdp
    cfg.ddp.use_distributed_optimizer = args.use_distributed_optimizer
    cfg.ddp.overlap_grad_reduce = args.overlap_grad_reduce
    cfg.ddp.overlap_param_gather = args.overlap_param_gather

    if args.tp_comm_overlap:
        cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)

    if args.recompute_granularity != "none":
        cfg.model.recompute_granularity = args.recompute_granularity
    if args.recompute_method is not None:
        cfg.model.recompute_method = args.recompute_method
    if args.recompute_num_layers is not None:
        cfg.model.recompute_num_layers = args.recompute_num_layers
    if args.recompute_modules is not None:
        cfg.model.recompute_modules = args.recompute_modules

    if args.profile == "pytorch":
        cfg.profiling.use_pytorch_profiler = True
        cfg.profiling.profile_step_start = args.profile_step_start
        cfg.profiling.profile_step_end = args.profile_step_end
        cfg.profiling.profile_ranks = args.profile_ranks

    cfg.checkpoint.save = None
    cfg.checkpoint.load = None
    if args.save is not None:
        cfg.checkpoint.save = args.save
    if args.load is not None:
        cfg.checkpoint.load = args.load
    if args.save_interval is not None:
        cfg.checkpoint.save_interval = args.save_interval
    if args.async_save:
        cfg.checkpoint.async_save = True

    pretrain(cfg, forward_step)
