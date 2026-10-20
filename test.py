# Adapted through JamMa from the LoFTR framework; modified for WovenAnchor.
# Inherited MIT/Apache-2.0 terms and attribution: THIRD_PARTY_NOTICES.md.

import argparse
import os
import pprint
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.elastic.utils.distributed import get_free_port
import pytorch_lightning as pl
from loguru import logger as loguru_logger

from src.config.default import get_cfg_defaults
from src.utils.misc import setup_gpus
from src.lightning.data import MultiSceneDataModule
from src.lightning.lightning_wam import WovenAnchorLightningModule
from src.utils.profiler import build_profiler


def setup_dummy_distributed(max_retries=64):
    """Initialize a single-process group using an available local port."""
    if dist.is_available() and not dist.is_initialized():
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        
        for _ in range(max_retries):
            os.environ["MASTER_PORT"] = str(get_free_port())
            try:
                dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
                return
            except RuntimeError as e:
                # Retry if another process claimed the port before initialization.
                if "EADDRINUSE" in str(e) or "Address already in use" in str(e):
                    continue
                raise
                
        raise RuntimeError(f"Failed to find a free port and initialize dummy process group after {max_retries} tries.")


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("data_cfg_path", type=str, help="data config path")
    parser.add_argument("main_cfg_path", type=str, help="main config path")
    parser.add_argument("--ablation_cfg", type=str, default=None, help="optional ablation config path")
    parser.add_argument("--ckpt_path", type=str, default="weights/outdoor.ckpt", help="path to the checkpoint")
    
    parser.add_argument("--gpus", default=1, help="number of gpus per node")
    parser.add_argument("--num_nodes", type=int, default=1)

    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--parallel_load_data", action="store_true", help="build scene datasets in parallel using num_workers threads")
    parser.add_argument("--batch_size", type=int, default=1, help='batch_size per gpu')
    parser.add_argument("--benchmark", action="store_true", default=True)
    parser.add_argument("--profiler_name", type=str, default="inference", choices=["inference", "pytorch"])
    parser.add_argument("--deterministic", action="store_true", default=False)

    parser.add_argument("--vis", type=lambda s: None if s.lower() == "none" else s, choices=[None, "qualitative", "woven", "anchor"], help="visualization mode")
    parser.add_argument("--vis_name", type=str, default=None, help="label for visualization filenames")
    parser.add_argument("--vis_max_pairs", type=int, default=10, help="pairs to visualize (-1: all, 0: off)")
    parser.add_argument("--stats_anchor", action="store_true", help="print anchor statistics over all pairs")

    args = parser.parse_args()
    if args.vis_max_pairs < -1:
        parser.error("--vis_max_pairs must be -1 or non-negative")
    return args

def main():
    args = parse_args()
    pprint.pprint(vars(args))

    config = get_cfg_defaults()
    config.merge_from_file(args.main_cfg_path)
    config.merge_from_file(args.data_cfg_path)
    if args.ablation_cfg is not None:
        config.merge_from_file(args.ablation_cfg)
    
    torch.backends.cudnn.deterministic = args.deterministic
    pl.seed_everything(config.TRAINER.SEED)

    args.gpus = setup_gpus(args.gpus)
    config.TRAINER.WORLD_SIZE = args.gpus * args.num_nodes
    config.TRAINER.TRUE_BATCH_SIZE = config.TRAINER.WORLD_SIZE * args.batch_size

    loguru_logger.info("Args and config initialized!")

    if not args.ckpt_path:
        loguru_logger.error(
            "Checkpoint path is empty. Pass --ckpt_path <path>."
        )
        sys.exit(1)
    if not os.path.exists(args.ckpt_path):
        loguru_logger.error(f"Checkpoint path does not exist: {args.ckpt_path}")
        sys.exit(1)
    print(f"Checkpoint: {args.ckpt_path}")

    profiler = build_profiler(args.profiler_name)
    model = WovenAnchorLightningModule(config, pretrained_ckpt=args.ckpt_path, profiler=profiler)
    loguru_logger.info("WovenAnchorMatcher-lightning initialized!")

    data_module = MultiSceneDataModule(args, config)
    loguru_logger.info("DataModule initialized!")

    # Initialize distributed state after checkpoint loading and model construction.
    setup_dummy_distributed()

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.gpus,
        benchmark=args.benchmark,
        logger=False,
    )

    loguru_logger.info("Trainer initialized!")

    vis_name = args.vis_name or (Path(args.ablation_cfg).stem if args.ablation_cfg else None)
    model._set_vis(args.vis, vis_name=vis_name, max_pairs=args.vis_max_pairs)
    model._set_anchor_stats(args.stats_anchor)

    loguru_logger.info("Start testing!")
    pl.seed_everything(config.TRAINER.SEED)
    trainer.test(model, datamodule=data_module, verbose=False)

if __name__ == "__main__":
    main()
