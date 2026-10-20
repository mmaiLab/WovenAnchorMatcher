# Adapted through JamMa from the LoFTR framework; modified for WovenAnchor.
# Inherited MIT/Apache-2.0 terms and attribution: THIRD_PARTY_NOTICES.md.

import math
import argparse
import pprint
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, ModelSummary
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from src.config.default import get_cfg_defaults
from src.utils.misc import setup_gpus
from src.utils.profiler import build_profiler
from src.lightning.data import MultiSceneDataModule
from src.lightning.lightning_wam import WovenAnchorLightningModule


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("data_cfg_path", type=str, help="data config path")
    parser.add_argument("main_cfg_path", type=str, help="main config path")
    parser.add_argument("--ablation_cfg", type=str, default=None, help="optional ablation config path")
    parser.add_argument("--exp_name", type=str, default=None, help="experiment name")

    parser.add_argument("--pretrained_ckpt_path", type=str, default=None)
    parser.add_argument("--gpus", default=1, help="number of gpus per node")
    parser.add_argument("--num_nodes", type=int, default=1)

    parser.add_argument("--batch_size", type=int, default=4, help="batch_size per gpu")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--parallel_load_data", action="store_true", help="build scene datasets in parallel using num_workers threads")
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--log_every_n_steps", type=int, default=10)
    parser.add_argument("--benchmark", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1)
    parser.add_argument("--limit_val_batches", type=float, default=1.0)
    parser.add_argument("--num_sanity_val_steps", type=int, default=10)
    parser.add_argument("--profiler_name", choices=["inference", "pytorch"], default=None)
    parser.add_argument("--logger", type=str, default="tensorboard", choices=["tensorboard", "wandb", "none"])
    parser.add_argument("--lr_scale", type=float, default=1.0, help="initial learning rate scale")
    parser.add_argument("--resume_ckpt_path", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()
    if args.exp_name is None:
        args.exp_name = Path(args.ablation_cfg).stem if args.ablation_cfg is not None else "wam"
    rank_zero_only(pprint.pprint)(vars(args))

    config = get_cfg_defaults()
    config.merge_from_file(args.main_cfg_path)
    config.merge_from_file(args.data_cfg_path)
    if args.ablation_cfg is not None:
        config.merge_from_file(args.ablation_cfg)

    pl.seed_everything(config.TRAINER.SEED)

    args.gpus = setup_gpus(args.gpus)
    config.TRAINER.WORLD_SIZE = args.gpus * args.num_nodes
    config.TRAINER.TRUE_BATCH_SIZE = config.TRAINER.WORLD_SIZE * args.batch_size
    scaling = config.TRAINER.TRUE_BATCH_SIZE / config.TRAINER.CANONICAL_BS * args.lr_scale
    config.TRAINER.TRUE_LR = config.TRAINER.CANONICAL_LR * scaling
    config.TRAINER.WARMUP_STEP = math.floor(config.TRAINER.WARMUP_STEP / scaling)
    config.TRAINER.COSA_TMAX = args.max_epochs

    if args.logger == "wandb":
        logger = WandbLogger(
            save_dir="logs",
            project="WAM",
            name=args.exp_name,
            log_model=False,
        )
    elif args.logger == "tensorboard":
        logger = TensorBoardLogger(
            save_dir="logs/tb_logs",
            name=args.exp_name,
            default_hp_metric=False,
        )
    else:
        logger = False

    profiler = build_profiler(args.profiler_name)
    model = WovenAnchorLightningModule(config, pretrained_ckpt=args.pretrained_ckpt_path, profiler=profiler)

    data_module = MultiSceneDataModule(args, config)

    callbacks = [
        ModelSummary(max_depth=1),
        ModelCheckpoint(
            monitor="auc@10",
            verbose=True,
            save_top_k=3,
            save_last=True,
            mode="max",
            auto_insert_metric_name=False,
            filename="{epoch:02d}-{auc@5:.3f}-{auc@10:.3f}-{auc@20:.3f}",
        ),
    ]
    if logger:
        callbacks.append(LearningRateMonitor(logging_interval="step"))

    trainer = pl.Trainer(
        default_root_dir="logs",
        accelerator="gpu" if args.gpus > 0 else "cpu",
        devices=args.gpus if args.gpus > 0 else 1,
        strategy=DDPStrategy(find_unused_parameters=False, process_group_backend="nccl"),
        num_nodes=args.num_nodes,
        max_epochs=args.max_epochs,
        log_every_n_steps=args.log_every_n_steps,
        precision="16-mixed" if config.WOVEN_ANCHOR.MP else 32,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        limit_val_batches=args.limit_val_batches,
        num_sanity_val_steps=args.num_sanity_val_steps,
        benchmark=args.benchmark,
        gradient_clip_val=config.TRAINER.GRADIENT_CLIPPING,
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=True,
        sync_batchnorm=config.TRAINER.WORLD_SIZE > 1,
        use_distributed_sampler=False,
        profiler=profiler,
    )
    trainer.fit(model, datamodule=data_module, ckpt_path=args.resume_ckpt_path)


if __name__ == "__main__":
    main()
