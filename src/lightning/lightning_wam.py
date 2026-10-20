# Adapted through JamMa from the LoFTR framework; modified for WovenAnchor.
# Inherited MIT/Apache-2.0 terms and attribution: THIRD_PARTY_NOTICES.md.

from collections import defaultdict
from loguru import logger
from pathlib import Path

import torch
import numpy as np
import pytorch_lightning as pl
from src.woven_anchor.matcher import WovenAnchorMatcher
from src.woven_anchor.utils.supervision import compute_supervision_fine, compute_supervision_coarse
from src.losses.loss import Loss
from src.optimizers import build_optimizer, build_scheduler
from src.utils.metrics import (
    compute_f1,
    compute_symmetrical_epipolar_errors,
    compute_pose_errors,
    aggregate_metrics_train_val, aggregate_metrics_test
)
from src.utils.homography import compute_homography_metrics, aggregate_homography_metrics
from src.utils.comm import gather, all_gather
from src.utils.misc import lower_config, flattenList
from src.utils.profiler import PassThroughProfiler
from src.utils.plotting import (
    make_anchor_probability_figures,
    make_matching_figures,
    make_woven_receptive_field_figures,
)

class WovenAnchorLightningModule(pl.LightningModule):
    def __init__(self, config, pretrained_ckpt=None, profiler=None):
        super().__init__()
        # Misc
        self.config = config  # full config
        _config = lower_config(self.config)
        self.profiler = profiler or PassThroughProfiler()
        self.visualization = None
        self.visualization_max_pairs = 10
        self.vis_name = None
        self.viz_path = Path('visualization')
        self.qualitative_viz_path = self.viz_path / "qualitative"
        self.anchor_viz_path = self.viz_path / "anchor_probs"
        self.anchor_statistics = False
        self._anchor_stats_accumulators = {}
        self.woven_erf_viz_path = self.viz_path / "woven_erf"

        # Matcher: WovenAnchorMatcher
        self.matcher = WovenAnchorMatcher(
            config=_config['woven_anchor'], profiler=profiler,
            pretrained_backbone=not pretrained_ckpt,
        )
        self.loss = Loss(_config)

        if pretrained_ckpt:
            state_dict = torch.load(pretrained_ckpt, map_location='cpu')['state_dict']
            self.load_state_dict(state_dict, strict=True)
            logger.info(f"Load '{pretrained_ckpt}' as pretrained checkpoint")

        # Testing
        self.test_data_source = _config['dataset']['test_data_source']
        self._train_epoch_outputs = []
        self._val_epoch_outputs = []
        self._test_epoch_outputs = []
        self.done_warmup_test = False
        self.total_ms = 0

        self._compute_model_stats()

    def _set_vis(self, mode=None, vis_name=None, max_pairs=10):
        if mode not in (None, "qualitative", "woven", "anchor"):
            raise ValueError(f"Unknown visualization mode: {mode}")
        if max_pairs < -1:
            raise ValueError("Visualization max_pairs must be -1 or non-negative")

        self.visualization = mode
        self.visualization_max_pairs = max_pairs
        self.vis_name = vis_name or "wam"
        self.matcher.set_anchor_probability_capture(False)

        if mode is not None and max_pairs != 0:
            output_dir = {
                "qualitative": self.qualitative_viz_path,
                "anchor": self.anchor_viz_path,
                "woven": self.woven_erf_viz_path,
            }[mode]
            output_dir.mkdir(parents=True, exist_ok=True)

    def _set_anchor_stats(self, enabled=False):
        self.anchor_statistics = bool(enabled)
        self._anchor_stats_accumulators.clear()
        self.matcher.set_anchor_probability_statistics(False)

    def _compute_model_stats(self):
        """Log total, trainable, and per-module parameter counts."""
        n_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total_parameters = sum(p.numel() for p in self.parameters())
        
        logger.info("="*60)
        logger.info("Model Statistics:")
        logger.info(f"  Trainable parameters: {n_parameters/1e6:.2f}M")
        logger.info(f"  Total parameters: {n_total_parameters/1e6:.2f}M")
        
        for name, module in self.named_children():
            module_param_count = sum(p.numel() for p in module.parameters())
            logger.info(f"  {name}: {module_param_count/1e6:.2f}M parameters")
        
        logger.info("="*60)

    # ----------------------------
    # Optimizer / Scheduler
    # ----------------------------
    def configure_optimizers(self):
        if self.config.TRAINER.SCHEDULER_INTERVAL == "step":
            self.config.TRAINER.COSA_TMAX = self.trainer.estimated_stepping_batches
        optimizer = build_optimizer(self, self.config)
        scheduler = build_scheduler(self.config, optimizer)
        return [optimizer], [scheduler]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None, **kwargs):
        warmup_step = self.config.TRAINER.WARMUP_STEP
        if self.global_step < warmup_step:
            if self.config.TRAINER.WARMUP_TYPE == 'linear':
                base_lr = self.config.TRAINER.WARMUP_RATIO * self.config.TRAINER.TRUE_LR
                lr = base_lr + (self.global_step / warmup_step) * abs(self.config.TRAINER.TRUE_LR - base_lr)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr
            elif self.config.TRAINER.WARMUP_TYPE == 'constant':
                pass
            else:
                raise ValueError(f'Unknown lr warm-up strategy: {self.config.TRAINER.WARMUP_TYPE}')

        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    # ----------------------------
    # Inference helpers
    # ----------------------------
    def _train_val_inference(self, batch, mode):
        with self.profiler.profile("Compute coarse supervision"):
            compute_supervision_coarse(batch, self.config)

        with self.profiler.profile("Matcher"):
            self.matcher(batch, mode=mode) # mode = "train", "val"

        with self.profiler.profile("Compute fine supervision"):
            with torch.autocast(enabled=self.config.WOVEN_ANCHOR.MP, device_type='cuda'):
                compute_supervision_fine(batch, self.config)

        with self.profiler.profile("Compute losses"):
            with torch.autocast(enabled=self.config.WOVEN_ANCHOR.MP, device_type='cuda'):
                self.loss(batch)

    def _warmup_test(self, batch):
        if self.done_warmup_test:
            return

        profiler = self.matcher.profiler
        self.matcher.profiler = PassThroughProfiler()
        try:
            with torch.autocast(enabled=self.config.WOVEN_ANCHOR.MP, device_type='cuda'):
                for _ in range(50):
                    self.matcher(batch, mode="test")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        finally:
            self.matcher.profiler = profiler

        self.done_warmup_test = True
    
    def _timed_test_inference(self, batch):
        with torch.autocast(enabled=self.config.WOVEN_ANCHOR.MP, device_type='cuda'):
            with self.profiler.profile("[Test] Matcher"):
                self.start_event.record()
                self.matcher(batch, mode='test')
                self.end_event.record()
                self.end_event.synchronize()
            elapsed_time = self.start_event.elapsed_time(self.end_event)
            self.total_ms += elapsed_time
            batch['runtime'] = elapsed_time

    def _compute_metrics(self, batch, validation=False):
        with self.profiler.profile("[Ignore] Compute metrics" if validation else "Compute metrics"):
            if validation:
                compute_f1(batch)

            rel_pair_names = list(zip(*batch['pair_names']))
            bs = batch['imagec_0'].size(0)
            metrics = {
                'identifiers': ['#'.join(rel_pair_names[b]) for b in range(bs)],
            }

            if validation or self.test_data_source in ['MegaDepth', 'ScanNet']:
                compute_symmetrical_epipolar_errors(batch)
                compute_pose_errors(batch, self.config)
                metrics.update({
                    'epi_errs': [batch['epi_errs'][batch['m_bids'] == b].cpu().numpy() for b in range(bs)],
                    'R_errs': batch['R_errs'],
                    't_errs': batch['t_errs'],
                    'inliers': batch['inliers'],
                })
                if validation:
                    metrics.update({
                        'precision': batch['precision'],
                        'recall': batch['recall'],
                        'f1_score': batch['f1_score'],
                        'max_matches': [batch['num_candidates_max']],
                    })
            elif self.test_data_source == 'HPatches':
                metrics.update(compute_homography_metrics(batch, self.config).get('metrics', {}))

        return {'metrics': metrics}

    # ----------------------------
    # Training
    # ----------------------------
    def training_step(self, batch, batch_idx):
        self._train_val_inference(batch, "train")
        self._train_epoch_outputs.append({'loss': batch['loss'].detach().clone()})
        if self.trainer.global_rank == 0 and self.global_step % self.trainer.log_every_n_steps == 0:
            self.log('train/loss_step', batch['loss'], on_step=True, on_epoch=False, rank_zero_only=True)
        return {'loss': batch['loss']}

    def on_train_epoch_end(self):
        if not self._train_epoch_outputs:
            return
        avg_loss = torch.stack([x['loss'] for x in self._train_epoch_outputs]).mean()
        self.log('train/avg_loss_on_epoch', avg_loss, on_epoch=True, rank_zero_only=True)
        self._train_epoch_outputs.clear()

    # ----------------------------
    # Validation
    # ----------------------------
    def validation_step(self, batch, batch_idx):
        self._train_val_inference(batch, "val")
        out = self._compute_metrics(batch, validation=True)
        out['loss_scalars'] = {
            k: v.detach().cpu().clone() if torch.is_tensor(v) else v
            for k, v in batch['loss_scalars'].items()
        }
        self._val_epoch_outputs.append(out)
        return out

    def on_validation_epoch_end(self):
        if not self._val_epoch_outputs:
            return

        multi_outputs = (
            [self._val_epoch_outputs]
            if not isinstance(self._val_epoch_outputs[0], (list, tuple))
            else self._val_epoch_outputs
        )

        def collect(outputs, field):
            d = defaultdict(list)
            for o in outputs:
                x = (o or {}).get(field)
                if not x:
                    continue
                for k, v in x.items():
                    d[k].extend(v if isinstance(v, list) else [v])
            return d

        auc_all = defaultdict(list)
        last_metrics = None
        exp = getattr(self.logger, "experiment", None)

        for vidx, outputs in enumerate(multi_outputs):
            loss_scalars = {k: flattenList(all_gather(v)) for k, v in collect(outputs, "loss_scalars").items() if v}
            metrics_raw  = {k: flattenList(all_gather(v)) for k, v in collect(outputs, "metrics").items() if v}

            if not metrics_raw:
                continue

            val_metrics = aggregate_metrics_train_val(
                metrics_raw,
                self.config.TRAINER.EPI_ERR_THR,
                config=self.config
            )
            last_metrics = val_metrics

            for thr in [5, 10, 20]:
                if f"auc@{thr}" in val_metrics:
                    auc_all[f"auc@{thr}"].append(val_metrics[f"auc@{thr}"])

            if self.trainer.global_rank == 0 and exp is not None:
                if hasattr(exp, "add_scalar"):
                    for k, v in loss_scalars.items():
                        vals = [x.item() if torch.is_tensor(x) else float(x) for x in v]
                        if vals:
                            exp.add_scalar(f"val_{vidx}/avg_{k}", torch.tensor(vals).mean(), global_step=self.current_epoch)
                    for k, v in val_metrics.items():
                        exp.add_scalar(f"metrics_{vidx}/{k}", v, global_step=self.current_epoch)

        for thr in [5, 10, 20]:
            vals = auc_all[f"auc@{thr}"]
            if vals:
                self.log(f"auc@{thr}", float(np.mean(vals)), on_epoch=True, prog_bar=True, sync_dist=True)

        if last_metrics is not None:
            print(last_metrics)

        self._val_epoch_outputs.clear()

    # ----------------------------
    # Test
    # ----------------------------
    def on_test_start(self):
        assert self.config.TRAINER.TRUE_BATCH_SIZE == 1
        self.total_ms = 0.0
        self._test_epoch_outputs.clear()
        self._anchor_stats_accumulators.clear()

        if torch.cuda.is_available():
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)

    def test_step(self, batch, batch_idx):
        self.matcher.set_anchor_probability_capture(False)
        self.matcher.set_anchor_probability_statistics(False)
        self._warmup_test(batch)
        vis_mode = self.visualization if (
            self.visualization_max_pairs < 0
            or batch_idx < self.visualization_max_pairs
        ) else None
        self._timed_test_inference(batch)
        ret_dict = self._compute_metrics(batch)

        if vis_mode == "qualitative":
            self._save_test_visualization(batch, batch_idx)
        if vis_mode == "anchor" or self.anchor_statistics:
            self._run_anchor_probability_analysis(
                batch, batch_idx, visualize=vis_mode == "anchor"
            )
        if vis_mode == "woven":
            self._run_receptive_field_analysis(batch, batch_idx)

        self._test_epoch_outputs.append(ret_dict)
        return ret_dict

    def on_test_epoch_end(self):
        outputs = self._test_epoch_outputs
        if not outputs:
            return

        if self.anchor_statistics:
            self._finalize_anchor_probability_statistics()
            
        metrics_list = [o.get("metrics") for o in outputs if o.get("metrics") is not None]
        if not metrics_list:
            if self.trainer.global_rank == 0:
                print("No metrics found in test outputs")
            self._test_epoch_outputs.clear()
            return

        if self.test_data_source in ["MegaDepth", "ScanNet"]:
            aggregated = self._aggregate_relative_pose(metrics_list)
        elif self.test_data_source == "HPatches":
            aggregated = self._aggregate_homography(metrics_list)
        else:
            raise NotImplementedError(f"Unsupported test_data_source: {self.test_data_source}")            
        
        num_pairs = len(outputs)
        avg_time_ms = self.total_ms / max(num_pairs, 1)
        throughput = 1000.0 / avg_time_ms if avg_time_ms > 0 else 0.0
        n_params_m = sum(p.numel() for p in self.parameters()) / 1e6

        if self.trainer.global_rank == 0:
            self._print_test_summary(
                dataset=self.test_data_source,
                metrics=aggregated,
                num_pairs=num_pairs,
                avg_time_ms=avg_time_ms,
                throughput=throughput,
                n_params_m=n_params_m,
            )

            profile_summary = self.profiler.summary()
            if profile_summary:
                print(profile_summary)

        self._test_epoch_outputs.clear()


    def _save_test_visualization(self, batch, batch_idx):
        path = f"{self.qualitative_viz_path}/{batch_idx}_{self.vis_name}.png"
        make_matching_figures(batch, 'evaluation', path=path)

    def _run_anchor_probability_analysis(self, batch, batch_idx, visualize):
        profiler = self.matcher.profiler
        self.matcher.set_anchor_probability_capture(visualize)
        self.matcher.set_anchor_probability_statistics(self.anchor_statistics)
        try:
            self.matcher.profiler = PassThroughProfiler()
            try:
                with torch.autocast(
                    enabled=self.config.WOVEN_ANCHOR.MP,
                    device_type='cuda',
                ):
                    self.matcher.anchor_probability_analysis(batch)
            finally:
                self.matcher.profiler = profiler

            if self.anchor_statistics:
                statistics = self.matcher.pop_anchor_probability_statistics()
                if statistics:
                    self._accumulate_anchor_probability_statistics(statistics)
                else:
                    logger.warning(
                        f"No anchor statistics captured for test pair {batch_idx}."
                    )

            if visualize:
                visualizations = self.matcher.pop_anchor_probability_views()
                if not visualizations:
                    logger.warning(
                        f"No anchor probabilities captured for test pair {batch_idx}."
                    )
                else:
                    filename_prefix = f"{batch_idx:04d}_{self.vis_name}"
                    for visualization in visualizations:
                        make_anchor_probability_figures(
                            batch,
                            visualization,
                            output_dir=self.anchor_viz_path,
                            filename_prefix=filename_prefix,
                        )
        finally:
            self.matcher.set_anchor_probability_capture(False)
            self.matcher.set_anchor_probability_statistics(False)

    def _run_receptive_field_analysis(self, batch, batch_idx):
        """Create local ERFs for each supported layer outside profiler statistics."""
        profiler = self.matcher.profiler
        self.matcher.profiler = PassThroughProfiler()
        try:
            layer_analyses = self.matcher.receptive_field_analysis(batch)
        finally:
            self.matcher.profiler = profiler

        filename_prefix = f"{batch_idx:04d}_{self.vis_name}"
        for layer_analysis in layer_analyses:
            saved_paths = make_woven_receptive_field_figures(
                batch,
                layer_analysis,
                output_dir=self.woven_erf_viz_path,
                filename_prefix=filename_prefix,
            )
            for output_path in saved_paths:
                logger.info(f"Saved ERF figure to '{output_path}'.")

    def _accumulate_anchor_probability_statistics(self, statistics):
        for block_statistics in statistics:
            block_index = int(block_statistics["anchor_block_index"])
            layer_index = int(block_statistics["layer_index"])
            n_anchors = int(block_statistics["n_anchors"])
            key = (block_index, layer_index)

            anchor_prob_sum = np.asarray(block_statistics["sum"], dtype=np.float64)
            anchor_prob_sq_sum = np.asarray(
                block_statistics["sum_sq"], dtype=np.float64
            )
            if (
                anchor_prob_sum.shape != (n_anchors,)
                or anchor_prob_sq_sum.shape != (n_anchors,)
            ):
                raise RuntimeError(
                    f"Invalid anchor statistics shape for block {block_index}: "
                    f"sum={anchor_prob_sum.shape}, sum_sq={anchor_prob_sq_sum.shape}."
                )

            accumulator = self._anchor_stats_accumulators.setdefault(
                key,
                {
                    "n_anchors": n_anchors,
                    "sum": np.zeros(n_anchors, dtype=np.float64),
                    "sum_sq": np.zeros(n_anchors, dtype=np.float64),
                    "count": 0,
                },
            )
            if accumulator["n_anchors"] != n_anchors:
                raise RuntimeError(f"Anchor count changed for block {block_index}.")

            accumulator["sum"] += anchor_prob_sum
            accumulator["sum_sq"] += anchor_prob_sq_sum
            accumulator["count"] += int(block_statistics["count"])

    def _finalize_anchor_probability_statistics(self):
        local_statistics = self._anchor_stats_accumulators
        self._anchor_stats_accumulators = {}
        gathered_statistics = gather(local_statistics)
        if self.trainer.global_rank != 0:
            return

        merged = {}
        for rank_statistics in gathered_statistics:
            for key, rank_accumulator in rank_statistics.items():
                n_anchors = int(rank_accumulator["n_anchors"])
                accumulator = merged.setdefault(
                    key,
                    {
                        "n_anchors": n_anchors,
                        "sum": np.zeros(n_anchors, dtype=np.float64),
                        "sum_sq": np.zeros(n_anchors, dtype=np.float64),
                        "count": 0,
                    },
                )
                if accumulator["n_anchors"] != n_anchors:
                    raise RuntimeError(f"Anchor count differs across ranks for block {key[0]}.")
                accumulator["sum"] += np.asarray(
                    rank_accumulator["sum"], dtype=np.float64
                )
                accumulator["sum_sq"] += np.asarray(
                    rank_accumulator["sum_sq"], dtype=np.float64
                )
                accumulator["count"] += int(rank_accumulator["count"])

        rows = []
        for (block_index, layer_index), accumulator in sorted(merged.items()):
            count = accumulator["count"]
            for anchor_index in range(accumulator["n_anchors"]):
                if count > 0:
                    mean = accumulator["sum"][anchor_index] / count
                    second_moment = accumulator["sum_sq"][anchor_index] / count
                    variance = max(second_moment - mean * mean, 0.0)
                else:
                    mean = float("nan")
                    variance = float("nan")

                rows.append(
                    {
                        "anchor_block_index": block_index,
                        "layer_index": layer_index,
                        "anchor_index": anchor_index,
                        "valid_count": count,
                        "mean": mean,
                        "population_variance": variance,
                    }
                )

        if not rows:
            logger.warning("No anchor probability statistics were accumulated.")
            return

        print("\n" + "=" * 86)
        print("Anchor probability statistics")
        print(
            "Aggregation: anchor assignment probabilities over valid token-window entries "
            "from both images and all pairs"
        )
        print(f"{'Block':>7} {'Layer':>7} {'Anchor':>8} {'Count':>12} {'Mean':>16} {'Variance(pop.)':>18}")
        print("-" * 86)
        for row in rows:
            print(
                f"{row['anchor_block_index']:>7d} "
                f"{row['layer_index']:>7d} "
                f"{row['anchor_index']:>8d} "
                f"{row['valid_count']:>12d} "
                f"{row['mean']:>16.10f} "
                f"{row['population_variance']:>18.10f}"
            )
        print("=" * 86)

    def _aggregate_homography(self, metrics_list):
        merged = defaultdict(list)
        for metrics in metrics_list:
            for k, v in metrics.items():
                if k == "identifiers":
                    continue
                merged[k].extend(v if isinstance(v, list) else [v])

        # Rebuild per-pair records for homography aggregation.
        metrics_for_aggregation = []
        n = max((len(v) for v in merged.values()), default=0)
        for i in range(n):
            row = {}
            for k, vals in merged.items():
                if i < len(vals):
                    row[k] = vals[i]
            if row:
                metrics_for_aggregation.append(row)

        return aggregate_homography_metrics(metrics_for_aggregation, self.config)


    def _aggregate_relative_pose(self, metrics_list):
        merged = defaultdict(list)
        for metrics in metrics_list:
            for k, v in metrics.items():
                merged[k].extend(v if isinstance(v, list) else [v])

        gathered = {k: flattenList(gather(v)) for k, v in merged.items()}
        return aggregate_metrics_test(
            gathered,
            self.config.TRAINER.EPI_ERR_THR,
            config=self.config
        )



    def _print_test_summary(self, dataset, metrics, num_pairs, avg_time_ms, throughput, n_params_m):
        print("\n" + "=" * 60)
        print(f"Dataset              : {dataset}")
        print(f"Pairs evaluated      : {num_pairs}")
        print(f"Parameters           : {n_params_m:.4g} M")
        print(f"Avg matching time    : {avg_time_ms:.4g} ms")
        print(f"Throughput           : {throughput:.4g} FPS")

        important_keys = []

        if dataset == "HPatches":
            important_keys = [
                "H_error_ransac@3px", "H_error_ransac@5px", "H_error_ransac@10px",
                "H_error_dlt@3px", "H_error_dlt@5px", "H_error_dlt@10px",
                "mean_prec@3px", "mean_prec@5px", "mean_prec@10px",
                "mean_num_matches",
            ]
        else:
            important_keys = ["auc@5", "auc@10", "auc@20"]
            important_keys += [
                key for key in metrics if key.startswith(("prec@", "num_matches@"))
            ]

        print("-" * 60)
        for k in important_keys:
            if k in metrics:
                v = metrics[k]
                if isinstance(v, float):
                    print(f"{k:<22}: {v:.5f}")
                else:
                    print(f"{k:<22}: {v}")

        print("=" * 60 + "\n")
