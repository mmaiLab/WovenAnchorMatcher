"""Coarse-to-fine image matching with optional anchor and local ERF analysis."""

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from kornia.utils import create_meshgrid

from src.utils.profiler import PassThroughProfiler
from src.woven_anchor.backbone import ConvNextV2_nano
from src.woven_anchor.coarse_matching import CoarseMatching
from src.woven_anchor.fine_matching import FineSubMatching
from src.woven_anchor.inter_fpn import InterFPN
from src.woven_anchor.token_mixer import MLPMixerL4
from src.woven_anchor.utils.utils import KeypointEncoder2d, complete_pair_masks
from src.woven_anchor.woven_anchor_module import WovenAnchorModule


class WovenAnchorMatcher(nn.Module):
    """Extract features, match on the 1/8 grid, then refine on the 1/2 grid."""

    def __init__(self, config, profiler=None, pretrained_backbone=True):
        super().__init__()
        self.config = config
        self.profiler = profiler or PassThroughProfiler()
        self.d_model_c = config["coarse"]["d_model"]
        self.d_model_m = config["fine"]["d_model_m"]
        self.d_model_f = config["fine"]["d_model"]
        self.backbone = ConvNextV2_nano(self.d_model_m, self.d_model_c, pretrained=pretrained_backbone)
        self.kenc = KeypointEncoder2d([32, 64, 128, self.d_model_c], self.d_model_c)
        self.wam = WovenAnchorModule(self.d_model_c, config["module"])
        self.coarse_matching = CoarseMatching(config["coarse"], self.profiler)
        self.inter_fpn = InterFPN([self.d_model_c, self.d_model_m, self.d_model_f])
        self.fine_enc = MLPMixerL4(2 * config["fine_window_size"] ** 2, self.d_model_f)
        self.fine_matching = FineSubMatching(config, self.profiler)

    def set_anchor_probability_capture(self, enabled: bool):
        self.wam.set_anchor_probability_capture(enabled)

    def set_anchor_probability_statistics(self, enabled: bool):
        self.wam.set_anchor_probability_statistics(enabled)

    def pop_anchor_probability_views(self):
        return self.wam.pop_anchor_probability_views()

    def pop_anchor_probability_statistics(self):
        return self.wam.pop_anchor_probability_statistics()

    def anchor_probability_analysis(self, data):
        """Compute anchor assignment probabilities without matching."""
        data["mode"] = "test"
        self.backbone_stage(data)
        self.wam(data)

    def receptive_field_analysis(self, data):
        """Run untimed local ERFs for every WovenMamba/OrthMamba layer."""
        with torch.inference_mode(False):
            # Clone outside inference mode so local ERFs can use autograd.
            analysis_data = {
                "imagec_0": data["imagec_0"].detach().clone(),
                "imagec_1": data["imagec_1"].detach().clone(),
            }
            for mask_name in ("mask0", "mask1"):
                mask = data.get(mask_name)
                if mask is not None:
                    analysis_data[mask_name] = mask.detach().clone()

            # Use normal autocast settings for the backbone and position encoding.
            with torch.autocast(
                enabled=self.config["mp"],
                device_type="cuda",
            ):
                with torch.no_grad():
                    self.backbone_stage(analysis_data)

                with torch.enable_grad():
                    return self.wam.receptive_field_analysis(
                        analysis_data["feat_8_0"],
                        analysis_data["feat_8_1"],
                        analysis_data.get("mask0"),
                        analysis_data.get("mask1"),
                    )

    @staticmethod
    def _valid_size_wh(features, mask=None):
        """Return valid (width, height) as [B, 2], assuming bottom/right padding."""
        if mask is None:
            batch_size, _, height, width = features.shape
            return features.new_tensor([width, height]).view(1, 2).expand(batch_size, -1)

        valid = mask > 0
        valid_w = valid[:, 0, :].sum(dim=1).to(features.dtype)
        valid_h = valid[:, :, 0].sum(dim=1).to(features.dtype)
        return torch.stack([valid_w, valid_h], dim=-1)

    def _add_position_encoding(self, features, mask=None):
        batch_size, _, height, width = features.shape
        grid = create_meshgrid(height, width, False, features.device).to(features.dtype)
        grid = grid.expand(batch_size, -1, -1, -1)
        size_wh = self._valid_size_wh(features, mask)
        return features + self.kenc(grid, size_wh)

    def backbone_stage(self, data):
        """Extract 1/4 and 1/8 features, then add position encodings at 1/8."""
        with self.profiler.profile("Backbone"):
            with torch.autocast(enabled=self.config["mp"], device_type="cuda"):
                self.backbone(data)

        coarse_features0, coarse_features1 = data["feat_8_0"], data["feat_8_1"]
        mask0, mask1 = complete_pair_masks(
            coarse_features0, coarse_features1, data.get("mask0"), data.get("mask1")
        )
        if mask0 is not None:
            # Coarse matching needs both masks if either image is padded.
            data.update(mask0=mask0, mask1=mask1)

        if data["hw0_c"] == data["hw1_c"]:
            coarse_features = torch.cat([coarse_features0, coarse_features1], dim=0)
            mask = torch.cat([mask0, mask1], dim=0) if mask0 is not None else None
            coarse_features = self._add_position_encoding(coarse_features, mask)
            coarse_features0, coarse_features1 = torch.chunk(coarse_features, 2, dim=0)
        else:
            coarse_features0 = self._add_position_encoding(coarse_features0, mask0)
            coarse_features1 = self._add_position_encoding(coarse_features1, mask1)

        data.update(feat_8_0=coarse_features0, feat_8_1=coarse_features1)

    def coarse_stage(self, data):
        """Aggregate context and select coarse correspondences."""
        with self.profiler.profile("WovenAnchorModule"):
            self.wam(data)

        with self.profiler.profile("CoarseMatching"):
            self.coarse_matching(data)

    @staticmethod
    def _extract_fine_windows(features, batch_ids, coarse_ids, coarse_width, window_size):
        """Gather [M, window_size**2, C] windows selected by coarse matches.

        Each step on the 1/8 coarse grid spans four pixels on the 1/2 fine grid.
        """
        stride = 4
        pad_left = pad_top = window_size // 2
        pad_right = pad_bottom = max(window_size - pad_left - stride, 0)

        coarse_x = coarse_ids % coarse_width
        coarse_y = coarse_ids // coarse_width
        fine_x = coarse_x * stride
        fine_y = coarse_y * stride

        dy, dx = torch.meshgrid(
            torch.arange(window_size, device=features.device),
            torch.arange(window_size, device=features.device),
            indexing="ij",
        )

        batch_indices = batch_ids[:, None, None]
        y_indices = fine_y[:, None, None] + dy
        x_indices = fine_x[:, None, None] + dx

        padded_features = F.pad(
            features,
            (pad_left, pad_right, pad_top, pad_bottom),
        ).permute(0, 2, 3, 1)  # [B, H, W, C]

        windows = padded_features[batch_indices, y_indices, x_indices]
        return rearrange(windows, "m w1 w2 c -> m (w1 w2) c")

    def fine_stage(self, data):
        """Upsample features, mix paired local windows, and refine matches."""
        window_size = self.config["fine_window_size"]
        window_area = window_size ** 2

        with self.profiler.profile("Fine preprocess"):
            if data["b_ids"].shape[0] == 0:
                device = data["feat_8_0"].device
                fine_windows0 = torch.empty(0, window_area, self.d_model_f, device=device)
                fine_windows1 = torch.empty(0, window_area, self.d_model_f, device=device)
            else:
                if (
                    data["feat_8_0"].shape == data["feat_8_1"].shape
                    and data["feat_4_0"].shape == data["feat_4_1"].shape
                ):
                    feat_8 = torch.cat([data["feat_8_0"], data["feat_8_1"]], dim=0)
                    feat_4 = torch.cat([data["feat_4_0"], data["feat_4_1"]], dim=0)
                    fine_features = self.inter_fpn(feat_8, feat_4)
                    fine_features0, fine_features1 = torch.chunk(fine_features, 2, dim=0)
                else:
                    fine_features0 = self.inter_fpn(data["feat_8_0"], data["feat_4_0"])
                    fine_features1 = self.inter_fpn(data["feat_8_1"], data["feat_4_1"])

                data.update({
                    "hw0_f": fine_features0.shape[2:],
                    "hw1_f": fine_features1.shape[2:],
                })

                fine_windows0 = self._extract_fine_windows(
                    fine_features0, data["b_ids"], data["i_ids"], data["hw0_c"][1], window_size
                )
                fine_windows1 = self._extract_fine_windows(
                    fine_features1, data["b_ids"], data["j_ids"], data["hw1_c"][1], window_size
                )

                paired_windows = torch.cat([fine_windows0, fine_windows1], dim=1).transpose(1, 2)
                paired_windows = self.fine_enc(paired_windows).transpose(1, 2)
                fine_windows0 = paired_windows[:, :window_area]
                fine_windows1 = paired_windows[:, window_area:]

            data.update(feat_f0=fine_windows0, feat_f1=fine_windows1)

        with self.profiler.profile("Fine Matching"):
            self.fine_matching(data)

    def forward(self, data, mode="test"):
        """Add coarse and refined matches to data."""
        data["mode"] = mode

        with self.profiler.profile("Backbone Stage"):
            self.backbone_stage(data)

        with self.profiler.profile("Coarse Stage"):
            self.coarse_stage(data)

        with self.profiler.profile("Fine Stage"):
            self.fine_stage(data)

        return data
