"""AnchorCrossAttention and its within-image ablation, AnchorSelfAttention."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.woven_anchor.utils.utils import RMSNormDim
from .anchor_analysis import AnchorProbabilityAnalysis
from .anchor_components import (
    Attention,
    CNNGeoEncoder,
    LocalWindowNorm,
    StatsGeoEncoder,
)
from .window_ops import window_fold, window_unfold


class AnchorAttentionBlock(nn.Module):
    """ACA/ASA: anchor construction, attention, projection, and overlap fusion.

    Inputs are [B, C, H, W] maps with optional [B, H, W] validity masks.
    Paper symbols: d_model=C_c, n_anchors=K, dim_anchor=D, window_size=w.
    With x1=None, x0 contains [image0_batch; image1_batch]. Equal-size pairs
    share a batched path; cross-attention also supports different image sizes.
    """

    def __init__(
        self,
        d_model: int,
        n_anchors: int = 4,
        dim_anchor: int = 256,
        window_size: int = 8,
        use_overlap: bool = True,
        n_heads: int = 4,
        use_anc_norm: bool = False,
        use_f_raw: bool = True,
        use_f_norm: bool = True,
        use_f_geo: bool = True,
        dim_geo: int = 16,
        geo_mode: str = "stats",
        overlap_weight_mode: str = "entropy",
        device=None,
        dtype=None,
        *,
        assignment_mode: str = "learned",
        attention_scope: str = "per_anchor",
        attention_mode: str = "cross",
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        assert use_f_raw or use_f_norm or use_f_geo
        assert geo_mode in ["stats", "cnn"]
        if assignment_mode not in {"learned", "uniform"}:
            raise ValueError(
                "assignment_mode must be either 'learned' or 'uniform', "
                f"but got {assignment_mode!r}."
            )
        if attention_scope not in {"per_anchor", "global"}:
            raise ValueError(
                "attention_scope must be either 'per_anchor' or 'global', "
                f"but got {attention_scope!r}."
            )
        if attention_mode not in {"self", "cross"}:
            raise ValueError(
                "attention_mode must be either 'self' or 'cross', "
                f"but got {attention_mode!r}."
            )
        if overlap_weight_mode not in {"entropy", "uniform"}:
            raise ValueError(
                "overlap_weight_mode must be either 'entropy' or 'uniform', "
                f"but got {overlap_weight_mode!r}."
            )

        self.d_model = d_model
        self.n_anchors = n_anchors
        self.dim_anchor = dim_anchor
        self.window_size = window_size
        self.use_overlap = use_overlap
        self.n_heads = n_heads
        self.use_anc_norm = use_anc_norm
        self.use_f_raw = use_f_raw
        self.use_f_norm = use_f_norm
        self.use_f_geo = use_f_geo
        self.geo_mode = geo_mode
        self.assignment_mode = assignment_mode
        self.attention_scope = attention_scope
        self.attention_mode = attention_mode
        self.overlap_weight_mode = overlap_weight_mode

        self.local_norm = LocalWindowNorm(window_size=window_size)
        # Keep frozen assignment parameters for checkpoint compatibility in uniform ablations.
        learn_assignment = self.assignment_mode == "learned"
        # cluster_conv.weight[k, :, 0, 0] is q_k; tau = softplus(cluster_temp).
        self.cluster_conv = nn.Conv2d(
            d_model, n_anchors, kernel_size=1, bias=False, **factory_kwargs
        )
        self.cluster_conv.requires_grad_(learn_assignment)
        self.cluster_temp = nn.Parameter(
            torch.tensor(math.sqrt(d_model), **factory_kwargs),
            requires_grad=learn_assignment,
        )

        if self.use_f_geo:
            if self.geo_mode == "stats":
                self.geo_encoder = StatsGeoEncoder(dim_geo, window_size)
            else:
                self.geo_encoder = CNNGeoEncoder(dim_geo, window_size)

        anchor_input_dim = 0
        if use_f_raw:
            anchor_input_dim += d_model
        if use_f_norm:
            anchor_input_dim += d_model
        if use_f_geo:
            anchor_input_dim += dim_geo

        if self.use_anc_norm:
            self.anc_norm = RMSNormDim(anchor_input_dim, target_dim=-1)
        self.anchor_proj = nn.Linear(anchor_input_dim, dim_anchor, **factory_kwargs)
        self.cross_attn = Attention(dim_anchor, n_heads=n_heads, bias=False)
        self.delta_proj = nn.Linear(dim_anchor, d_model, **factory_kwargs)

        nn.init.constant_(self.delta_proj.weight, 0.0)
        nn.init.constant_(self.delta_proj.bias, 0.0)

        # Entropy-to-weight scale and bias; frozen for uniform overlap weights.
        learn_overlap_weights = self.overlap_weight_mode == "entropy"
        self.overlap_weight_scale = nn.Parameter(
            torch.tensor(2.0, **factory_kwargs),
            requires_grad=learn_overlap_weights,
        )
        self.overlap_weight_bias = nn.Parameter(
            torch.tensor(-1.0, **factory_kwargs),
            requires_grad=learn_overlap_weights,
        )

        # Analysis adds no state_dict entries.
        self._analysis = AnchorProbabilityAnalysis(n_anchors, window_size, use_overlap)

    def set_anchor_probability_capture(self, enabled: bool):
        self._analysis.set_capture(enabled)

    def set_anchor_probability_statistics(self, enabled: bool):
        self._analysis.set_statistics(enabled)

    def pop_anchor_probability_views(self):
        return self._analysis.pop_views()

    def pop_anchor_probability_statistics(self):
        return self._analysis.pop_statistics()

    def _format_mask(self, mask, x):
        if mask is None:
            return None
        if mask.dim() == 3:
            return mask.unsqueeze(1).to(dtype=x.dtype, device=x.device)
        if mask.dim() == 4 and mask.shape[1] == 1:
            return mask.to(dtype=x.dtype, device=x.device)
        raise ValueError(f"Unsupported mask shape: {tuple(mask.shape)}")

    def _window_avg_pool(self, x_unfolded):
        """Average each window in the unfolded layout."""
        return F.avg_pool2d(
            x_unfolded, kernel_size=self.window_size,
            stride=self.window_size, count_include_pad=False,
        )

    def _window_unfold(self, x, mask):
        """Unfold, using the efficient offset-window layout."""
        return window_unfold(x, mask, self.window_size, self.use_overlap)

    def _window_fold(self, x_unfolded, original_hw, pad_hw):
        """Fold, summing overlapping updates on the image grid."""
        return window_fold(x_unfolded, original_hw, pad_hw, self.window_size, self.use_overlap)

    def _compute_assignment_probabilities(self, x_unfolded_norm, mask_unfolded):
        """Return assignment probabilities [B, K, H_u, W_u], zero at masked positions."""
        B, _, H_u, W_u = x_unfolded_norm.shape
        if self.assignment_mode == "uniform":
            assign_probs = x_unfolded_norm.new_full(
                (B, self.n_anchors, H_u, W_u), 1.0 / self.n_anchors
            )
        else:
            inv_temperature = 1.0 / F.softplus(self.cluster_temp)
            logits = self.cluster_conv(x_unfolded_norm) * inv_temperature
            assign_probs = logits.softmax(dim=1)

        if mask_unfolded is not None:
            assign_probs = assign_probs * mask_unfolded
        return assign_probs

    def _compute_window_weights(self, assign_probs):
        """Compute unnormalized window weights from assignment entropy, or use ones."""
        if self.overlap_weight_mode == "uniform":
            return torch.ones_like(assign_probs[:, :1])

        assignment_entropy = -(assign_probs * (assign_probs + 1e-6).log()).sum(dim=1, keepdim=True)
        return torch.sigmoid(
            assignment_entropy * self.overlap_weight_scale + self.overlap_weight_bias
        )

    def _construct_anchors(self, x_unfolded, mask_unfolded):
        """Construct anchors from f_raw, f_norm, and f_geo."""
        B, C, H_u, W_u = x_unfolded.shape
        w = self.window_size

        x_unfolded_norm = self.local_norm(x_unfolded, mask_unfolded)

        assign_probs = self._compute_assignment_probabilities(x_unfolded_norm, mask_unfolded)

        window_weights = self._compute_window_weights(assign_probs)
        # Normalize by mean assignment weight, with a floor for near-empty anchors.
        anchor_pool_scale = 1.0 / self._window_avg_pool(assign_probs).clamp_min(1e-6)
        H_a, W_a = anchor_pool_scale.shape[2:]

        feat_list = []
        if self.use_f_raw:
            weighted_raw = (x_unfolded.unsqueeze(1) * assign_probs.unsqueeze(2)).reshape(
                B * self.n_anchors, C, H_u, W_u
            )
            f_raw = self._window_avg_pool(weighted_raw).view(B, self.n_anchors, C, H_a, W_a) * anchor_pool_scale.unsqueeze(2)
            feat_list.append(f_raw.permute(0, 1, 3, 4, 2))  # [B, K, H_a, W_a, C]

        if self.use_f_norm:
            weighted_norm = (x_unfolded_norm.unsqueeze(1) * assign_probs.unsqueeze(2)).reshape(
                B * self.n_anchors, C, H_u, W_u
            )
            f_norm = self._window_avg_pool(weighted_norm).view(B, self.n_anchors, C, H_a, W_a) * anchor_pool_scale.unsqueeze(2)
            feat_list.append(f_norm.permute(0, 1, 3, 4, 2))  # [B, K, H_a, W_a, C]

        if self.use_f_geo:
            assign_probs_patch = (
                assign_probs.reshape(B, self.n_anchors, H_a, w, W_a, w)
                .permute(0, 1, 2, 4, 3, 5)
                .contiguous()
            )
            f_geo = self.geo_encoder(assign_probs_patch)  # [B, K, H_a, W_a, dim_geo]
            feat_list.append(f_geo)

        anchor_inputs = torch.cat(feat_list, dim=-1)
        if self.use_anc_norm:
            anchor_inputs = self.anc_norm(anchor_inputs)
        anchors = self.anchor_proj(anchor_inputs).reshape(
            B, self.n_anchors, H_a * W_a, -1
        )  # [B, K, M, D], where M = H_a * W_a windows.

        if mask_unfolded is None:
            anchor_mask = None
        else:
            valid_win = self._window_avg_pool(mask_unfolded).squeeze(1) > 0
            anchor_mask = valid_win.reshape(B, 1, H_a * W_a).expand(-1, self.n_anchors, -1)

        return anchors, anchor_mask, assign_probs, window_weights, (H_a, W_a)

    def _overlap_fusion(self, anchor_updates, assign_probs, window_weights, ori_hw, pad_hw):
        """Project anchor updates with assignment weights and fuse overlapping windows.

        Return the feature update before residual addition.
        """
        # [B,K,H_a,W_a,C] -> [B,K,H_u,W_u,C]
        w = self.window_size
        window_updates = anchor_updates.repeat_interleave(w, dim=2).repeat_interleave(w, dim=3)

        # Fold sums weights at each pixel; Unfold maps the sum back to each window.
        window_weight_sum, _, _, _ = self._window_unfold(
            self._window_fold(window_weights, ori_hw, pad_hw), None
        )
        # Guard uncovered padded positions, where assignment probabilities are zero.
        safe_window_weight_sum = torch.where(
            window_weight_sum > 0,
            window_weight_sum.clamp_min(1e-6),
            torch.ones_like(window_weight_sum),
        )
        overlap_weights = window_weights / safe_window_weight_sum
        fold_weights = assign_probs * overlap_weights

        # Project anchor updates to each window's features.
        window_updates = (
            (window_updates * fold_weights.unsqueeze(-1)).sum(dim=1).permute(0, 3, 1, 2)
        )

        return self._window_fold(window_updates, ori_hw, pad_hw)

    def _reshape_anchor_sequences(self, anchors, mask=None):
        """Group attention sequences by anchor ID: [B,K,M,D] -> [B*K,M,D].

        The joint-attention ablation combines IDs into [B,K*M,D] instead.
        """
        B, K, M, D = anchors.shape
        if self.attention_scope == "per_anchor":
            sequence = anchors.reshape(B * K, M, D)
        else:
            sequence = anchors.reshape(B, K * M, D)
        if mask is not None:
            mask = mask.reshape(sequence.shape[:2])
        return sequence, mask

    def _attend_anchors_batched(self, anchors, mask=None):
        """Add attention context to anchors before delta_proj."""
        if self.attention_mode == "cross":
            # An even anchor count must not hide an unpaired image batch.
            assert anchors.shape[0] % 2 == 0, "Cross-attention expects paired image batches."
        sequence, sequence_mask = self._reshape_anchor_sequences(anchors, mask)
        if self.attention_mode == "self":
            context = self.cross_attn.forward_self(sequence, sequence_mask)
        else:
            context = self.cross_attn(sequence, m0=sequence_mask)
        return context.reshape_as(anchors) + anchors

    def _forward_sep(self, x0, x1, mask0=None, mask1=None):
        residual0 = x0
        residual1 = x1

        B, C, _, _ = x0.shape

        m0 = self._format_mask(mask0, x0)
        m1 = self._format_mask(mask1, x1)

        x_unfolded0, mask_unfolded0, ori_hw0, pad_hw0 = self._window_unfold(x0, m0)
        x_unfolded1, mask_unfolded1, ori_hw1, pad_hw1 = self._window_unfold(x1, m1)

        anchors0, anchor_mask0, assign_probs0, window_weights0, window_hw0 = self._construct_anchors(
            x_unfolded0, mask_unfolded0
        )
        anchors1, anchor_mask1, assign_probs1, window_weights1, window_hw1 = self._construct_anchors(
            x_unfolded1, mask_unfolded1
        )

        if self._analysis.capture_anchor_probs:
            self._analysis.cache_probabilities(
                assign_probs0,
                ori_hw0,
                pad_hw0,
                assign_probs1,
                ori_hw1,
                pad_hw1,
            )

        if self._analysis.capture_anchor_stats:
            self._analysis.cache_statistics(assign_probs0, assign_probs1)

        seq0, seq_mask0 = self._reshape_anchor_sequences(anchors0, anchor_mask0)
        seq1, seq_mask1 = self._reshape_anchor_sequences(anchors1, anchor_mask1)
        context0, context1 = self.cross_attn(seq0, seq1, seq_mask0, seq_mask1)
        context0, context1 = context0.reshape_as(anchors0), context1.reshape_as(anchors1)

        anchor_updates0 = self.delta_proj(context0 + anchors0).reshape(
            B, self.n_anchors, window_hw0[0], window_hw0[1], -1
        )
        anchor_updates1 = self.delta_proj(context1 + anchors1).reshape(
            B, self.n_anchors, window_hw1[0], window_hw1[1], -1
        )

        feature_updates0 = self._overlap_fusion(
            anchor_updates0, assign_probs0, window_weights0, ori_hw0, pad_hw0
        )
        feature_updates1 = self._overlap_fusion(
            anchor_updates1, assign_probs1, window_weights1, ori_hw1, pad_hw1
        )

        if m0 is not None:
            feature_updates0 = feature_updates0 * m0
        if m1 is not None:
            feature_updates1 = feature_updates1 * m1

        out0 = residual0 + feature_updates0
        out1 = residual1 + feature_updates1

        return out0, out1

    def _forward_batched(self, x, mask=None):
        residual = x

        B, C, _, _ = x.shape

        m = self._format_mask(mask, x)

        x_unfolded, mask_unfolded, ori_hw, pad_hw = self._window_unfold(x, m)

        anchors, anchor_mask, assign_probs, window_weights, window_hw = self._construct_anchors(
            x_unfolded, mask_unfolded
        )

        if self._analysis.capture_anchor_probs:
            if B % 2 != 0:
                raise RuntimeError("Batched anchor visualization expects image-pair batches.")
            assign_probs0, assign_probs1 = torch.chunk(assign_probs, 2, dim=0)
            self._analysis.cache_probabilities(
                assign_probs0,
                ori_hw,
                pad_hw,
                assign_probs1,
                ori_hw,
                pad_hw,
            )

        if self._analysis.capture_anchor_stats:
            self._analysis.cache_statistics(assign_probs)

        anchors = self._attend_anchors_batched(anchors, anchor_mask)

        anchor_updates = self.delta_proj(anchors).reshape(
            B, self.n_anchors, window_hw[0], window_hw[1], -1
        )

        feature_updates = self._overlap_fusion(
            anchor_updates, assign_probs, window_weights, ori_hw, pad_hw
        )

        if m is not None:
            feature_updates = feature_updates * m

        return residual + feature_updates

    def forward(self, x0, x1=None, mask0=None, mask1=None):
        if x1 is not None:
            if self.attention_mode != "cross":
                raise ValueError(
                    "Anchor self-attention requires equal-size image "
                    "pairs through the batched path."
                )
            return self._forward_sep(x0, x1, mask0, mask1)
        return self._forward_batched(x0, mask0)


class AnchorCrossAttention(AnchorAttentionBlock):
    """ACA: retrieve context from the paired image's anchors."""

    def __init__(self, d_model, **kwargs):
        kwargs["attention_mode"] = "cross"
        super().__init__(d_model, **kwargs)


class AnchorSelfAttention(AnchorAttentionBlock):
    """ASA ablation: attend to anchors within each image."""

    def __init__(self, d_model, **kwargs):
        kwargs["attention_mode"] = "self"
        super().__init__(d_model, **kwargs)
