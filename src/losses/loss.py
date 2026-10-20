# Adapted through JamMa from the LoFTR framework; modified for WovenAnchor.
# Inherited MIT/Apache-2.0 terms and attribution: THIRD_PARTY_NOTICES.md.

import torch
import torch.nn as nn
from kornia.geometry.conversions import convert_points_to_homogeneous
from kornia.geometry.epipolar import numeric


class Loss(nn.Module):
    """Coarse matching, fine matching, and sub-pixel refinement losses."""

    def __init__(self, config):
        super().__init__()
        self.loss_config = config["woven_anchor"]["loss"]
        self.pos_w = self.loss_config["pos_weight"]
        self.neg_w = self.loss_config["neg_weight"]

    def compute_fine_matching_loss(self, data):
        """Focal loss on fine-window confidences of shape (N, W_f^2, W_f^2)."""
        conf_matrix_fine = data["conf_matrix_fine"]
        conf_matrix_f_gt = data["conf_matrix_f_gt"]
        pos_mask, neg_mask = conf_matrix_f_gt > 0, conf_matrix_f_gt == 0
        pos_w, neg_w = self.pos_w, self.neg_w

        if not pos_mask.any():
            pos_mask[0, 0, 0] = True
            pos_w = 0.0
        if not neg_mask.any():
            neg_mask[0, 0, 0] = True
            neg_w = 0.0

        conf_matrix_fine = torch.clamp(conf_matrix_fine, 1e-6, 1 - 1e-6)
        alpha = self.loss_config["focal_alpha"]
        gamma = self.loss_config["focal_gamma"]

        loss_pos = (
            -alpha * torch.pow(1 - conf_matrix_fine[pos_mask], gamma)
            * conf_matrix_fine[pos_mask].log()
        )
        loss_neg = (
            -alpha * torch.pow(conf_matrix_fine[neg_mask], gamma)
            * (1 - conf_matrix_fine[neg_mask]).log()
        )

        return pos_w * loss_pos.mean() + neg_w * loss_neg.mean()

    def _symmetric_epipolar_distance(self, pts0, pts1, E, K0, K1):
        """Squared symmetric epipolar distance for (N, 2) points and (N, 3, 3) matrices."""
        pts0 = (pts0 - K0[:, [0, 1], [2, 2]]) / K0[:, [0, 1], [0, 1]]
        pts1 = (pts1 - K1[:, [0, 1], [2, 2]]) / K1[:, [0, 1], [0, 1]]
        pts0 = convert_points_to_homogeneous(pts0)
        pts1 = convert_points_to_homogeneous(pts1)

        Ep0 = (pts0[:, None, :] @ E.transpose(-2, -1)).squeeze(1)
        p1Ep0 = torch.sum(pts1 * Ep0, -1)
        Etp1 = (pts1[:, None, :] @ E).squeeze(1)

        return p1Ep0**2 * (
            1.0 / (Ep0[:, 0]**2 + Ep0[:, 1]**2 + 1e-9)
            + 1.0 / (Etp1[:, 0]**2 + Etp1[:, 1]**2 + 1e-9)
        )

    def compute_sub_pixel_loss(self, data):
        """Epipolar refinement loss, or None when no matches are available."""
        Tx = numeric.cross_product_matrix(data["T_0to1"][:, :3, 3])
        E_mat = Tx @ data["T_0to1"][:, :3, :3]

        m_bids = data["m_bids"]
        pts0 = data["mkpts0_f_train"]
        pts1 = data["mkpts1_f_train"]
        sym_dist = self._symmetric_epipolar_distance(
            pts0, pts1, E_mat[m_bids], data["K0"][m_bids], data["K1"][m_bids]
        )
        valid_dist = sym_dist[sym_dist < 1e-4]

        if len(sym_dist) == 0:
            return None
        elif len(sym_dist) == 1 or len(valid_dist) == 0:
            loss = sym_dist * 1e-9
        else:
            loss = valid_dist

        return loss.mean()

    def compute_coarse_loss(self, data, weight=None):
        """Bidirectional focal loss evaluated at ground-truth coarse matches."""
        conf_gt = data["conf_matrix_gt"]
        sim_matrix = data["sim_matrix"]
        alpha = self.loss_config["focal_alpha"]
        gamma = self.loss_config["focal_gamma"]
        pos_weight = self.pos_w

        batch_ids, row_ids, col_ids = torch.nonzero(conf_gt, as_tuple=True)
        if batch_ids.numel() == 0:
            batch_ids = row_ids = col_ids = torch.tensor(
                [0], device=conf_gt.device, dtype=torch.long
            )
            if weight is not None:
                weight[0, 0, 0] = 0.0
            pos_weight = 0.0

        _, num_tokens, _ = sim_matrix.shape
        row_keys, row_inverse = torch.unique(
            batch_ids * num_tokens + row_ids, return_inverse=True
        )
        row_denominator = torch.logsumexp(
            sim_matrix[row_keys // num_tokens, row_keys % num_tokens], dim=1
        )[row_inverse]

        col_keys, col_inverse = torch.unique(
            batch_ids * num_tokens + col_ids, return_inverse=True
        )
        col_denominator = torch.logsumexp(
            sim_matrix[col_keys // num_tokens, :, col_keys % num_tokens], dim=1
        )[col_inverse]

        row_probability = torch.clamp(
            torch.exp(sim_matrix[batch_ids, row_ids, col_ids] - row_denominator),
            1e-6,
            1 - 1e-6,
        )
        col_probability = torch.clamp(
            torch.exp(sim_matrix[batch_ids, row_ids, col_ids] - col_denominator),
            1e-6,
            1 - 1e-6,
        )
        positive_loss = (
            -alpha * (1 - row_probability).pow(gamma) * row_probability.log()
            -alpha * (1 - col_probability).pow(gamma) * col_probability.log()
        ).mean()

        if weight is not None:
            positive_loss = positive_loss * weight[batch_ids, row_ids, col_ids]

        return pos_weight * positive_loss.mean()

    @torch.no_grad()
    def compute_c_weight(self, data):
        """Mask padded token pairs in the coarse loss."""
        if "mask0" not in data:
            return None
        return (
            data["mask0"].flatten(-2)[..., None]
            * data["mask1"].flatten(-2)[:, None]
        ).float()

    def forward(self, data):
        """Write the scalar loss and detached logging values into data."""
        loss_scalars = {}
        c_weight = self.compute_c_weight(data)

        loss_c = self.compute_coarse_loss(data, weight=c_weight)
        loss_c *= self.loss_config["coarse_weight"]
        loss = loss_c
        loss_scalars["loss_c"] = loss_c.detach()

        loss_f_match = self.compute_fine_matching_loss(data)
        loss_f_match *= self.loss_config["fine_weight"]
        loss = loss + loss_f_match
        loss_scalars["loss_f"] = loss_f_match.detach()

        loss_sub = self.compute_sub_pixel_loss(data)
        if loss_sub is not None:
            loss_sub = loss_sub * self.loss_config["sub_weight"]
            loss = loss + loss_sub
            loss_scalars["loss_sub"] = loss_sub.detach()
        else:
            assert self.training is False
            loss_scalars["loss_sub"] = torch.tensor(1.0)

        loss_scalars["loss"] = loss.detach()
        data.update({"loss": loss, "loss_scalars": loss_scalars})
