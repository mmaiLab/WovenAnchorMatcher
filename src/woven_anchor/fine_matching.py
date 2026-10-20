import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from loguru import logger
from src.woven_anchor.refiner import MLPRefiner
INF = 1e9


def generate_random_mask(n, num_true, device=None):
    mask = torch.zeros(n, dtype=torch.bool, device=device)
    indices = torch.randperm(n, device=device)[:num_true]
    mask[indices] = True
    return mask

class FineSubMatching(nn.Module):
    """Fine matching with learned subpixel refinement."""

    def __init__(self, config, profiler):
        super().__init__()
        self.temperature = config['fine']['dsmax_temperature']
        self.W_f = config['fine_window_size']
        self.d_model = config['fine']['d_model']
        self.fine_thr = config['fine']['thr']
        self.in_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.refiner = MLPRefiner(self.d_model)
        self.fine_spv_max = 500  # Limit confidence-matrix supervision to save memory.
        self.profiler = profiler

    def forward(self, data):
        """Match paired windows with features [M, W_f**2, C] per image."""
        feat_f0_in, feat_f1_in = data['feat_f0'], data['feat_f1']
        M, WW, C = feat_f0_in.shape
        W_f = self.W_f

        # Training supplies ground-truth matches, so only evaluation can be empty.
        if M == 0:
            assert self.training == False, "M is always >0, when training, see coarse_matching.py"
            logger.warning('No matches found in coarse-level.')
            if data["mode"] == "test":
                data.update({
                    'mkpts0_f': data['mkpts0_c'],
                    'mkpts1_f': data['mkpts1_c'],
                    'mconf_c': torch.zeros(0, device=feat_f0_in.device),
                    'mconf_f': torch.zeros(0, device=feat_f0_in.device),
                })
            else:
                data.update({
                    'mkpts0_f': data['mkpts0_c'],
                    'mkpts1_f': data['mkpts1_c'],
                    'mconf_f': torch.zeros(0, device=feat_f0_in.device),
                    'mkpts0_f_train': data['mkpts0_c_train'],
                    'mkpts1_f_train': data['mkpts1_c_train'],
                    'conf_matrix_fine': torch.zeros(1, W_f * W_f, W_f * W_f, device=feat_f0_in.device),
                    'b_ids_fine': torch.zeros(0, device=feat_f0_in.device),
                    'i_ids_fine': torch.zeros(0, device=feat_f0_in.device),
                    'j_ids_fine': torch.zeros(0, device=feat_f0_in.device),
                })
            return

        feat_f = torch.cat([feat_f0_in, feat_f1_in], dim=0)
        feat_f = self.in_proj(feat_f) * (1.0 / math.sqrt(self.d_model))
        feat_f0, feat_f1 = torch.chunk(feat_f, 2, dim=0)
        feat_f0 = feat_f0 * (1.0 / self.temperature)
        sim_matrix = torch.bmm(feat_f0, feat_f1.transpose(1, 2))
        conf_matrix_fine = F.softmax(sim_matrix, 1) * F.softmax(sim_matrix, 2)
        
        sub_pixel_matches = self.get_fine_sub_match(conf_matrix_fine, feat_f0_in, feat_f1_in, data)
        data.update(**sub_pixel_matches)
        
    def get_fine_sub_match(self, conf_matrix_fine, feat_f0_in, feat_f1_in, data):
        """Select and refine window matches in original image coordinates."""
        with torch.no_grad():
            W_f = self.W_f
            mask = conf_matrix_fine > self.fine_thr
            if mask.sum() == 0:
                # Keep one fallback match when all candidates fail the threshold.
                mask[0, 0, 0] = 1
                conf_matrix_fine[0, 0, 0] = 1
            # Keep window-pair maxima, taking the first column for row ties.
            mask = mask * (conf_matrix_fine == conf_matrix_fine.amax(dim=[1, 2], keepdim=True))
            mask_v, all_j_ids = mask.max(dim=2)
            fine_ids, i_ids = torch.where(mask_v)
            j_ids = all_j_ids[fine_ids, i_ids]
            mconf = conf_matrix_fine[fine_ids, i_ids, j_ids]
            b_ids_c, i_ids_c, j_ids_c = data['b_ids'], data['i_ids'], data['j_ids']

            # Map coarse centers to the 1/2-resolution feature grid.
            scale_f_c = data['hw0_f'][0] // data['hw0_c'][0]

            mkpts0_c_scaled_to_f = torch.stack(
                [i_ids_c % data['hw0_c'][1], torch.div(i_ids_c, data['hw0_c'][1], rounding_mode='trunc')],
                dim=1) * scale_f_c

            mkpts1_c_scaled_to_f = torch.stack(
                [j_ids_c % data['hw1_c'][1], torch.div(j_ids_c, data['hw1_c'][1], rounding_mode='trunc')],
                dim=1) * scale_f_c

            updated_b_ids = b_ids_c[fine_ids]

            # Convert fine-grid coordinates to original image pixels.
            scale = data['hw0_i'][0] / data['hw0_f'][0]
            scale0 = scale * data['scale0'][updated_b_ids] if 'scale0' in data else scale
            scale1 = scale * data['scale1'][updated_b_ids] if 'scale1' in data else scale

            # Local (x, y) coordinates within each window.
            mkpts0_f_window = torch.stack(
                [i_ids % W_f, torch.div(i_ids, W_f, rounding_mode='trunc')],
                dim=1)

            mkpts1_f_window = torch.stack(
                [j_ids % W_f, torch.div(j_ids, W_f, rounding_mode='trunc')],
                dim=1)

        sub_ref0, sub_ref1 = self.refiner(feat_f0_in, feat_f1_in, fine_ids, i_ids, j_ids)
        pad = 0 if W_f % 2 == 0 else W_f // 2
        # Combine window coordinates, coarse centers, and learned subpixel offsets.
        mkpts0_f1 = (mkpts0_f_window + mkpts0_c_scaled_to_f[fine_ids] - pad) * scale0
        mkpts1_f1 = (mkpts1_f_window + mkpts1_c_scaled_to_f[fine_ids] - pad) * scale1
        mkpts0_f_train = mkpts0_f1 + sub_ref0 * scale0
        mkpts1_f_train = mkpts1_f1 + sub_ref1 * scale1
        # Keep gradients for supervision; detach the reported coordinates.
        mkpts0_f = mkpts0_f_train.clone().detach()
        mkpts1_f = mkpts1_f_train.clone().detach()

        valid = mconf != 0
        sub_pixel_matches = {
            'm_bids': b_ids_c[fine_ids[valid]],
            'mkpts0_f1': mkpts0_f1[valid],
            'mkpts1_f1': mkpts1_f1[valid],
            'mkpts0_f': mkpts0_f[valid],
            'mkpts1_f': mkpts1_f[valid],
            'mconf_f': mconf[valid],
        }

        if data['mode'] == "test":
            mconf_c = data["mconf"][fine_ids]
            sub_pixel_matches.update({
                'mconf_c': mconf_c[valid],
            })
        else:
            # Optionally subsample confidence supervision, keeping all refined coordinates.
            if self.fine_spv_max is None or self.fine_spv_max > len(data['b_ids']):
                sub_pixel_matches.update({
                    'mkpts0_f_train': mkpts0_f_train,
                    'mkpts1_f_train': mkpts1_f_train,
                    'b_ids_fine': data['b_ids'],
                    'i_ids_fine': data['i_ids'],
                    'j_ids_fine': data['j_ids'],
                    'conf_matrix_fine': conf_matrix_fine
                })
            else:
                train_mask = generate_random_mask(
                    len(data['b_ids']),
                    self.fine_spv_max,
                    device=data['b_ids'].device
                )
                sub_pixel_matches.update({
                    'mkpts0_f_train': mkpts0_f_train,
                    'mkpts1_f_train': mkpts1_f_train,
                    'b_ids_fine': data['b_ids'][train_mask],
                    'i_ids_fine': data['i_ids'][train_mask],
                    'j_ids_fine': data['j_ids'][train_mask],
                    'conf_matrix_fine': conf_matrix_fine[train_mask]
                })

        return sub_pixel_matches
