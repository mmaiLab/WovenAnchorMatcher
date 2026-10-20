import torch
from torch import nn
import math
from einops import rearrange
from src.woven_anchor.extract_matches import extract_coarse_matches
INF = 1e9


def compute_max_candidates(p_m0, p_m1):
    """Compute the max candidates of all pairs within a batch

    Args:
        p_m0, p_m1 (torch.Tensor): padded masks
    """
    h0s, w0s = p_m0.sum(1).max(-1)[0], p_m0.sum(-1).max(-1)[0]
    h1s, w1s = p_m1.sum(1).max(-1)[0], p_m1.sum(-1).max(-1)[0]
    max_cand = torch.sum(
        torch.min(torch.stack([h0s * w0s, h1s * w1s], -1), -1)[0])
    return max_cand


class CoarseMatching(nn.Module):
    def __init__(self, config, profiler):
        super().__init__()
        self.config = config
        self.d_model = config['d_model']
        self.thr = config['thr']
        self.border_rm = config['border_rm']
        self.temperature = config['dsmax_temperature']
        self.profiler = profiler

        self.in_proj = nn.Linear(self.d_model, self.d_model, bias=True)

    def forward(self, data):
        feat_c0 = rearrange(data['feat_8_0'], "b c h w -> b (h w) c")
        feat_c1 = rearrange(data['feat_8_1'], "b c h w -> b (h w) c")
        if data.get('mask0', None) is not None:
            mask_c0, mask_c1 = data['mask0'].flatten(-2), data['mask1'].flatten(-2)
        else:
            mask_c0 = mask_c1 = None

        if feat_c0.shape == feat_c1.shape:
            feat_c = torch.cat([feat_c0, feat_c1], dim=0)
            feat_c = self.in_proj(feat_c) * (1.0 / math.sqrt(self.d_model))
            feat_c0, feat_c1 = torch.chunk(feat_c, 2, dim=0)
        else:
            feat_c0 = self.in_proj(feat_c0) * (1.0 / math.sqrt(self.d_model))
            feat_c1 = self.in_proj(feat_c1) * (1.0 / math.sqrt(self.d_model))
        feat_c0 = feat_c0 * (1.0 / self.temperature)
        sim_matrix = torch.bmm(feat_c0, feat_c1.transpose(1, 2))
        
        if mask_c0 is not None:
            sim_matrix.masked_fill_(
                ~(mask_c0[..., None] * mask_c1[:, None]).bool(),
                -INF)

        if data['mode'] == "test":
            data.update(**self.get_coarse_match_inference(sim_matrix, data))
        else:
            data.update({'sim_matrix': sim_matrix})
            data.update(**self.get_coarse_match_training(sim_matrix, data))

    @torch.no_grad()
    def get_coarse_match_training(self, sim_matrix, data):
        _device = sim_matrix.device
        b_ids, i_ids, j_ids, mconf = extract_coarse_matches(sim_matrix, data, self.thr, self.border_rm)

        if self.training:
            if 'mask0' not in data:
                # Use the feature-grid capacity even when no matches are predicted.
                batch_size, num_tokens0, num_tokens1 = sim_matrix.shape
                num_candidates_max = batch_size * max(num_tokens0, num_tokens1)
            else:
                num_candidates_max = compute_max_candidates(data['mask0'], data['mask1'])
            num_matches_train = int(num_candidates_max * self.config['train_coarse_percent'])
            num_matches_pred = len(b_ids)
            train_pad_num_gt_min = self.config['train_pad_num_gt_min']
            assert train_pad_num_gt_min < num_matches_train, "min-num-gt-pad should be less than num-train-matches"

            # pred_indices is to select from prediction
            if num_matches_pred <= num_matches_train - train_pad_num_gt_min:
                pred_indices = torch.arange(num_matches_pred, device=_device)
            else:
                pred_indices = torch.randint(
                    num_matches_pred,
                    (num_matches_train - train_pad_num_gt_min,),
                    device=_device)

            # gt_pad_indices is to select from gt padding. e.g. max(3787-4800, 200)
            gt_pad_indices = torch.randint(
                len(data['spv_b_ids']),
                (max(num_matches_train - num_matches_pred,
                     train_pad_num_gt_min),),
                device=_device)
            mconf_gt = torch.zeros(len(data['spv_b_ids']), device=_device)  # set conf of gt paddings to all zero

            b_ids, i_ids, j_ids, mconf = map(
                lambda x, y: torch.cat([x[pred_indices], y[gt_pad_indices]],
                                       dim=0),
                *zip([b_ids, data['spv_b_ids']], [i_ids, data['spv_i_ids']],
                     [j_ids, data['spv_j_ids']], [mconf, mconf_gt]))

        # these matches are selected patches that feed into fine-level network
        coarse_matches = {'b_ids': b_ids, 'i_ids': i_ids, 'j_ids': j_ids}

        # update with matches in original image resolution
        scale = data['hw0_i'][0] / data['hw0_c'][0]
        scale0 = scale * data['scale0'][b_ids] if 'scale0' in data else scale
        scale1 = scale * data['scale1'][b_ids] if 'scale1' in data else scale
        mkpts0_c = torch.stack(
            [i_ids % data['hw0_c'][1], torch.div(i_ids, data['hw0_c'][1], rounding_mode='trunc')],
            dim=1) * scale0
        mkpts1_c = torch.stack(
            [j_ids % data['hw1_c'][1], torch.div(j_ids, data['hw1_c'][1], rounding_mode='trunc')],
            dim=1) * scale1

        # these matches is the current prediction (for visualization)
        coarse_matches.update({
            'gt_mask': mconf == 0,
            'm_bids': b_ids[mconf != 0],  # mconf == 0 => gt matches
            'mkpts0_c': mkpts0_c[mconf != 0],
            'mkpts1_c': mkpts1_c[mconf != 0],
            'mkpts0_c_train': mkpts0_c,
            'mkpts1_c_train': mkpts1_c,
            'mconf': mconf[mconf != 0]
        })

        return coarse_matches

    @torch.no_grad()
    def get_coarse_match_inference(self, sim_matrix, data):
        b_ids, i_ids, j_ids, mconf = extract_coarse_matches(sim_matrix, data, self.thr, self.border_rm)

        # these matches are selected patches that feed into fine-level network
        coarse_matches = {'b_ids': b_ids, 'i_ids': i_ids, 'j_ids': j_ids}

        # update with matches in original image resolution
        scale = data['hw0_i'][0] / data['hw0_c'][0]
        scale0 = scale * data['scale0'][b_ids] if 'scale0' in data else scale
        scale1 = scale * data['scale1'][b_ids] if 'scale1' in data else scale
        mkpts0_c = torch.stack(
            [i_ids % data['hw0_c'][1], torch.div(i_ids, data['hw0_c'][1], rounding_mode='trunc')],
            dim=1) * scale0
        mkpts1_c = torch.stack(
            [j_ids % data['hw1_c'][1], torch.div(j_ids, data['hw1_c'][1], rounding_mode='trunc')],
            dim=1) * scale1

        # these matches are the current coarse level predictions
        coarse_matches.update({
            'mconf': mconf,
            'm_bids': b_ids,  # mconf == 0 => gt matches
            'mkpts0_c': mkpts0_c,
            'mkpts1_c': mkpts1_c,
        })
        return coarse_matches
