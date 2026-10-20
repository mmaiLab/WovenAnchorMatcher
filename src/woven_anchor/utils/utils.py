import torch
import torch.nn as nn
from mamba_ssm.ops.triton.layer_norm import rms_norm_fn


class LayerNormDim(nn.Module):
    def __init__(self, channels: int, target_dim: int, eps: float = 1e-5):
        super().__init__()
        self.channels = channels
        self.target_dim = target_dim
        self.ln = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ndim = x.ndim
        target_dim = self.target_dim if self.target_dim >= 0 else ndim + self.target_dim

        if not (0 <= target_dim < ndim):
            raise ValueError(f"Invalid target_dim={self.target_dim} for x.ndim={ndim}")

        if x.shape[target_dim] != self.channels:
            raise ValueError(
                f"Expected size {self.channels} at dim {target_dim}, "
                f"but got x.shape={tuple(x.shape)}"
            )

        if target_dim == ndim - 1:
            return self.ln(x)

        perm = list(range(ndim))
        perm[target_dim], perm[-1] = perm[-1], perm[target_dim]
        x = x.permute(*perm)
        x = self.ln(x)

        inv_perm = [0] * ndim
        for i, p in enumerate(perm):
            inv_perm[p] = i
        return x.permute(*inv_perm)


class RMSNormDim(nn.Module):
    def __init__(self, channels: int, target_dim: int, eps: float = 1e-5):
        super().__init__()
        self.channels = channels
        self.target_dim = target_dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ndim = x.ndim
        target_dim = self.target_dim if self.target_dim >= 0 else ndim + self.target_dim

        if not (0 <= target_dim < ndim):
            raise ValueError(f"Invalid target_dim={self.target_dim} for x.ndim={ndim}")

        if x.shape[target_dim] != self.channels:
            raise ValueError(
                f"Expected size {self.channels} at dim {target_dim}, "
                f"but got x.shape={tuple(x.shape)}"
            )

        if target_dim == ndim - 1:
            return rms_norm_fn(x, self.weight, None, eps=self.eps)

        perm = list(range(ndim))
        perm[target_dim], perm[-1] = perm[-1], perm[target_dim]
        x = x.permute(*perm)
        x = rms_norm_fn(x, self.weight, None, eps=self.eps)

        inv_perm = [0] * ndim
        for i, p in enumerate(perm):
            inv_perm[p] = i
        return x.permute(*inv_perm)


def MLP2d(channels: list, do_bn=True):
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv2d(channels[i - 1], channels[i], kernel_size=1, bias=True)
        )
        if i < (n - 1):
            if do_bn:
                layers.append(LayerNormDim(channels[i], 1))
            layers.append(nn.GELU())
    return nn.Sequential(*layers)


@torch.amp.custom_fwd(device_type='cuda', cast_inputs=torch.float32)
def normalize_keypoints(
    kpts: torch.Tensor,
    size: torch.Tensor,
) -> torch.Tensor:
    """
    Args:
        kpts:
            [..., 2]
        size:
            [..., 2] or [2]

    Returns:
        with shape of kpts
    """
    if not isinstance(size, torch.Tensor):
        size = torch.tensor(size, device=kpts.device, dtype=kpts.dtype)

    size = size.to(device=kpts.device, dtype=kpts.dtype)
    if size.dim() == 1:
        # [2] -> [1, 1, 1, 2]
        shift = (size / 2).view(1, 1, 1, 2)
        scale = (size.max() / 2).view(1, 1, 1, 1)
    else:
        # [B, 2] -> [B, 1, 1, 2]
        shift = (size / 2).view(size.shape[0], 1, 1, 2)
        scale = (size.max(dim=-1).values / 2).view(size.shape[0], 1, 1, 1)

    kpts = (kpts - shift) / scale
    return kpts


class KeypointEncoder2d(nn.Module):
    def __init__(self, layers, feature_dim, do_bn=True):
        super().__init__()
        dims = [2] + layers + [feature_dim]
        self.encoder = MLP2d(dims, do_bn=do_bn)
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, grid, size):
        # kpts: [B or 1, H, W, 2]
        # out: [B or 1, C, H, W]
        kpts = normalize_keypoints(grid, size).permute(0,3,1,2)
        return self.encoder(kpts)


class GLU_3(nn.Module):
    def __init__(self, dim, mid_dim):
        super(GLU_3, self).__init__()
        self.W = nn.Conv2d(dim, mid_dim, kernel_size=3, padding=1, bias=False)
        self.V = nn.Conv2d(dim, mid_dim, kernel_size=3, padding=1, bias=False)
        self.W2 = nn.Conv2d(mid_dim, dim, kernel_size=3, padding=1, bias=False)
        self.act = nn.ReLU(True)
        self.act = nn.GELU()

    def forward(self, feat):
        feat_act = self.act(self.W(feat))
        feat_linear = self.V(feat)
        feat = feat_act * feat_linear
        feat = self.W2(feat)
        return feat


def complete_pair_masks(x0, x1, mask0=None, mask1=None):
    """Fill a missing [B, H, W] image mask with valid positions; keep both None."""
    if mask0 is None and mask1 is not None:
        mask0 = torch.ones_like(x0[:, 0], dtype=torch.bool)
    elif mask1 is None and mask0 is not None:
        mask1 = torch.ones_like(x1[:, 0], dtype=torch.bool)
    return mask0, mask1


def valid_hw_from_padding_mask(p_m: torch.Tensor):
    """Return valid height and width for [B, H, W] masks with bottom/right padding."""
    h = p_m.sum(1).max(-1).values
    w = p_m.sum(-1).max(-1).values
    return h, w

def prune_matches(b_ids, i_ids, j_ids, mconf, data, border_rm: int):
    """Remove padded matches and optionally an extra border of coarse-grid cells."""
    if border_rm == 0 and 'mask0' not in data:
        return b_ids, i_ids, j_ids, mconf
    b_ids_ = b_ids.to(torch.long)
    i_ids_ = i_ids.to(torch.long)
    j_ids_ = j_ids.to(torch.long)
    h0c, w0c = data['hw0_c']
    h1c, w1c = data['hw1_c']

    x0, y0 = i_ids_ % w0c, i_ids_ // w0c
    x1, y1 = j_ids_ % w1c, j_ids_ // w1c

    if 'mask0' in data:
        # Padding remains invalid even when border removal is disabled.
        keep = data['mask0'][b_ids_, y0, x0] & data['mask1'][b_ids_, y1, x1]

        if border_rm > 0:
            # Measure the border from each image's valid extent.
            h0s, w0s = valid_hw_from_padding_mask(data['mask0'])
            h1s, w1s = valid_hw_from_padding_mask(data['mask1'])
            h0, w0 = h0s[b_ids_], w0s[b_ids_]
            h1, w1 = h1s[b_ids_], w1s[b_ids_]

            keep &= (x0 >= border_rm) & (x0 < (w0 - border_rm)) & \
                    (y0 >= border_rm) & (y0 < (h0 - border_rm)) & \
                    (x1 >= border_rm) & (x1 < (w1 - border_rm)) & \
                    (y1 >= border_rm) & (y1 < (h1 - border_rm))
    else:
        # Without padding masks, use the full feature-grid extent.
        if border_rm > 0:
            keep = (x0 >= border_rm) & (x0 < w0c - border_rm) & \
                   (y0 >= border_rm) & (y0 < h0c - border_rm) & \
                   (x1 >= border_rm) & (x1 < w1c - border_rm) & \
                   (y1 >= border_rm) & (y1 < h1c - border_rm)
        else:
            return b_ids, i_ids, j_ids, mconf
            
    return b_ids[keep], i_ids[keep], j_ids[keep], mconf[keep]
