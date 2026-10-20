"""Window normalization, geometry features, and attention."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.woven_anchor.utils.utils import RMSNormDim


class LocalWindowNorm(nn.Module):
    """Normalize each channel over valid positions within each window.

    x:    [B, C, H_u, W_u]
    mask: [B, 1, H_u, W_u] or None
    """
    def __init__(self, window_size: int, eps: float = 1e-5):
        super().__init__()
        self.window_size = window_size
        self.eps = eps

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        B, C, H, W = x.shape
        w = self.window_size

        if mask is not None:
            x_masked = x * mask
        else:
            x_masked = x

        mu = F.avg_pool2d(x_masked, kernel_size=w, stride=w, count_include_pad=False)

        if mask is not None:
            # Exclude masked cells from the window means.
            valid_fraction = F.avg_pool2d(mask, kernel_size=w, stride=w, count_include_pad=False).clamp_min(1e-6)
            mu = mu / valid_fraction
            mu2 = F.avg_pool2d(x.square() * mask, kernel_size=w, stride=w, count_include_pad=False) / valid_fraction
        else:
            mu2 = F.avg_pool2d(x.square(), kernel_size=w, stride=w, count_include_pad=False)

        var = (mu2 - mu.square()).clamp_min(0.0)
        inv_sigma = torch.rsqrt(var + self.eps)

        mu_up = mu.repeat_interleave(w, dim=2).repeat_interleave(w, dim=3)[:, :, :H, :W]
        inv_sigma_up = inv_sigma.repeat_interleave(w, dim=2).repeat_interleave(w, dim=3)[:, :, :H, :W]

        x_norm = (x - mu_up) * inv_sigma_up
        if mask is not None:
            x_norm = x_norm * mask
        return x_norm


class CNNGeoEncoder(nn.Module):
    """Encode assignment probabilities and local coordinates with a CNN."""

    def __init__(self, dim_geo: int, window_size: int):
        super().__init__()
        self.window_size = window_size

        ys, xs = torch.meshgrid(
            torch.linspace(-1.0, 1.0, window_size),
            torch.linspace(-1.0, 1.0, window_size),
            indexing="ij",
        )
        pos = torch.stack([xs, ys], dim=0)   # [2, Hp, Wp]
        self.register_buffer("pos", pos)

        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(32, dim_geo, kernel_size=1),
        )

    def forward(self, assign_probs_patch: torch.Tensor) -> torch.Tensor:
        """Map [B, K, H_a, W_a, Hp, Wp] assignments to [B, K, H_a, W_a, dim_geo]."""
        B, K, H_a, W_a, Hp, Wp = assign_probs_patch.shape
        N = B * K * H_a * W_a

        x = assign_probs_patch.reshape(N, 1, Hp, Wp)

        pos = self.pos.unsqueeze(0).expand(N, -1, -1, -1)

        x = torch.cat([x, pos], dim=1)  # [N, 3, Hp, Wp]: assignment, x, y.

        out = self.net(x)

        out = out.squeeze(-1).squeeze(-1).view(B, K, H_a, W_a, -1)
        return out


class StatsGeoEncoder(nn.Module):
    """Encode assignment center, covariance, and log mass into geometry features."""

    def __init__(self, dim_geo: int, window_size: int):
        super().__init__()
        ys, xs = torch.meshgrid(
            torch.linspace(-1.0, 1.0, window_size),
            torch.linspace(-1.0, 1.0, window_size),
            indexing="ij",
        )
        self.register_buffer("pos", torch.stack([xs, ys], dim=-1))  # Local (x, y) in [-1, 1].
        self.proj = nn.Linear(6, dim_geo)

    def forward(self, assign_probs_patch: torch.Tensor) -> torch.Tensor:
        """Map [B, K, H_a, W_a, Hp, Wp] assignments to [B, K, H_a, W_a, dim_geo]."""
        B, K, H_a, W_a, Hp, Wp = assign_probs_patch.shape
        Lp = Hp * Wp

        assign_probs = assign_probs_patch.reshape(B, K, H_a, W_a, Lp, 1)
        pos = self.pos.reshape(1, 1, 1, 1, Lp, 2)

        # Stabilize normalization and log mass for empty anchors.
        mass_eps = assign_probs.sum(dim=4, keepdim=True) + 1e-6
        center = (assign_probs * pos).sum(dim=4, keepdim=True) / mass_eps
        centered_coords = pos - center
        cov_diag = (assign_probs * centered_coords.square()).sum(dim=4, keepdim=True) / mass_eps
        cov_xy = (assign_probs * (centered_coords[..., 0:1] * centered_coords[..., 1:2])).sum(dim=4, keepdim=True) / mass_eps
        log_mass = torch.log(mass_eps.squeeze(4).squeeze(-1)).unsqueeze(-1)

        # Descriptor order: [center_x, center_y, cov_xx, cov_yy, cov_xy, log_mass].
        geometry_stats = torch.cat(
            [
                center.squeeze(4),
                cov_diag.squeeze(4),
                cov_xy.squeeze(4),
                log_mass,
            ],
            dim=-1,
        )  # [B,K,H_a,W_a,6]

        return self.proj(geometry_stats)


class Attention(nn.Module):
    """Self- or bidirectional cross-attention over [B, L, D] sequences.

    ``forward(x0, x1)`` exchanges context between two sequences. With x1=None,
    x0 contains two equally sized batches concatenated along the batch axis.
    ``forward_self(x)`` attends independently within each sequence.
    Masks have shape [B, L], with True marking valid tokens.
    """

    def __init__(self, dim: int, n_heads: int = 4, bias: bool = False):
        super().__init__()
        assert dim % n_heads == 0
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.norm = RMSNormDim(dim, target_dim=-1)
        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=bias)
        self.o_proj = nn.Linear(dim, dim, bias=bias)

    def _project_qkv(self, x):
        B, L, _ = x.shape
        x = self.norm(x)
        qkv = self.qkv_proj(x).view(B, L, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, L, head_dim]
        return qkv[0], qkv[1], qkv[2]

    def _apply_attention(self, q, k, v, q_mask=None, kv_mask=None):
        B, _, Lq, _ = q.shape
        attn_mask = None
        if kv_mask is not None and not kv_mask.all():
            attn_mask = kv_mask.reshape(B, 1, 1, -1)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(B, Lq, self.dim)
        out = self.o_proj(out)
        if q_mask is not None:
            out = out * q_mask.unsqueeze(-1).to(out.dtype)
        return out

    def forward(self, x0, x1=None, m0=None, m1=None):
        """Return cross-attention contexts as a pair or a concatenated batch."""
        if x1 is None:
            assert x0.shape[0] % 2 == 0, "Concatenated cross-attention expects an even batch."
            q_all, k_all, v_all = self._project_qkv(x0)
            q0, q1 = torch.chunk(q_all, 2, dim=0)
            k0, k1 = torch.chunk(k_all, 2, dim=0)
            v0, v1 = torch.chunk(v_all, 2, dim=0)
            qm0 = qm1 = None
            if m0 is not None:
                qm0, qm1 = torch.chunk(m0, 2, dim=0)
            ctx0 = self._apply_attention(q0, k1, v1, q_mask=qm0, kv_mask=qm1)
            ctx1 = self._apply_attention(q1, k0, v0, q_mask=qm1, kv_mask=qm0)
            return torch.cat([ctx0, ctx1], dim=0)

        q0, k0, v0 = self._project_qkv(x0)
        q1, k1, v1 = self._project_qkv(x1)
        ctx0 = self._apply_attention(q0, k1, v1, q_mask=m0, kv_mask=m1)
        ctx1 = self._apply_attention(q1, k0, v0, q_mask=m1, kv_mask=m0)
        return ctx0, ctx1

    def forward_self(self, x, mask=None):
        """Return self-attention context for each sequence independently."""
        q, k, v = self._project_qkv(x)
        return self._apply_attention(q, k, v, q_mask=mask, kv_mask=mask)
