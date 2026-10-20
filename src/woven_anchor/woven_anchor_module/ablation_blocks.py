# Initialization adapted from Mamba; copyright (c) 2023 Albert Gu, Tri Dao.
# Modified for WovenAnchor. Apache-2.0 terms: LICENSES/Apache-2.0.txt.
# See THIRD_PARTY_NOTICES.md for the scope of the adaptation.

"""Directional Mamba and dense attention baselines used in the ablations."""

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mamba_ssm import Mamba

from src.woven_anchor.utils.utils import RMSNormDim


# -----------------------------------------------------------------------------
# Mamba utilities
# -----------------------------------------------------------------------------

class _MambaBlock(nn.Module):
    def __init__(self, dim, layer_idx, device=None, dtype=None):
        super().__init__()
        self.mixer = Mamba(dim, layer_idx=layer_idx, device=device, dtype=dtype)

    def forward(self, x):
        return self.mixer(x)


def _init_mamba_weights(module, n_directions):
    """Initialize biases and scale each scan's output projection as in training."""
    if isinstance(module, nn.Linear) and module.bias is not None:
        # Preserve Mamba's specially initialized time-step bias.
        if not getattr(module.bias, "_no_reinit", False):
            nn.init.zeros_(module.bias)

    for name, weight in module.named_parameters():
        if name == "out_proj.weight":
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
            with torch.no_grad():
                weight /= math.sqrt(n_directions)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _apply_mask(x, mask=None):
    if mask is None:
        return x
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    return x * mask.to(dtype=x.dtype)


# -----------------------------------------------------------------------------
# Directional raster-scan Mamba blocks
# -----------------------------------------------------------------------------

class DirectionalMamba(nn.Module):
    """Average independently learned scans, then add a within-image residual.

    ``>`` / ``<`` are row-major / reverse row-major scans;
    ``v`` / ``^`` are column-major / reverse column-major scans.
    """

    def __init__(self, d_model: int, directions, device=None, dtype=None):
        super().__init__()
        self.directions = directions
        self.norm = RMSNormDim(d_model, target_dim=1)
        self.layers = nn.ModuleList([
            _MambaBlock(d_model, layer_idx=i, device=device, dtype=dtype)
            for i in range(len(self.directions))
        ])
        self.apply(partial(_init_mamba_weights, n_directions=len(self.directions)))

    def _run_direction(self, x_bhwc, direction, layer):
        h = x_bhwc.shape[1]

        if direction == ">":
            seq = layer(rearrange(x_bhwc, "b h w c -> b (h w) c", h=h))
            return rearrange(seq, "b (h w) c -> b h w c", h=h)

        if direction == "<":
            seq = layer(rearrange(x_bhwc, "b h w c -> b (h w) c", h=h).flip(dims=[1]))
            return rearrange(seq.flip(dims=[1]), "b (h w) c -> b h w c", h=h)

        if direction == "v":
            seq = layer(rearrange(x_bhwc, "b h w c -> b (w h) c", h=h))
            return rearrange(seq, "b (w h) c -> b h w c", h=h)

        if direction == "^":
            seq = layer(rearrange(x_bhwc, "b h w c -> b (w h) c", h=h).flip(dims=[1]))
            return rearrange(seq.flip(dims=[1]), "b (w h) c -> b h w c", h=h)

        raise ValueError(f"Unknown direction: {direction}")

    def _scan(self, x_bhwc):
        """Combine independent directional responses using their arithmetic mean."""
        out = torch.zeros_like(x_bhwc)
        for direction, layer in zip(self.directions, self.layers):
            out += self._run_direction(x_bhwc, direction, layer)
        out /= len(self.directions)
        return out

    def forward_features(self, x, mask=None):
        """Return the residual update before adding the identity connection."""
        x = self.norm(x)
        x = _apply_mask(x, mask)
        x = x.permute(0, 2, 3, 1)  # B C H W -> B H W C
        x = self._scan(x)
        x = x.permute(0, 3, 1, 2)  # B H W C -> B C H W
        x = _apply_mask(x, mask)
        return x

    def forward(self, x, mask=None):
        return x + self.forward_features(x, mask)


class CrossDirectionalMamba(DirectionalMamba):
    """Add each image's independently scanned update to its paired image."""

    @staticmethod
    def _crop_or_pad_like(src, ref):
        """Match spatial size by bottom/right cropping or edge-value padding."""
        _, _, h0, w0 = src.shape
        _, _, h1, w1 = ref.shape

        src = src[:, :, :min(h0, h1), :min(w0, w1)]

        pad_h = max(h1 - h0, 0)
        pad_w = max(w1 - w0, 0)
        if pad_h > 0 or pad_w > 0:
            src = F.pad(src, (0, pad_w, 0, pad_h), mode="replicate")

        return src

    def forward(self, x0, x1=None, mask0=None, mask1=None):
        if x1 is None:
            b_total = x0.shape[0]
            assert b_total % 2 == 0, "When x1 is None, x0.shape[0] must be even."

            residual = x0
            updates = self.forward_features(x0, mask0)
            update0, update1 = torch.chunk(updates, 2, dim=0)
            if mask0 is not None:
                mask0, mask1 = torch.chunk(mask0, 2, dim=0)
                update0 = _apply_mask(update0, mask1)
                update1 = _apply_mask(update1, mask0)
            residual0, residual1 = torch.chunk(residual, 2, dim=0)
            out0 = residual0 + update1
            out1 = residual1 + update0
            return torch.cat([out0, out1], dim=0)

        residual0 = x0
        residual1 = x1
        update0 = self.forward_features(x0, mask0)
        update1 = self.forward_features(x1, mask1)
        update0 = self._crop_or_pad_like(update0, x1)
        update1 = self._crop_or_pad_like(update1, x0)
        update0 = _apply_mask(update0, mask1)
        update1 = _apply_mask(update1, mask0)
        out0 = residual0 + update1
        out1 = residual1 + update0
        return out0, out1


class UniMamba(nn.Module):
    def __init__(self, d_model, device=None, dtype=None):
        super().__init__()
        self.core = DirectionalMamba(d_model, directions=[">"], device=device, dtype=dtype)

    def forward(self, x, mask=None):
        return self.core(x, mask=mask)


class BiMamba(nn.Module):
    def __init__(self, d_model, device=None, dtype=None):
        super().__init__()
        self.core = DirectionalMamba(d_model, directions=[">", "<"], device=device, dtype=dtype)

    def forward(self, x, mask=None):
        return self.core(x, mask=mask)


class OrthMamba(nn.Module):
    def __init__(self, d_model, device=None, dtype=None):
        super().__init__()
        self.core = DirectionalMamba(d_model, directions=[">", "v"], device=device, dtype=dtype)

    def forward(self, x, mask=None):
        return self.core(x, mask=mask)


class FourMamba(nn.Module):
    def __init__(self, d_model, device=None, dtype=None):
        super().__init__()
        self.core = DirectionalMamba(d_model, directions=[">", "v", "<", "^"], device=device, dtype=dtype)

    def forward(self, x, mask=None):
        return self.core(x, mask=mask)


class CrossUniMamba(nn.Module):
    def __init__(self, d_model, device=None, dtype=None):
        super().__init__()
        self.core = CrossDirectionalMamba(d_model, directions=[">"], device=device, dtype=dtype)

    def forward(self, x0, x1=None, mask0=None, mask1=None):
        return self.core(x0, x1, mask0, mask1)


class CrossBiMamba(nn.Module):
    def __init__(self, d_model, device=None, dtype=None):
        super().__init__()
        self.core = CrossDirectionalMamba(d_model, directions=[">", "<"], device=device, dtype=dtype)

    def forward(self, x0, x1=None, mask0=None, mask1=None):
        return self.core(x0, x1, mask0, mask1)


class CrossOrthMamba(nn.Module):
    def __init__(self, d_model, device=None, dtype=None):
        super().__init__()
        self.core = CrossDirectionalMamba(d_model, directions=[">", "v"], device=device, dtype=dtype)

    def forward(self, x0, x1=None, mask0=None, mask1=None):
        return self.core(x0, x1, mask0, mask1)


class CrossFourMamba(nn.Module):
    def __init__(self, d_model, device=None, dtype=None):
        super().__init__()
        self.core = CrossDirectionalMamba(d_model, directions=[">", "v", "<", "^"], device=device, dtype=dtype)

    def forward(self, x0, x1=None, mask0=None, mask1=None):
        return self.core(x0, x1, mask0, mask1)


# -----------------------------------------------------------------------------
# Self-attention block
# -----------------------------------------------------------------------------

class SelfAttention(nn.Module):
    """Dense self-attention over all spatial tokens of each image."""

    def __init__(self, d_model: int, n_heads: int = 4, bias: bool = False, device=None, dtype=None):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.norm = RMSNormDim(d_model, target_dim=1)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=bias, device=device, dtype=dtype)
        self.o_proj = nn.Linear(d_model, d_model, bias=bias, device=device, dtype=dtype)

    def forward(self, x, mask=None):
        residual = x
        x = self.norm(x)

        b, c, h, w = x.shape
        seq = x.permute(0, 2, 3, 1).reshape(b, h * w, c)

        qkv = self.qkv_proj(seq).view(b, h * w, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_mask = None
        if mask is not None:
            if mask.dim() == 4:
                mask = mask.squeeze(1)
            attn_mask = mask.reshape(b, 1, 1, h * w).to(torch.bool)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(b, h * w, c)
        out = self.o_proj(out)
        out = out.view(b, h, w, c).permute(0, 3, 1, 2)

        out = _apply_mask(out, mask)
        return residual + out


# -----------------------------------------------------------------------------
# Cross-attention block
# -----------------------------------------------------------------------------

class CrossAttention(nn.Module):
    """Dense cross-attention with spatial inputs and residual updates.

    Inputs are [B, C, H, W] feature maps. With x1=None, x0 and mask0 contain
    both images concatenated along the batch dimension.
    """

    def __init__(self, d_model: int, n_heads: int = 4, bias: bool = False, device=None, dtype=None):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.norm = RMSNormDim(d_model, target_dim=1)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=bias, device=device, dtype=dtype)
        self.o_proj = nn.Linear(d_model, d_model, bias=bias, device=device, dtype=dtype)

    def _project_qkv(self, x):
        b, l, d = x.shape
        qkv = self.qkv_proj(x).view(b, l, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        return qkv[0], qkv[1], qkv[2]

    def _apply_attention(self, q, k, v, q_mask=None, kv_mask=None):
        b, _, lq, _ = q.shape

        attn_mask = None
        if kv_mask is not None:
            attn_mask = kv_mask.reshape(b, 1, 1, -1).to(torch.bool)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(b, lq, self.d_model)
        out = self.o_proj(out)

        if q_mask is not None:
            out = out * q_mask.unsqueeze(-1).to(out.dtype)

        return out

    def _attend_sequences(self, seq0, seq1=None, mask0=None, mask1=None):
        """Exchange context between the two [B, L, C] token sequences."""
        if seq1 is None:
            b_total = seq0.shape[0]
            assert b_total % 2 == 0, "When x1 is None, batch size must be even."

            q_all, k_all, v_all = self._project_qkv(seq0)
            q0, q1 = torch.chunk(q_all, 2, dim=0)
            k0, k1 = torch.chunk(k_all, 2, dim=0)
            v0, v1 = torch.chunk(v_all, 2, dim=0)

            if mask0 is not None:
                mask0, mask1 = torch.chunk(mask0, 2, dim=0)

            ctx0 = self._apply_attention(q0, k1, v1, q_mask=mask0, kv_mask=mask1)
            ctx1 = self._apply_attention(q1, k0, v0, q_mask=mask1, kv_mask=mask0)
            return torch.cat([ctx0, ctx1], dim=0)

        q0, k0, v0 = self._project_qkv(seq0)
        q1, k1, v1 = self._project_qkv(seq1)

        ctx0 = self._apply_attention(q0, k1, v1, q_mask=mask0, kv_mask=mask1)
        ctx1 = self._apply_attention(q1, k0, v0, q_mask=mask1, kv_mask=mask0)
        return ctx0, ctx1

    @staticmethod
    def _flatten_mask(mask, h, w):
        if mask is None:
            return None
        if mask.dim() == 4:
            mask = mask.squeeze(1)
        return mask.reshape(mask.shape[0], h * w).to(torch.bool)

    def forward(self, x0, x1=None, mask0=None, mask1=None):
        if x1 is None:
            residual = x0
            x0 = self.norm(x0)

            b, c, h, w = x0.shape
            seq = x0.permute(0, 2, 3, 1).reshape(b, h * w, c)
            m0 = self._flatten_mask(mask0, h, w)

            out = self._attend_sequences(seq, mask0=m0)
            out = out.view(b, h, w, c).permute(0, 3, 1, 2)
            out = _apply_mask(out, mask0)
            return residual + out

        residual0 = x0
        residual1 = x1

        x0 = self.norm(x0)
        x1 = self.norm(x1)

        b0, c0, h0, w0 = x0.shape
        b1, c1, h1, w1 = x1.shape

        seq0 = x0.permute(0, 2, 3, 1).reshape(b0, h0 * w0, c0)
        seq1 = x1.permute(0, 2, 3, 1).reshape(b1, h1 * w1, c1)

        m0 = self._flatten_mask(mask0, h0, w0)
        m1 = self._flatten_mask(mask1, h1, w1)

        out0, out1 = self._attend_sequences(seq0, seq1, m0, m1)
        out0 = out0.view(b0, h0, w0, c0).permute(0, 3, 1, 2)
        out1 = out1.view(b1, h1, w1, c1).permute(0, 3, 1, 2)

        out0 = _apply_mask(out0, mask0)
        out1 = _apply_mask(out1, mask1)
        return residual0 + out0, residual1 + out1
