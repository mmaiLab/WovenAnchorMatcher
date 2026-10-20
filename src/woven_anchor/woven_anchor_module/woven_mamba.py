# SSM initialization adapted from Mamba; copyright (c) 2023 Tri Dao, Albert Gu.
# Modified for WovenAnchor. Apache-2.0 terms: LICENSES/Apache-2.0.txt.
# See THIRD_PARTY_NOTICES.md for the scope of the adaptation.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat

from src.woven_anchor.utils.utils import RMSNormDim
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn


class WovenMamba(nn.Module):
    """Horizontal-then-vertical state-space propagation.

    Input/output: [B, C, H, W]; optional mask: [B, H, W].
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 8,
        d_conv: int = 3,
        expand: int = 2,
        dt_rank: int | str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.4,
        dt_init_floor: float = 1e-4,
        bias: bool = False,
        device=None,
        dtype=None,
        use_checkpoint: bool = True,
        use_separator: bool = True,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else int(dt_rank)
        self.d_state = d_state
        self.use_checkpoint = use_checkpoint
        self.use_separator = use_separator

        self.in_norm = RMSNormDim(d_model, target_dim=1, eps=1e-5)

        self.in_proj = nn.Conv2d(
            d_model, self.d_inner * 2, kernel_size=1, bias=bias, **factory_kwargs
        )

        self.h_conv = nn.Conv2d(
            self.d_inner, self.d_inner, kernel_size=d_conv, padding=d_conv // 2,
            groups=self.d_inner, bias=True, **factory_kwargs
        )
        self.h_x_proj = nn.Conv1d(self.d_inner, self.dt_rank + 2 * self.d_state, kernel_size=1, bias=False, **factory_kwargs)
        self.h_dt_proj = nn.Conv1d(self.dt_rank, self.d_inner, kernel_size=1, bias=True, **factory_kwargs)

        self.v_conv = nn.Conv2d(
            self.d_inner, self.d_inner, kernel_size=d_conv, padding=d_conv // 2,
            groups=self.d_inner, bias=True, **factory_kwargs
        )
        self.v_x_proj = nn.Conv1d(self.d_inner, self.dt_rank + 2 * self.d_state, kernel_size=1, bias=False, **factory_kwargs)
        self.v_dt_proj = nn.Conv1d(self.dt_rank, self.d_inner, kernel_size=1, bias=True, **factory_kwargs)

        h_A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        v_A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()

        self.A_h_log = nn.Parameter(torch.log(h_A))
        self.A_v_log = nn.Parameter(torch.log(v_A))
        self.h_D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.v_D = nn.Parameter(torch.ones(self.d_inner, device=device))

        self.A_h_log._no_weight_decay = True
        self.A_v_log._no_weight_decay = True
        self.h_D._no_weight_decay = True
        self.v_D._no_weight_decay = True

        self.mid_norm = RMSNormDim(self.d_inner, target_dim=1, eps=1e-5)
        self.out_proj = nn.Conv2d(
            self.d_inner, d_model, kernel_size=1, bias=bias, **factory_kwargs
        )

        self._init_dt_params(self.h_dt_proj, dt_min, dt_max, dt_init_floor)
        self._init_dt_params(self.v_dt_proj, dt_min, dt_max, dt_init_floor)

    def _init_dt_params(self, dt_proj, dt_min, dt_max, dt_init_floor):
        nn.init.uniform_(dt_proj.weight, -0.01, 0.01)
        dt = torch.exp(
            torch.rand(dt_proj.bias.shape, device=dt_proj.bias.device)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))  # Inverse softplus.
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)

    def _prepare_mask(self, x: torch.Tensor, mask: torch.Tensor | None):
        if mask is None:
            return None
        if mask.dim() != 3:
            raise ValueError(f"Expected mask as [B, H, W], got {tuple(mask.shape)}")
        return mask.to(dtype=x.dtype, device=x.device)

    def _ssm_scan(self, x, dt, A, B, C, D):
        return selective_scan_fn(
            x, dt, A, B, C, D,
            z=None,
            delta_bias=None,
            delta_softplus=True,
        )

    def _serialize_scan(self, x: torch.Tensor, mask: torch.Tensor | None, axis: str):
        """Flatten in row or column order, optionally adding zero separators.

        Separators follow each row/column and remain unmasked.
        Returns [B, C, L], optional mask [B, 1, L], and original H, W.
        """
        B, C, H, W = x.shape

        if axis == "h":
            if self.use_separator:
                sep = torch.zeros(B, C, H, 1, dtype=x.dtype, device=x.device)
                x_cat = torch.cat([x, sep], dim=3)
                x_seq = rearrange(x_cat, "b c h w -> b c (h w)")
            else:
                x_seq = rearrange(x, "b c h w -> b c (h w)")


            if mask is not None:
                if self.use_separator:
                    sep_mask = torch.ones(B, H, 1, dtype=mask.dtype, device=mask.device)
                    m_cat = torch.cat([mask, sep_mask], dim=2)
                    m_seq = rearrange(m_cat, "b h w -> b 1 (h w)")
                else:
                    m_seq = rearrange(mask, "b h w -> b 1 (h w)")
            else:
                m_seq = None

        elif axis == "v":
            if self.use_separator:
                sep = torch.zeros(B, C, 1, W, dtype=x.dtype, device=x.device)
                x_cat = torch.cat([x, sep], dim=2)
                x_seq = rearrange(x_cat, "b c h w -> b c (w h)")
            else:
                x_seq = rearrange(x, "b c h w -> b c (w h)")

            if mask is not None:
                if self.use_separator:
                    sep_mask = torch.ones(B, 1, W, dtype=mask.dtype, device=mask.device)
                    m_cat = torch.cat([mask, sep_mask], dim=1)
                    m_seq = rearrange(m_cat, "b h w -> b 1 (w h)")
                else:
                    m_seq = rearrange(mask, "b h w -> b 1 (w h)")
            else:
                m_seq = None
        else:
            raise ValueError(f"Unsupported axis: {axis}")

        return x_seq, m_seq, H, W

    def _restore_scan(self, y_seq_bcl: torch.Tensor, axis: str, H: int, W: int):
        """Restore the image grid, discarding separator outputs."""
        if axis == "h":
            if self.use_separator:
                y_cat = rearrange(y_seq_bcl, "b c (h w) -> b c h w", h=H, w=W + 1)
                y_img = y_cat[:, :, :, :W]
            else:
                y_img = rearrange(y_seq_bcl, "b c (h w) -> b c h w", h=H, w=W)
        elif axis == "v":
            if self.use_separator:
                y_cat = rearrange(y_seq_bcl, "b c (w h) -> b c h w", h=H + 1, w=W)
                y_img = y_cat[:, :, :H]
            else:
                y_img = rearrange(y_seq_bcl, "b c (w h) -> b c h w", h=H, w=W)
        else:
            raise ValueError(f"Unsupported axis: {axis}")
        return y_img

    def _scan_axis(
        self,
        x_img: torch.Tensor,
        mask: torch.Tensor | None,
        conv: nn.Conv2d,
        x_proj: nn.Conv1d,
        dt_proj: nn.Conv1d,
        A_log: torch.Tensor,
        D: torch.Tensor,
        axis: str,
    ):
        if mask is not None:
            x_img = x_img * mask.unsqueeze(1)

        x = F.silu(conv(x_img))

        if mask is not None:
            x = x * mask.unsqueeze(1)

        x_seq, m_seq, H, W = self._serialize_scan(x, mask, axis=axis)

        x_dbl = x_proj(x_seq)
        dt_lr, B_ssm, C_ssm = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=1
        )

        dt = dt_proj(dt_lr)

        if self.use_separator:
            if axis == "h":
                l = W + 1
            else:
                l = H + 1
            # Increase separator time steps before softplus.
            dt[:, :, l-1::l] += 6.0

        if m_seq is not None:
            # Masked tokens have near-zero time steps after softplus.
            dt = dt.masked_fill(m_seq == 0, -20.0)

        y = self._ssm_scan(
            x_seq,
            dt,
            -torch.exp(A_log.float()),
            B_ssm,
            C_ssm,
            D.float(),
        )

        y_img = self._restore_scan(y, axis=axis, H=H, W=W)

        if mask is not None:
            y_img = y_img * mask.unsqueeze(1)

        return y_img

    def _compose_scans(
        self,
        x_main: torch.Tensor,
        mask: torch.Tensor | None,
    ):
        """Apply horizontal propagation, normalization, then vertical propagation."""
        y_h = self._scan_axis(
            x_main, mask,
            self.h_conv, self.h_x_proj, self.h_dt_proj,
            self.A_h_log, self.h_D,
            axis="h",
        )

        v_input = self.mid_norm(y_h)
        return self._scan_axis(
            v_input, mask,
            self.v_conv, self.v_x_proj, self.v_dt_proj,
            self.A_v_log, self.v_D,
            axis="v",
        )

    def forward_features(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        """Return the update before residual addition."""
        x = self.in_norm(x)
        mask = self._prepare_mask(x, mask)

        xz = self.in_proj(x)
        x_main, z = xz.chunk(2, dim=1)

        y_scan = self._compose_scans(x_main, mask)

        y = y_scan * F.silu(z)
        y = self.out_proj(y)

        if mask is not None:
            y = y * mask.unsqueeze(1)

        return y

    def _forward_body(self, x: torch.Tensor, mask: torch.Tensor | None):
        return x + self.forward_features(x, mask)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        if self.use_checkpoint and self.training:
            return checkpoint.checkpoint(self._forward_body, x, mask, use_reentrant=False)
        return self._forward_body(x, mask)


class CrossWovenMamba(WovenMamba):
    """Swap only the updates between batches [image0_batch; image1_batch]."""

    def _forward_cross_body(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None,
    ):
        if x.shape[0] % 2 != 0:
            raise ValueError(
                "CrossWovenMamba expects an even batch arranged as "
                "[image0_batch; image1_batch]."
            )

        update = self.forward_features(x, mask)
        update0, update1 = torch.chunk(update, 2, dim=0)
        cross_update = torch.cat([update1, update0], dim=0)

        if mask is not None:
            target_mask = self._prepare_mask(x, mask)
            cross_update = cross_update * target_mask.unsqueeze(1)

        return x + cross_update

    def forward(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor | None = None,
        mask0: torch.Tensor | None = None,
        mask1: torch.Tensor | None = None,
    ):
        if x1 is not None or mask1 is not None:
            raise ValueError(
                "CrossWovenMamba supports only equal-size image pairs through "
                "the concatenated batched path."
            )
        if self.use_checkpoint and self.training:
            return checkpoint.checkpoint(
                self._forward_cross_body,
                x0,
                mask0,
                use_reentrant=False,
            )
        return self._forward_cross_body(x0, mask0)
