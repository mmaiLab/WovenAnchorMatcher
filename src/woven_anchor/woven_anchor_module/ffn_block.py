import torch
import torch.nn as nn
import torch.nn.functional as F
from src.woven_anchor.utils.utils import RMSNormDim


class FFNBlock(nn.Module):
    """
    RMSNorm + 1x1 Conv FFN + residual.
    x:    [B, C, H, W]
    mask: [B, H, W] or None
    """

    def __init__(
        self,
        d_model: int,
        ffn_expand: float = 4.0,
        dropout: float = 0.0,
        activation: str = "gelu",
        device=None,
        dtype=None,
    ):
        super().__init__()

        hidden_dim = int(ffn_expand * d_model)

        self.norm = RMSNormDim(d_model, target_dim=1)
        self.fc1 = nn.Conv2d(d_model, hidden_dim, kernel_size=1, device=device, dtype=dtype)
        self.fc2 = nn.Conv2d(hidden_dim, d_model, kernel_size=1, device=device, dtype=dtype)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        if activation == "gelu":
            self.act = F.gelu
        elif activation == "relu":
            self.act = F.relu
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        assert x.dim() == 4, f"FFNBlock expects x as [B, C, H, W], got {x.shape}"

        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        if mask is not None:
            assert mask.dim() == 3, f"Expected mask as [B, H, W], got {mask.shape}"
            mask = mask.unsqueeze(1).to(dtype=x.dtype, device=x.device)  # [B, 1, H, W]
            x = x * mask

        return residual + x