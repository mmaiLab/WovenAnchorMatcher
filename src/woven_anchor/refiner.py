"""Sub-pixel offsets predicted from matched fine-window descriptors."""

import torch
from torch import nn


class MLPRefiner(nn.Module):
    """Predict an (x, y) offset for each image, bounded to half a fine-grid cell."""

    def __init__(self, feature_dim):
        super().__init__()
        self.subpixel_mlp = nn.Sequential(
            nn.Linear(2 * feature_dim, 2 * feature_dim, bias=False),
            nn.ReLU(),
            nn.Linear(2 * feature_dim, 4, bias=False),
        )

    def forward(self, features0, features1, window_ids, token_ids0, token_ids1):
        """Select matched descriptors from [M, window_area, C] feature windows."""
        paired_features = torch.cat(
            [features0[window_ids, token_ids0], features1[window_ids, token_ids1]],
            dim=-1,
        )
        offsets0, offsets1 = self.subpixel_mlp(paired_features).chunk(2, dim=1)
        return torch.tanh(offsets0) * 0.5, torch.tanh(offsets1) * 0.5
