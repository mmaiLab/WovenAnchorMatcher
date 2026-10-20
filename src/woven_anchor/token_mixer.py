"""Token and channel mixing within paired fine-level windows."""

from torch import nn


class MLPMixerEncoderLayer(nn.Module):
    """Apply residual token and channel MLPs to features of shape [M, C, L]."""

    def __init__(self, num_tokens, feature_dim, factor=1):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(num_tokens, num_tokens * factor),
            nn.GELU(),
            nn.Linear(num_tokens * factor, num_tokens),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * factor),
            nn.LayerNorm(feature_dim * factor),
            nn.GELU(),
            nn.Linear(feature_dim * factor, feature_dim),
        )

    def forward(self, x):
        x = x + self.mlp1(x)  # Token mixing: [M, C, L].
        x = x.transpose(1, 2)
        x = x + self.mlp2(x)  # Channel mixing: [M, L, C].
        return x.transpose(1, 2)


class MLPMixerL4(nn.Module):
    """Mix the L = 2 * window_area tokens of each paired window in four layers."""

    def __init__(self, num_tokens, feature_dim, factor=1):
        super().__init__()
        self.layers = nn.Sequential(
            *[MLPMixerEncoderLayer(num_tokens, feature_dim, factor) for _ in range(4)]
        )

    def forward(self, x):
        return self.layers(x)
