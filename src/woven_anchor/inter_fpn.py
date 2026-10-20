"""Fuse 1/8 and 1/4 features and produce 1/2-resolution fine features."""

import torch
from torch import nn


class DepthwiseSeparableConv(nn.Module):
    """A spatial depthwise convolution followed by a channel projection."""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1,
                 identity_init=True, init_scale=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_scale = init_scale
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            padding=padding, groups=in_channels, bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        if identity_init:
            self._identity_init()

    @torch.no_grad()
    def _identity_init(self):
        """Initialize the convolutions as channel copies with small noise."""
        nn.init.zeros_(self.depthwise.weight)
        center = self.depthwise.kernel_size[0] // 2
        self.depthwise.weight[:, 0, center, center] = 1.0
        self.depthwise.weight.add_(torch.randn_like(self.depthwise.weight) * self.init_scale * 0.01)

        nn.init.zeros_(self.pointwise.weight)
        channels = torch.arange(min(self.in_channels, self.out_channels), device=self.pointwise.weight.device)
        self.pointwise.weight[channels, channels, 0, 0] = 1.0
        self.pointwise.weight.add_(torch.randn_like(self.pointwise.weight) * self.init_scale * 0.1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class DepthwisePixelShuffle(nn.Module):
    """Upsample by two, then normalize and activate the projected features."""

    def __init__(self, dim_in, dim_out, identity_init=True, init_scale=0.1):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.init_scale = init_scale
        self.conv = DepthwiseSeparableConv(dim_in, dim_out * 4, identity_init=False)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.norm = nn.GroupNorm(8, dim_out)
        self.act = nn.GELU()
        if identity_init:
            self._init_upsampling_weights()

    @torch.no_grad()
    def _init_upsampling_weights(self):
        """Initialize scaled 2x2 pixel copies before GroupNorm and GELU."""
        nn.init.zeros_(self.conv.depthwise.weight)
        center = self.conv.depthwise.kernel_size[0] // 2
        self.conv.depthwise.weight[:, 0, center, center] = 1.0

        # Four consecutive channels become one 2x2 block after PixelShuffle.
        nn.init.zeros_(self.conv.pointwise.weight)
        device = self.conv.pointwise.weight.device
        input_channels = torch.arange(min(self.dim_in, self.dim_out), device=device).repeat_interleave(4)
        output_channels = torch.arange(input_channels.numel(), device=device)
        self.conv.pointwise.weight[output_channels, input_channels, 0, 0] = 0.5

        self.conv.depthwise.weight.add_(torch.randn_like(self.conv.depthwise.weight) * self.init_scale * 0.01)
        self.conv.pointwise.weight.add_(torch.randn_like(self.conv.pointwise.weight) * self.init_scale * 0.1)
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)

    def forward(self, x):
        x = self.pixel_shuffle(self.conv(x))
        return self.act(self.norm(x))


class InterFPN(nn.Module):
    """Fuse coarse and backbone features using two stages of 2x upsampling.

    identity_init controls convolution initialization; the normalization and
    nonlinearities still transform the features.
    """

    def __init__(self, dim, identity_init=True, init_scale=0.1):
        super().__init__()
        self.init_scale = init_scale
        coarse_dim, middle_dim, fine_dim = dim

        self.up2 = DepthwisePixelShuffle(coarse_dim, middle_dim, identity_init, init_scale)
        self.conv7a = nn.Conv2d(2 * middle_dim, middle_dim, 3, padding=1, bias=False)
        self.conv7b = nn.Conv2d(middle_dim, middle_dim, 3, padding=1, bias=False)
        self.up3 = DepthwisePixelShuffle(middle_dim, fine_dim, identity_init, init_scale)
        self.conv8a = nn.Conv2d(fine_dim, fine_dim, 3, padding=1, bias=False)
        self.conv8b = nn.Conv2d(fine_dim, fine_dim, 3, padding=1)
        self.act = nn.GELU()
        if identity_init:
            self._init_conv_layers()

    @torch.no_grad()
    def _init_conv_layers(self):
        """Initialize equal feature fusion and channel copies, with small noise."""
        nn.init.zeros_(self.conv7a.weight)
        channels = torch.arange(self.conv7a.out_channels, device=self.conv7a.weight.device)
        self.conv7a.weight[channels, channels + self.conv7a.in_channels // 2, 1, 1] = 0.5
        self.conv7a.weight[channels, channels, 1, 1] = 0.5
        self.conv7a.weight.add_(torch.randn_like(self.conv7a.weight) * self.init_scale * 0.1)

        for conv in (self.conv7b, self.conv8a, self.conv8b):
            nn.init.dirac_(conv.weight)
            conv.weight.add_(torch.randn_like(conv.weight) * self.init_scale * 0.1)
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)

    def forward(self, feat_8, feat_4):
        upsampled_4 = self.up2(feat_8)
        fused_4 = self.act(self.conv7a(torch.cat([feat_4, upsampled_4], dim=1)))
        fused_4 = self.act(self.conv7b(fused_4))
        fine_features = self.up3(fused_4)
        fine_features = self.act(self.conv8a(fine_features))
        return self.conv8b(fine_features)
