import torch
import torch.nn as nn
from src.third_party.convnextv2 import convnextv2_nano


class ConvNextV2_nano(nn.Module):
    def __init__(self, dim_4=128, dim_8=256, pretrained=True):
        super().__init__()
        self.cnn = convnextv2_nano()
        self.cnn.norm = None
        self.cnn.head = None
        self.cnn.downsample_layers[2] = None
        self.cnn.downsample_layers[3] = None
        self.cnn.stages[2] = None
        self.cnn.stages[3] = None

        if pretrained:
            state_dict = torch.hub.load_state_dict_from_url(
                'https://github.com/leoluxxx/JamMa/releases/download/v0.1/convnextv2_nano_pretrain.ckpt',
                file_name='convnextv2_nano_pretrain.ckpt')
            self.cnn.load_state_dict(state_dict, strict=True)

        self.lin_4 = nn.Conv2d(80, dim_4, 1)
        self.lin_8 = nn.Conv2d(160, dim_8, 1)

    def _forward_batched(self, data):
        img0 = data["imagec_0"]
        img1 = data["imagec_1"]

        B, _, H, W = img0.shape
        B1, _, H1, W1 = img1.shape
        assert B == B1
        assert H == H1 and W == W1

        x = torch.cat([img0, img1], dim=0)
        feature_pyramid = self.cnn.forward_features_8(x)

        feat_8 = self.lin_8(feature_pyramid[8])
        feat_4 = self.lin_4(feature_pyramid[4])

        feat_8_0, feat_8_1 = feat_8.split(B, dim=0)
        feat_4_0, feat_4_1 = feat_4.split(B, dim=0)

        h_8, w_8 = feat_8_0.shape[-2:]
        h_4, w_4 = feat_4_0.shape[-2:]

        data.update({
            "bs": B,
            "h_8": h_8,
            "w_8": w_8,
            "hw_8": h_8 * w_8,
            "h_4": h_4,
            "w_4": w_4,
            "hw_4": h_4 * w_4,
            "hw0_i": [H, W],
            "hw1_i": [H1, W1],
            "hw0_c": [h_8, w_8],
            "hw1_c": [h_8, w_8],
            "feat_8_0": feat_8_0,
            "feat_8_1": feat_8_1,
            "feat_4_0": feat_4_0,
            "feat_4_1": feat_4_1,
        })
        return data

    def _forward_sep(self, data):
        img0 = data["imagec_0"]
        img1 = data["imagec_1"]

        B, _, H, W = img0.shape
        B1, _, H1, W1 = img1.shape
        assert B == B1

        feat_pyr0 = self.cnn.forward_features_8(img0)
        feat_pyr1 = self.cnn.forward_features_8(img1)

        feat_8_0 = self.lin_8(feat_pyr0[8])
        feat_8_1 = self.lin_8(feat_pyr1[8])
        feat_4_0 = self.lin_4(feat_pyr0[4])
        feat_4_1 = self.lin_4(feat_pyr1[4])

        h0_8, w0_8 = feat_8_0.shape[-2:]
        h1_8, w1_8 = feat_8_1.shape[-2:]
        h0_4, w0_4 = feat_4_0.shape[-2:]
        h1_4, w1_4 = feat_4_1.shape[-2:]

        data.update({
            "bs": B,
            "h_8_0": h0_8,
            "w_8_0": w0_8,
            "hw_8_0": h0_8 * w0_8,
            "h_8_1": h1_8,
            "w_8_1": w1_8,
            "hw_8_1": h1_8 * w1_8,
            "h_4_0": h0_4,
            "w_4_0": w0_4,
            "hw_4_0": h0_4 * w0_4,
            "h_4_1": h1_4,
            "w_4_1": w1_4,
            "hw_4_1": h1_4 * w1_4,
            "hw0_i": [H, W],
            "hw1_i": [H1, W1],
            "hw0_c": [h0_8, w0_8],
            "hw1_c": [h1_8, w1_8],
            "feat_8_0": feat_8_0,
            "feat_8_1": feat_8_1,
            "feat_4_0": feat_4_0,
            "feat_4_1": feat_4_1,
        })
        return data

    def forward(self, data):
        img0 = data["imagec_0"]
        img1 = data["imagec_1"]

        if img0.shape[-2:] == img1.shape[-2:]:
            return self._forward_batched(data)
        return self._forward_sep(data)
