"""Compose paper-named intra-image, inter-image, and feed-forward blocks."""

import torch
import torch.nn as nn

from src.woven_anchor.utils.utils import complete_pair_masks

from .ablation_blocks import (
    BiMamba, CrossAttention, CrossBiMamba, CrossFourMamba,
    CrossOrthMamba, CrossUniMamba, FourMamba, OrthMamba,
    SelfAttention, UniMamba,
)
from .anchor_attention import (
    AnchorAttentionBlock, AnchorCrossAttention, AnchorSelfAttention,
)
from .ffn_block import FFNBlock
from .receptive_field import ReceptiveFieldAnalysis
from .woven_mamba import CrossWovenMamba, WovenMamba


_LAYER_TYPES = {
    "wovenmamba": WovenMamba,
    "crosswovenmamba": CrossWovenMamba,
    "anchorcrossattention": AnchorCrossAttention,
    "anchorselfattention": AnchorSelfAttention,
    "ffnblock": FFNBlock,
    "unimamba": UniMamba,
    "bimamba": BiMamba,
    "orthmamba": OrthMamba,
    "fourmamba": FourMamba,
    "crossunimamba": CrossUniMamba,
    "crossbimamba": CrossBiMamba,
    "crossorthmamba": CrossOrthMamba,
    "crossfourmamba": CrossFourMamba,
    "selfattention": SelfAttention,
    "crossattention": CrossAttention,
}

# Blocks with a paired-image API, including within-image AnchorSelfAttention.
_PAIRED_BLOCKS = (
    AnchorAttentionBlock, CrossAttention, CrossWovenMamba,
    CrossUniMamba, CrossBiMamba, CrossOrthMamba, CrossFourMamba,
)


class WovenAnchorModule(nn.Module):
    """Stack WovenMamba -> AnchorCrossAttention -> FFN stages or ablations."""

    def __init__(self, d_model: int, config, device=None, dtype=None):
        super().__init__()
        self.layers = nn.ModuleList()
        for name in config["layers"]:
            key = name.lower()
            if key not in _LAYER_TYPES:
                raise ValueError(f"Unknown layer type: {name}")
            block_type = _LAYER_TYPES[key]
            if issubclass(block_type, WovenMamba):
                options = config["woven"]
            elif issubclass(block_type, AnchorAttentionBlock):
                options = config["anchor"]
            else:
                options = {}
            self.layers.append(block_type(d_model, **options, device=device, dtype=dtype))

    def set_anchor_probability_capture(self, enabled: bool):
        for layer in self.layers:
            if isinstance(layer, AnchorAttentionBlock):
                layer.set_anchor_probability_capture(enabled)

    def set_anchor_probability_statistics(self, enabled: bool):
        for layer in self.layers:
            if isinstance(layer, AnchorAttentionBlock):
                layer.set_anchor_probability_statistics(enabled)

    def _pop_anchor_records(self, method_name):
        records = []
        anchor_block_index = 0
        for layer_index, layer in enumerate(self.layers):
            if not isinstance(layer, AnchorAttentionBlock):
                continue
            record = getattr(layer, method_name)()
            if record is not None:
                record.update(anchor_block_index=anchor_block_index, layer_index=layer_index)
                records.append(record)
            anchor_block_index += 1
        return records

    def pop_anchor_probability_views(self):
        return self._pop_anchor_records("pop_anchor_probability_views")

    def pop_anchor_probability_statistics(self):
        return self._pop_anchor_records("pop_anchor_probability_statistics")

    def receptive_field_analysis(self, x0, x1, mask0=None, mask1=None):
        return ReceptiveFieldAnalysis(self).run(x0, x1, mask0, mask1)

    @staticmethod
    def _forward_layer(block, x0, x1=None, mask0=None, mask1=None):
        """Dispatch single/paired block calls for training, evaluation, and ERFs."""
        if isinstance(block, _PAIRED_BLOCKS):
            return block(x0, x1, mask0=mask0, mask1=mask1)
        if x1 is None:
            return block(x0, mask=mask0)
        return block(x0, mask=mask0), block(x1, mask=mask1)

    def _forward_separate(self, x0, x1, mask0=None, mask1=None):
        """Keep separate feature tensors for images with different shapes."""
        for block in self.layers:
            x0, x1 = self._forward_layer(block, x0, x1, mask0, mask1)
        return x0, x1

    def _forward_batched(self, x0, x1, mask0=None, mask1=None):
        """Process equally shaped images together along the batch dimension."""
        x = torch.cat([x0, x1], dim=0)
        mask = torch.cat([mask0, mask1], dim=0) if mask0 is not None else None
        for block in self.layers:
            x = self._forward_layer(block, x, mask0=mask)
        return torch.chunk(x, 2, dim=0)

    def forward(self, data):
        x0, x1 = data["feat_8_0"], data["feat_8_1"]
        mask0, mask1 = complete_pair_masks(
            x0, x1, data.get("mask0"), data.get("mask1")
        )
        if x0.shape == x1.shape:
            x0, x1 = self._forward_batched(x0, x1, mask0, mask1)
        else:
            x0, x1 = self._forward_separate(x0, x1, mask0, mask1)
        data["feat_8_0"] = x0
        data["feat_8_1"] = x1
