"""Local ERFs of residual updates with respect to each spatial block's input."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from src.woven_anchor.utils.utils import complete_pair_masks

from .ablation_blocks import OrthMamba
from .woven_mamba import WovenMamba


DEFAULT_ERF_TARGET_SPECS = (
    ("center", (0.5, 0.5)),
    ("upper_left", (0.2, 0.2)),
    ("upper_right", (0.2, 0.8)),
    ("lower_left", (0.8, 0.2)),
    ("lower_right", (0.8, 0.8)),
)


def _validate_target_specs(target_specs):
    specs = tuple(
        (str(name), (float(fraction_y), float(fraction_x)))
        for name, (fraction_y, fraction_x) in target_specs
    )
    if not specs:
        raise ValueError("ERF analysis requires at least one target point.")

    names = [name for name, _ in specs]
    if len(names) != len(set(names)):
        raise ValueError(f"ERF target names must be unique, got {names}.")

    for name, fractions in specs:
        if any(value < 0.0 or value > 1.0 for value in fractions):
            raise ValueError(
                f"ERF target fractions for {name!r} must be in [0, 1], "
                f"got {fractions}."
            )
    return specs


def _valid_region_metadata(
    mask: torch.Tensor | None,
    *,
    batch_size: int,
    height: int,
    width: int,
    device: torch.device,
    target_specs: Sequence[tuple[str, tuple[float, float]]],
):
    """Return valid masks, crop extents, and nearest-valid target coordinates."""
    if mask is None:
        valid = torch.ones(
            batch_size,
            height,
            width,
            dtype=torch.bool,
            device=device,
        )
    else:
        if tuple(mask.shape) != (batch_size, height, width):
            raise ValueError(
                "ERF mask shape must match the feature map: "
                f"expected {(batch_size, height, width)}, got {tuple(mask.shape)}"
            )
        valid = mask.to(device=device, dtype=torch.bool)

    crop_hw = []
    valid_bbox_yxyx = []
    target_yx = []
    for batch_index in range(batch_size):
        coordinates = valid[batch_index].nonzero(as_tuple=False)
        if coordinates.numel() == 0:
            raise ValueError(
                f"ERF analysis requires at least one valid point for batch "
                f"{batch_index}."
            )

        y_min = int(coordinates[:, 0].min().item())
        y_max = int(coordinates[:, 0].max().item())
        x_min = int(coordinates[:, 1].min().item())
        x_max = int(coordinates[:, 1].max().item())

        # Crop from the origin; keep the full valid bounding box separately.
        crop_hw.append((y_max + 1, x_max + 1))
        valid_bbox_yxyx.append((y_min, y_max + 1, x_min, x_max + 1))

        coordinates_float = coordinates.float()
        sample_targets = []
        for _, (fraction_y, fraction_x) in target_specs:
            desired = coordinates_float.new_tensor(
                [
                    y_min + fraction_y * (y_max - y_min),
                    x_min + fraction_x * (x_max - x_min),
                ]
            )
            distances = (coordinates_float - desired).square().sum(dim=1)
            nearest = coordinates[distances.argmin()]
            sample_targets.append(
                (int(nearest[0].item()), int(nearest[1].item()))
            )
        target_yx.append(sample_targets)

    return valid, crop_hw, valid_bbox_yxyx, target_yx


def compute_effective_receptive_fields(
    feature_block,
    features: torch.Tensor,
    mask: torch.Tensor | None = None,
    target_specs: Sequence[
        tuple[str, tuple[float, float]]
    ] = DEFAULT_ERF_TARGET_SPECS,
):
    """Compute local block ERFs, even under inference mode."""
    specs = _validate_target_specs(target_specs)
    with torch.inference_mode(False), torch.enable_grad():
        # Clone outside inference mode so autograd can use these tensors.
        ordinary_features = features.detach().clone()
        ordinary_mask = None if mask is None else mask.detach().clone()
        return _compute_effective_receptive_fields(
            feature_block,
            ordinary_features,
            ordinary_mask,
            specs,
        )


def _compute_effective_receptive_fields(
    feature_block,
    features: torch.Tensor,
    mask: torch.Tensor | None,
    target_specs,
):
    """Return raw ERFs [B, targets, H, W] from target residual-update L2 norms.

    Normalization and log scaling are applied only when plotting.
    """
    if features.dim() != 4:
        raise ValueError(
            f"ERF features must be [B, C, H, W], got {tuple(features.shape)}"
        )
    if not hasattr(feature_block, "forward_features"):
        raise TypeError("ERF feature block must expose forward_features(x, mask).")

    batch_size, _, height, width = features.shape
    valid, crop_hw, valid_bbox_yxyx, target_yx = _valid_region_metadata(
        mask,
        batch_size=batch_size,
        height=height,
        width=width,
        device=features.device,
        target_specs=target_specs,
    )

    erf_maps = []
    target_output_norms = []
    # Recompute each target's graph to limit peak memory.
    for target_index in range(len(target_specs)):
        # Exclude gradients through preceding layers.
        analysis_input = features.detach().clone().requires_grad_(True)
        # Measure the update before adding the identity connection.
        residual_update = feature_block.forward_features(analysis_input, mask=mask)
        if residual_update.shape != analysis_input.shape:
            raise ValueError(
                "ERF residual-branch output must match its input shape: "
                f"input={tuple(analysis_input.shape)}, output={tuple(residual_update.shape)}"
            )

        sample_norms = []
        for batch_index in range(batch_size):
            target_y, target_x = target_yx[batch_index][target_index]
            target_vector = residual_update[batch_index, :, target_y, target_x].float()
            sample_norms.append(
                (target_vector.square().sum() + 1e-12).sqrt()
            )
        sample_norms = torch.stack(sample_norms)

        gradient = torch.autograd.grad(
            sample_norms.sum(),
            analysis_input,
            retain_graph=False,
            create_graph=False,
            only_inputs=True,
        )[0]
        # Channel L2 norm of the gradient at each source position.
        gradient_magnitude = gradient.detach().float().square().sum(dim=1).sqrt()
        gradient_magnitude = gradient_magnitude.masked_fill(~valid, 0.0)
        erf_maps.append(gradient_magnitude.cpu())
        target_output_norms.append(sample_norms.detach().float().cpu())

    return {
        "target_names": tuple(name for name, _ in target_specs),
        "target_fractions_yx": torch.tensor(
            [fractions for _, fractions in target_specs],
            dtype=torch.float32,
        ),
        "valid": valid.detach().cpu(),
        "crop_hw": torch.tensor(crop_hw, dtype=torch.long),
        "valid_bbox_yxyx": torch.tensor(valid_bbox_yxyx, dtype=torch.long),
        "target_yx": torch.tensor(target_yx, dtype=torch.long),
        "target_output_norms": torch.stack(target_output_norms, dim=1),
        "erf_maps": torch.stack(erf_maps, dim=1),
    }


class ReceptiveFieldAnalysis:
    """Analyze WovenMamba/OrthMamba layers using the model's forward path."""

    def __init__(self, module):
        self.module = module
        self.layers = module.layers

    @staticmethod
    def _receptive_field_operator(block):
        """Return metadata and the residual-update block for supported layers."""
        if type(block) is WovenMamba:
            return {
                "operator_name": "WovenMamba",
                "operator_slug": "wovenmamba",
                "operator_detail": "composition=sequential",
                "feature_block": block,
            }
        if type(block) is OrthMamba:
            return {
                "operator_name": "OrthMamba",
                "operator_slug": "orthmamba",
                "operator_detail": "independently learned horizontal/vertical scans",
                "feature_block": block.core,
            }
        return None

    @staticmethod
    def _slice_receptive_field_record(record, batch_slice):
        return {
            "target_names": record["target_names"],
            "target_fractions_yx": record["target_fractions_yx"],
            "valid": record["valid"][batch_slice],
            "crop_hw": record["crop_hw"][batch_slice],
            "valid_bbox_yxyx": record["valid_bbox_yxyx"][batch_slice],
            "target_yx": record["target_yx"][batch_slice],
            "target_output_norms": record["target_output_norms"][batch_slice],
            "erf_maps": record["erf_maps"][batch_slice],
        }

    @staticmethod
    def _receptive_field_result(
        operator,
        spatial_block_index,
        layer_index,
        image0,
        image1,
    ):
        return {
            "spatial_block_index": spatial_block_index,
            "layer_index": layer_index,
            "operator_name": operator["operator_name"],
            "operator_slug": operator["operator_slug"],
            "operator_detail": operator["operator_detail"],
            "gradient_reference": "local_block_input",
            "image0": image0,
            "image1": image1,
        }

    def _analyze_batched(
        self,
        x0,
        x1,
        mask0,
        mask1,
        supported_layer_indices,
    ):
        batch_size = x0.shape[0]
        x = torch.cat([x0, x1], dim=0)
        mask = torch.cat([mask0, mask1], dim=0) if mask0 is not None else None

        last_supported_layer = supported_layer_indices[-1]
        analyses = []
        spatial_block_index = 0
        for layer_index, block in enumerate(self.layers):
            operator = self._receptive_field_operator(block)
            if operator is not None:
                record = compute_effective_receptive_fields(
                    operator["feature_block"],
                    x,
                    mask,
                )
                analyses.append(
                    self._receptive_field_result(
                        operator,
                        spatial_block_index,
                        layer_index,
                        self._slice_receptive_field_record(
                            record, slice(0, batch_size)
                        ),
                        self._slice_receptive_field_record(
                            record, slice(batch_size, 2 * batch_size)
                        ),
                    )
                )
                spatial_block_index += 1
                if layer_index == last_supported_layer:
                    break

            with torch.no_grad():
                x = self.module._forward_layer(block, x, mask0=mask)

        return analyses

    def _analyze_separate(
        self,
        x0,
        x1,
        mask0,
        mask1,
        supported_layer_indices,
    ):
        last_supported_layer = supported_layer_indices[-1]
        analyses = []
        spatial_block_index = 0

        for layer_index, block in enumerate(self.layers):
            operator = self._receptive_field_operator(block)
            if operator is not None:
                analyses.append(
                    self._receptive_field_result(
                        operator,
                        spatial_block_index,
                        layer_index,
                        compute_effective_receptive_fields(
                            operator["feature_block"], x0, mask0
                        ),
                        compute_effective_receptive_fields(
                            operator["feature_block"], x1, mask1
                        ),
                    )
                )
                spatial_block_index += 1
                if layer_index == last_supported_layer:
                    break

            with torch.no_grad():
                x0, x1 = self.module._forward_layer(block, x0, x1, mask0, mask1)

        return analyses

    def run(
        self,
        x0,
        x1,
        mask0=None,
        mask1=None,
    ):
        """Return local ERFs for all supported layers."""
        supported_layer_indices = [
            layer_index
            for layer_index, block in enumerate(self.layers)
            if self._receptive_field_operator(block) is not None
        ]
        if not supported_layer_indices:
            layer_names = [type(block).__name__ for block in self.layers]
            raise RuntimeError(
                "ERF visualization requires at least one exact WovenMamba or "
                f"OrthMamba layer; got {layer_names}."
            )

        mask0, mask1 = complete_pair_masks(x0, x1, mask0, mask1)
        if x0.shape == x1.shape:
            analyses = self._analyze_batched(
                x0,
                x1,
                mask0,
                mask1,
                supported_layer_indices,
            )
        else:
            analyses = self._analyze_separate(
                x0,
                x1,
                mask0,
                mask1,
                supported_layer_indices,
            )

        spatial_block_count = len(supported_layer_indices)
        for analysis in analyses:
            analysis["spatial_block_count"] = spatial_block_count
        return analyses
