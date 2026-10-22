"""Anchor assignment probability visualizations."""

from pathlib import Path

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from matplotlib.patches import Patch

from .common import (
    _context_visualization_stem,
    _make_context_visualization_grid,
    _set_context_visualization_title,
    _to_display_image,
)


def _resize_anchor_phase(phase, batch_index, image_hw):
    probs = phase["probs"][batch_index].unsqueeze(0)
    valid = phase["valid"][batch_index].float().unsqueeze(0)
    probs = F.interpolate(probs, size=image_hw, mode="nearest")[0].numpy()
    valid = F.interpolate(valid, size=image_hw, mode="nearest")[0, 0].bool().numpy()
    return probs, valid


def _draw_anchor_window_grid(axis, phase, coarse_hw, image_hw):
    y0, y1, x0, x1 = phase["extent"]
    window_size = phase["window_size"]
    coarse_h, coarse_w = coarse_hw
    image_h, image_w = image_hw
    scale_y = image_h / coarse_h
    scale_x = image_w / coarse_w

    def boundary_positions(start, end):
        positions = list(range(start, end + 1, window_size))
        if not positions or positions[-1] != end:
            positions.append(end)
        return positions

    line_effects = [
        path_effects.Stroke(linewidth=1.5, foreground="black"),
        path_effects.Normal(),
    ]
    y_start = y0 * scale_y - 0.5
    y_end = y1 * scale_y - 0.5
    x_start = x0 * scale_x - 0.5
    x_end = x1 * scale_x - 0.5

    for x in boundary_positions(x0, x1):
        line = axis.plot(
            [x * scale_x - 0.5] * 2,
            [y_start, y_end],
            color="white",
            linewidth=0.7,
            alpha=0.9,
        )[0]
        line.set_path_effects(line_effects)

    for y in boundary_positions(y0, y1):
        line = axis.plot(
            [x_start, x_end],
            [y * scale_y - 0.5] * 2,
            color="white",
            linewidth=0.7,
            alpha=0.9,
        )[0]
        line.set_path_effects(line_effects)


def make_anchor_probability_figures(
    data,
    anchor_visualization,
    output_dir,
    filename_prefix,
    dpi=120,
):
    """Save one non-mixed probability figure per overlap offset family."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image0 = [_to_display_image(image) for image in data["imagec_0"]]
    image1 = [_to_display_image(image) for image in data["imagec_1"]]
    phases1 = {phase["name"]: phase for phase in anchor_visualization["image1"]}
    n_anchors = anchor_visualization["n_anchors"]
    anchor_colors = plt.get_cmap("tab10")(np.arange(n_anchors))
    probability_cmap = plt.get_cmap("magma").copy()
    probability_cmap.set_bad((0.0, 0.0, 0.0, 0.0))
    saved_paths = []

    for phase0 in anchor_visualization["image0"]:
        phase1 = phases1[phase0["name"]]
        for batch_index, (rgb0, rgb1) in enumerate(zip(image0, image1)):
            fig, axes = _make_context_visualization_grid(
                [rgb0.shape[:2], rgb1.shape[:2]],
                n_anchors + 2,
                dpi,
                footer_height=0.2,
            )
            heatmap_artist = None

            for row, (rgb, phase, image_name) in enumerate(
                [(rgb0, phase0, "Image 0"), (rgb1, phase1, "Image 1")]
            ):
                image_hw = rgb.shape[:2]
                coarse_hw = phase["probs"].shape[-2:]
                probs, valid = _resize_anchor_phase(phase, batch_index, image_hw)

                axes[row, 0].imshow(rgb)
                if row == 0:
                    _set_context_visualization_title(axes[row, 0], "Selected windows")
                _draw_anchor_window_grid(axes[row, 0], phase, coarse_hw, image_hw)

                hard_assignment = probs.argmax(axis=0)
                max_probability = probs.max(axis=0)
                categorical = anchor_colors[hard_assignment]
                categorical[..., 3] = np.where(
                    valid, 0.25 + 0.70 * max_probability, 0.0
                )
                axes[row, 1].imshow(rgb)
                axes[row, 1].imshow(categorical)
                if row == 0:
                    _set_context_visualization_title(
                        axes[row, 1], "Highest-probability anchor"
                    )
                _draw_anchor_window_grid(axes[row, 1], phase, coarse_hw, image_hw)

                for anchor_index in range(n_anchors):
                    axis = axes[row, anchor_index + 2]
                    probability = np.ma.masked_where(~valid, probs[anchor_index])
                    axis.imshow(rgb)
                    heatmap_artist = axis.imshow(
                        probability,
                        cmap=probability_cmap,
                        vmin=0.0,
                        vmax=1.0,
                        alpha=0.72,
                    )
                    if row == 0:
                        _set_context_visualization_title(axis, f"Anchor {anchor_index}")
                    _draw_anchor_window_grid(axis, phase, coarse_hw, image_hw)

                axes[row, 0].set_ylabel(
                    image_name, rotation=90, ha="center", va="bottom", labelpad=5
                )
                for axis in axes[row]:
                    axis.set_anchor("S" if row == 0 else "N")
                    axis.set_xticks([])
                    axis.set_yticks([])

            block_index = anchor_visualization["anchor_block_index"]
            layer_index = anchor_visualization["layer_index"]
            fig.suptitle(f"ACA-{block_index + 1}: learned anchor assignments")
            legend = [
                Patch(color=anchor_colors[index], label=f"Anchor {index}")
                for index in range(n_anchors)
            ]
            fig.legend(
                handles=legend,
                loc="outside lower center",
                ncol=n_anchors,
                borderaxespad=0.2,
            )
            if heatmap_artist is not None:
                fig.colorbar(
                    heatmap_artist,
                    ax=axes.ravel().tolist(),
                    label="Assignment probability",
                    shrink=0.85,
                    pad=0.008,
                )

            output_stem = _context_visualization_stem(
                filename_prefix,
                "anchor_prob",
                layer_index,
                phase=phase0["name"],
                pair_index=batch_index if len(image0) > 1 else None,
            )
            output_path = output_dir / f"{output_stem}.png"
            fig.savefig(output_path, bbox_inches="tight")
            plt.close(fig)
            saved_paths.append(output_path)

    return saved_paths
