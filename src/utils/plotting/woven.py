"""WovenMamba and OrthMamba effective receptive field figures and raw arrays."""

from pathlib import Path

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from .common import (
    _make_context_visualization_grid,
    _set_context_visualization_title,
    _to_display_image,
)


_ERF_TARGET_COLUMNS = {
    "upper_left": 1,
    "upper_right": 2,
    "center": 3,
    "lower_left": 4,
    "lower_right": 5,
}


_ERF_TARGET_SHORT_LABELS = {
    "upper_left": "UL",
    "upper_right": "UR",
    "center": "C",
    "lower_left": "LL",
    "lower_right": "LR",
}


def _woven_valid_image_hw(
    data,
    image_index,
    batch_index,
    image_hw,
    crop_hw,
    coarse_hw,
):
    prepad_size = data.get(f"prepad_size{image_index}")
    if prepad_size is not None:
        size = prepad_size[batch_index]
        if torch.is_tensor(size):
            size = size.detach().cpu().tolist()
        valid_image_h, valid_image_w = (int(value) for value in size)
    else:
        image_h, image_w = image_hw
        crop_h, crop_w = crop_hw
        coarse_h, coarse_w = coarse_hw
        valid_image_h = int(round(image_h * crop_h / coarse_h))
        valid_image_w = int(round(image_w * crop_w / coarse_w))

    return (
        min(max(valid_image_h, 1), image_hw[0]),
        min(max(valid_image_w, 1), image_hw[1]),
    )


def _resize_woven_erf(erf_map, valid, crop_hw, image_hw):
    crop_h, crop_w = crop_hw
    erf_map = erf_map[:crop_h, :crop_w][None, None]
    valid = valid[:crop_h, :crop_w].float()[None, None]
    erf_map = F.interpolate(erf_map, size=image_hw, mode="nearest")[0, 0]
    valid = F.interpolate(valid, size=image_hw, mode="nearest")[0, 0].bool()
    return erf_map.numpy(), valid.numpy()


def _woven_target_display_xy(target_yx, crop_hw, image_hw):
    target_y, target_x = target_yx
    crop_h, crop_w = crop_hw
    image_h, image_w = image_hw
    display_x = (target_x + 0.5) * image_w / crop_w - 0.5
    display_y = (target_y + 0.5) * image_h / crop_h - 0.5
    return display_x, display_y


def _draw_woven_target(
    axis,
    target_yx,
    crop_hw,
    image_hw,
    *,
    color="red",
    marker="x",
    annotation=None,
):
    display_x, display_y = _woven_target_display_xy(
        target_yx,
        crop_hw,
        image_hw,
    )
    marker_artist = axis.scatter(
        [display_x],
        [display_y],
        marker=marker,
        c=[color],
        s=90 if marker == "x" else 75,
        linewidths=2.0 if marker == "x" else 1.3,
        zorder=5,
    )
    marker_artist.set_path_effects(
        [path_effects.Stroke(linewidth=3.5, foreground="white"), path_effects.Normal()]
    )
    if annotation:
        text_artist = axis.annotate(
            annotation,
            (display_x, display_y),
            xytext=(5, 5),
            textcoords="offset points",
            color=color,
            fontsize=9,
            fontweight="bold",
            zorder=6,
        )
        text_artist.set_path_effects(
            [path_effects.Stroke(linewidth=2.5, foreground="white"), path_effects.Normal()]
        )


def _woven_target_columns(target_names):
    available = range(1, 6)
    assigned = {}
    used = set()
    for name in target_names:
        position = _ERF_TARGET_COLUMNS.get(name)
        if position is not None and position not in used:
            assigned[name] = position
            used.add(position)
    remaining = (position for position in available if position not in used)
    for name in target_names:
        if name not in assigned:
            assigned[name] = next(remaining)
    return assigned


def make_woven_receptive_field_figures(
    data,
    receptive_field_visualization,
    output_dir,
    filename_prefix,
    dpi=120,
):
    """Save one local ERF figure, with one six-panel row per image."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad((0.0, 0.0, 0.0, 0.0))
    target_cmap = plt.get_cmap("tab10")
    saved_paths = []

    batch_size = data["imagec_0"].shape[0]
    for batch_index in range(batch_size):
        display_images = []
        for image_index in (0, 1):
            record = receptive_field_visualization[f"image{image_index}"]
            rgb = _to_display_image(data[f"imagec_{image_index}"][batch_index])
            crop_hw = tuple(int(value) for value in record["crop_hw"][batch_index])
            valid_image_hw = _woven_valid_image_hw(
                data,
                image_index,
                batch_index,
                rgb.shape[:2],
                crop_hw,
                tuple(record["valid"].shape[-2:]),
            )
            valid_image_h, valid_image_w = valid_image_hw
            display_images.append(
                (rgb[:valid_image_h, :valid_image_w], crop_hw, valid_image_hw)
            )

        fig, axes = _make_context_visualization_grid(
            [image_hw for _, _, image_hw in display_images],
            6,
            dpi,
        )
        heatmap_artists = []
        raw_payload = {
            "schema_version": np.asarray(3, dtype=np.int64),
            "operator_name": np.asarray(
                receptive_field_visualization["operator_name"]
            ),
            "operator_detail": np.asarray(
                receptive_field_visualization["operator_detail"]
            ),
            "spatial_block_index": np.asarray(
                receptive_field_visualization["spatial_block_index"],
                dtype=np.int64,
            ),
            "spatial_block_count": np.asarray(
                receptive_field_visualization["spatial_block_count"],
                dtype=np.int64,
            ),
            "layer_index": np.asarray(
                receptive_field_visualization["layer_index"],
                dtype=np.int64,
            ),
            "gradient_reference": np.asarray(
                receptive_field_visualization["gradient_reference"]
            ),
            "gradient_definition": np.asarray(
                "channel-L2 of d(||residual_update[target]||2)/d(block_input)"
            ),
        }

        for image_index in (0, 1):
            image_key = f"image{image_index}"
            record = receptive_field_visualization[image_key]
            target_names = tuple(record["target_names"])
            target_columns = _woven_target_columns(target_names)
            target_colors = {
                name: target_cmap(target_index % 10)
                for target_index, name in enumerate(target_names)
            }

            rgb, crop_hw, valid_image_hw = display_images[image_index]
            target_yx = record["target_yx"][batch_index]
            valid = record["valid"][batch_index]

            overview_axis = axes[image_index, 0]
            overview_axis.imshow(rgb)
            if image_index == 0:
                _set_context_visualization_title(overview_axis, "Targets")
            overview_axis.set_ylabel(
                f"Image {image_index}", rotation=90, ha="center", va="bottom", labelpad=5
            )
            for target_index, name in enumerate(target_names):
                coordinates = tuple(int(value) for value in target_yx[target_index])
                _draw_woven_target(
                    overview_axis,
                    coordinates,
                    crop_hw,
                    valid_image_hw,
                    color=target_colors[name],
                    marker="o",
                    annotation=_ERF_TARGET_SHORT_LABELS.get(
                        name, str(target_index + 1)
                    ),
                )

            erf_maxima = []
            for target_index, name in enumerate(target_names):
                axis = axes[image_index, target_columns[name]]
                raw_erf_map = record["erf_maps"][batch_index, target_index]
                selected = raw_erf_map[valid]
                erf_max = (
                    float(selected.max().item()) if selected.numel() else 0.0
                )
                erf_maxima.append(erf_max)
                safe_max = max(erf_max, np.finfo(np.float32).tiny)

                erf_map, display_valid = _resize_woven_erf(
                    raw_erf_map,
                    valid,
                    crop_hw,
                    valid_image_hw,
                )
                # Eq. (16): log10(max(E_t(p) / max_{q in V} E_t(q), 1e-6)).
                relative = erf_map / safe_max
                log_relative = np.log10(np.clip(relative, 1e-6, 1.0))
                log_relative = np.ma.masked_where(~display_valid, log_relative)

                axis.imshow(rgb)
                artist = axis.imshow(
                    log_relative,
                    cmap=cmap,
                    vmin=-6.0,
                    vmax=0.0,
                    alpha=0.78,
                    interpolation="nearest",
                )
                heatmap_artists.append(artist)
                coordinates = tuple(int(value) for value in target_yx[target_index])
                if image_index == 0:
                    _set_context_visualization_title(
                        axis, name.replace("_", " ").title()
                    )
                _draw_woven_target(
                    axis,
                    coordinates,
                    crop_hw,
                    valid_image_hw,
                    color=target_colors[name],
                )

            for axis in axes[image_index]:
                axis.set_anchor("S" if image_index == 0 else "N")
                axis.set_xticks([])
                axis.set_yticks([])

            raw_payload[f"image{image_index}_valid"] = valid.numpy()
            raw_payload[f"image{image_index}_crop_hw"] = np.asarray(
                crop_hw, dtype=np.int64
            )
            raw_payload[f"image{image_index}_valid_bbox_yxyx"] = record[
                "valid_bbox_yxyx"
            ][batch_index].numpy()
            raw_payload[f"image{image_index}_target_names"] = np.asarray(
                target_names
            )
            raw_payload[f"image{image_index}_target_fractions_yx"] = record[
                "target_fractions_yx"
            ].numpy()
            raw_payload[f"image{image_index}_target_yx"] = target_yx.numpy()
            raw_payload[f"image{image_index}_target_output_norms"] = record[
                "target_output_norms"
            ][batch_index].numpy()
            raw_payload[f"image{image_index}_erf_max"] = np.asarray(
                erf_maxima, dtype=np.float32
            )
            raw_payload[f"image{image_index}_erf_maps"] = record[
                "erf_maps"
            ][batch_index].numpy()

        block_index = int(receptive_field_visualization["spatial_block_index"])
        layer_index = int(receptive_field_visualization["layer_index"])
        operator_name = receptive_field_visualization["operator_name"]
        fig.suptitle(
            f"{operator_name}-{block_index + 1}: local effective receptive fields"
        )
        if heatmap_artists:
            fig.colorbar(
                heatmap_artists[-1],
                ax=axes.ravel().tolist(),
                label=r"$\log_{10}$ (gradient L2 / target max)",
                shrink=0.88,
                pad=0.008,
            )

        operator_slug = receptive_field_visualization["operator_slug"]
        output_stem = (
            f"{filename_prefix}_pair{batch_index}_{operator_slug}_"
            f"spatial_block{block_index + 1}_layer{layer_index}_multipoint_erf"
        )
        output_path = output_dir / f"{output_stem}.png"
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        np.savez_compressed(output_dir / f"{output_stem}.npz", **raw_payload)
        saved_paths.append(output_path)

    return saved_paths
