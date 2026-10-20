"""Shared image conversion and figure layout helpers."""

from textwrap import fill

import matplotlib.pyplot as plt
import numpy as np
import torch


imagenet_mean = torch.tensor([0.485, 0.456, 0.406])


imagenet_std = torch.tensor([0.229, 0.224, 0.225])


def _to_display_image(image):
    mean = imagenet_mean[:, None, None].to(image.device)
    std = imagenet_std[:, None, None].to(image.device)
    image = image * std + mean
    image = image.detach().permute(1, 2, 0).cpu().numpy()
    return np.clip(image, 0.0, 1.0)


def _make_context_visualization_grid(
    image_hws, n_columns, dpi, panel_size=3.6, footer_height=0.0
):
    """Size panels by their longest side and reserve room for plot decorations."""
    row_heights = [height / max(height, width) for height, width in image_hws]
    row_widths = [width / max(height, width) for height, width in image_hws]
    column_width = max(row_widths)
    # Compression closes unused space for equally wide rows. When row widths
    # differ, it would also shrink wider panels to match the narrowest row.
    layout = "compressed" if max(row_widths) - min(row_widths) < 1e-6 else "constrained"
    fig, axes = plt.subplots(
        len(image_hws),
        n_columns,
        figsize=(
            panel_size * n_columns * column_width + 1.2,
            panel_size * sum(row_heights) + 0.7 + footer_height,
        ),
        dpi=dpi,
        squeeze=False,
        gridspec_kw={"height_ratios": row_heights},
        layout=layout,
    )
    fig.get_layout_engine().set(h_pad=0.02, hspace=0.0)
    return fig, axes


def _set_context_visualization_title(axis, title):
    """Wrap shared headers to fit narrow portrait panels."""
    n_columns = axis.get_subplotspec().get_gridspec().ncols
    panel_width = axis.figure.get_figwidth() / n_columns
    axis.set_title(
        fill(title, width=max(8, round(panel_width * 8)), break_long_words=False)
    )
