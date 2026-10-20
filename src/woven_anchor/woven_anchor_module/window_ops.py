"""Efficient Unfold/Fold using slicing, concatenation, and summation.

Avoids earlier unfold/fold overhead; overlap weights are applied by the caller.
"""

import torch
import torch.nn.functional as F


def window_unfold(x, mask, window_size, use_overlap):
    """Pack windows into [B, C, H_u, W_u] for pooling each window.

    Overlap uses four half-window offsets. Returns features, mask,
    original size, and bottom/right padding.
    """
    B, C, H0, W0 = x.shape
    w = window_size

    if not use_overlap:
        H = H0
        W = W0

        pad_h = 0
        if H % w != 0:
            pad_h = w - (H % w)
            H += pad_h

        pad_w = 0
        if W % w != 0:
            pad_w = w - (W % w)
            W += pad_w

        if pad_h > 0 or pad_w > 0:
            x_unfolded = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0)
        else:
            x_unfolded = x

        if mask is not None:
            if pad_h > 0 or pad_w > 0:
                mask_unfolded = F.pad(mask, (0, pad_w, 0, pad_h), mode="constant", value=0)
            else:
                mask_unfolded = mask
        else:
            if pad_h > 0 or pad_w > 0:
                mask_unfolded = x.new_ones(B, 1, H0, W0)
                mask_unfolded = F.pad(mask_unfolded, (0, pad_w, 0, pad_h), mode="constant", value=0)
            else:
                mask_unfolded = None

        return x_unfolded, mask_unfolded, (H0, W0), (pad_h, pad_w)

    s = w // 2

    H = H0
    W = W0

    pad_h = 0
    if H % s != 0:
        pad_h = s - (H % s)
        H += pad_h

    pad_w = 0
    if W % s != 0:
        pad_w = s - (W % s)
        W += pad_w

    h_top = (H // w) * w
    h_bottom = ((H - s) // w) * w
    w_left = (W // w) * w
    w_right = ((W - s) // w) * w

    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0)

    x_lt = x[:, :, :h_top, :w_left]
    x_rt = x[:, :, :h_top, s:s + w_right]
    x_lb = x[:, :, s:s + h_bottom, :w_left]
    x_rb = x[:, :, s:s + h_bottom, s:s + w_right]

    x_unfolded = torch.cat(
        [
            torch.cat([x_lt, x_rt], dim=3),
            torch.cat([x_lb, x_rb], dim=3),
        ],
        dim=2,
    )

    if mask is None and (pad_h > 0 or pad_w > 0):
        mask = x.new_ones(B, 1, H0, W0)

    if mask is not None:
        if pad_h > 0 or pad_w > 0:
            mask = F.pad(mask, (0, pad_w, 0, pad_h), mode="constant", value=0)

        mask_lt = mask[:, :, :h_top, :w_left]
        mask_rt = mask[:, :, :h_top, s:s + w_right]
        mask_lb = mask[:, :, s:s + h_bottom, :w_left]
        mask_rb = mask[:, :, s:s + h_bottom, s:s + w_right]

        mask_unfolded = torch.cat(
            [
                torch.cat([mask_lt, mask_rt], dim=3),
                torch.cat([mask_lb, mask_rb], dim=3),
            ],
            dim=2,
        )
    else:
        mask_unfolded = None

    return x_unfolded, mask_unfolded, (H0, W0), (pad_h, pad_w)


def window_fold(x_unfolded, original_hw, pad_hw, window_size, use_overlap):
    """Sum overlapping window values on the original image grid, removing padding."""
    H0, W0 = original_hw
    pad_h, pad_w = pad_hw
    w = window_size

    if not use_overlap:
        return x_unfolded[:,:,:H0,:W0]

    s = w // 2

    H = H0 + pad_h
    W = W0 + pad_w

    h_top = (H // w) * w
    h_bottom = ((H - s) // w) * w
    w_left = (W // w) * w
    w_right = ((W - s) // w) * w

    B, C, H_u, W_u = x_unfolded.shape
    assert H_u == h_top + h_bottom
    assert W_u == w_left + w_right

    x_lt = x_unfolded[:, :, :h_top, :w_left]
    x_rt = x_unfolded[:, :, :h_top, w_left:]
    x_lb = x_unfolded[:, :, h_top:, :w_left]
    x_rb = x_unfolded[:, :, h_top:, w_left:]

    x = x_unfolded.new_zeros(B, C, H0, W0)

    # Offset (y, x) = (0, 0).
    h = min(h_top, H0)
    w_ = min(w_left, W0)
    x[:, :, :h, :w_] += x_lt[:, :, :h, :w_]

    # Offset (0, s).
    hs, ws = 0, s
    h = min(h_top, H0 - hs)
    w_ = min(w_right, W0 - ws)
    if h > 0 and w_ > 0:
        x[:, :, hs:hs+h, ws:ws+w_] += x_rt[:, :, :h, :w_]

    # Offset (s, 0).
    hs, ws = s, 0
    h = min(h_bottom, H0 - hs)
    w_ = min(w_left, W0 - ws)
    if h > 0 and w_ > 0:
        x[:, :, hs:hs+h, ws:ws+w_] += x_lb[:, :, :h, :w_]

    # Offset (s, s).
    hs, ws = s, s
    h = min(h_bottom, H0 - hs)
    w_ = min(w_right, W0 - ws)
    if h > 0 and w_ > 0:
        x[:, :, hs:hs+h, ws:ws+w_] += x_rb[:, :, :h, :w_]

    return x
