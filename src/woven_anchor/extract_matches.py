"""Directional softmax nearest-neighbor selection with fused Triton reductions."""

import torch
import triton
import triton.language as tl
from src.woven_anchor.utils.utils import prune_matches


@triton.jit
def max_prob_dim1_kernel(
    logits_ptr,
    prob_max_ptr,
    idx_ptr,
    stride_0, stride_1, stride_2,
    dim1: tl.constexpr,
    dim2: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute the softmax maximum and row index for each (batch, column)."""
    batch_idx = tl.program_id(0)
    col_idx = tl.program_id(1)

    max_val = float('-inf')
    max_idx = 0
    sum_exp = 0.0

    for block_start in range(0, dim1, BLOCK_SIZE):
        block_mask = block_start + tl.arange(0, BLOCK_SIZE) < dim1
        row_offsets = block_start + tl.arange(0, BLOCK_SIZE)

        mem_offsets = (batch_idx * stride_0 +
                      row_offsets * stride_1 +
                      col_idx * stride_2)

        logits_block = tl.load(logits_ptr + mem_offsets, mask=block_mask, other=float('-inf'))

        block_max = tl.max(tl.where(block_mask, logits_block, float('-inf')))

        if block_max > max_val:
            # Rescale the running sum to the new maximum.
            if max_val > float('-inf'):
                sum_exp *= tl.exp(max_val - block_max)

            exp_block = tl.exp(logits_block - block_max)
            exp_block = tl.where(block_mask, exp_block, 0.0)
            sum_exp += tl.sum(exp_block)

            # Break ties by choosing the first index.
            block_max_mask = (logits_block == block_max) & block_mask
            if tl.sum(block_max_mask.to(tl.int32)) > 0:
                valid_indices = tl.where(block_max_mask, row_offsets, dim1)
                max_idx = tl.min(valid_indices)
            max_val = block_max
        else:
            exp_block = tl.exp(logits_block - max_val)
            exp_block = tl.where(block_mask, exp_block, 0.0)
            sum_exp += tl.sum(exp_block)

    # The maximum logit contributes exp(max - max) = 1.
    prob_max = 1.0 / sum_exp if sum_exp > 0 else 0.0

    output_offset = batch_idx * dim2 + col_idx
    tl.store(prob_max_ptr + output_offset, prob_max)
    tl.store(idx_ptr + output_offset, max_idx)


@triton.jit
def max_prob_dim2_kernel(
    logits_ptr,
    prob_max_ptr,
    idx_ptr,
    stride_0, stride_1, stride_2,
    dim1: tl.constexpr,
    dim2: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute the softmax maximum and column index for each (batch, row)."""
    batch_idx = tl.program_id(0)
    row_idx = tl.program_id(1)

    max_val = float('-inf')
    max_idx = 0
    sum_exp = 0.0

    for block_start in range(0, dim2, BLOCK_SIZE):
        block_mask = block_start + tl.arange(0, BLOCK_SIZE) < dim2
        col_offsets = block_start + tl.arange(0, BLOCK_SIZE)

        mem_offsets = (batch_idx * stride_0 +
                      row_idx * stride_1 +
                      col_offsets * stride_2)

        logits_block = tl.load(logits_ptr + mem_offsets, mask=block_mask, other=float('-inf'))

        block_max = tl.max(tl.where(block_mask, logits_block, float('-inf')))

        if block_max > max_val:
            # Rescale the running sum to the new maximum.
            if max_val > float('-inf'):
                sum_exp *= tl.exp(max_val - block_max)

            exp_block = tl.exp(logits_block - block_max)
            exp_block = tl.where(block_mask, exp_block, 0.0)
            sum_exp += tl.sum(exp_block)

            # Break ties by choosing the first index.
            block_max_mask = (logits_block == block_max) & block_mask
            if tl.sum(block_max_mask.to(tl.int32)) > 0:
                valid_indices = tl.where(block_max_mask, col_offsets, dim2)
                max_idx = tl.min(valid_indices)
            max_val = block_max
        else:
            exp_block = tl.exp(logits_block - max_val)
            exp_block = tl.where(block_mask, exp_block, 0.0)
            sum_exp += tl.sum(exp_block)

    # The maximum logit contributes exp(max - max) = 1.
    prob_max = 1.0 / sum_exp if sum_exp > 0 else 0.0

    output_offset = batch_idx * dim1 + row_idx
    tl.store(prob_max_ptr + output_offset, prob_max)
    tl.store(idx_ptr + output_offset, max_idx)


def max_prob_along_triton(logits: torch.Tensor, dim: int):
    """Return softmax maxima and first argmax indices along dim 1 or 2.

    Processes [B, N0, N1] without storing a dense probability matrix.
    """
    if logits.ndim != 3:
        raise ValueError(f"Expected 3D tensor for sim_matrix, got {logits.ndim}D")
    if dim not in (1, 2):
        raise ValueError(f"Only dim=1 and dim=2 are optimized, got dim={dim}")

    batch_size, num_rows, num_cols = logits.shape
    if dim == 1:
        reduction_size, num_outputs = num_rows, num_cols
        kernel = max_prob_dim1_kernel
    else:
        reduction_size, num_outputs = num_cols, num_rows
        kernel = max_prob_dim2_kernel

    output_shape = (batch_size, num_outputs)
    probabilities = torch.empty(output_shape, dtype=logits.dtype, device=logits.device)
    indices = torch.empty(output_shape, dtype=torch.long, device=logits.device)
    block_size = triton.next_power_of_2(min(reduction_size, 1024))
    kernel[output_shape](
        logits, probabilities, indices,
        *logits.stride(), num_rows, num_cols, block_size,
    )
    return probabilities, indices


@torch.no_grad()
def extract_coarse_matches(sim_matrix: torch.Tensor, data: dict, thr: float, border_rm: int):
    """Merge directional nearest neighbors, then remove padding and borders.

    Threshold each direction of [B, N0, N1] independently; duplicate pairs
    keep the larger softmax probability.
    """
    row_confidence, nearest_j = max_prob_along_triton(sim_matrix, 2)
    b0, i0 = (row_confidence > thr).nonzero(as_tuple=True)
    j0 = nearest_j[b0, i0]
    confidence0 = row_confidence[b0, i0]

    col_confidence, nearest_i = max_prob_along_triton(sim_matrix, 1)
    b1, j1 = (col_confidence > thr).nonzero(as_tuple=True)
    i1 = nearest_i[b1, j1]
    confidence1 = col_confidence[b1, j1]

    b_ids = torch.cat((b0, b1))
    i_ids = torch.cat((i0, i1))
    j_ids = torch.cat((j0, j1))
    mconf = torch.cat((confidence0, confidence1))

    # Pack (batch, i, j), reserving 24 bits per token index.
    match_keys = (b_ids.to(torch.int64) << 48) | (i_ids.to(torch.int64) << 24) | j_ids.to(torch.int64)
    match_keys, sort_order = match_keys.sort()
    b_ids = b_ids[sort_order]
    i_ids = i_ids[sort_order]
    j_ids = j_ids[sort_order]
    mconf = mconf[sort_order]

    # At most two entries per pair (one per direction); keep the larger confidence.
    same_next = match_keys[:-1] == match_keys[1:]
    mconf[:-1] = torch.where(
        same_next, torch.maximum(mconf[:-1], mconf[1:]), mconf[:-1],
    )
    keep = torch.ones_like(match_keys, dtype=torch.bool)
    keep[1:] = ~same_next

    return prune_matches(
        b_ids[keep], i_ids[keep], j_ids[keep], mconf[keep], data, border_rm,
    )
