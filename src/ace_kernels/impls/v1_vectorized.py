"""v1_vectorized: the reference with the B x C Python loop removed - nothing else.

All B*C slices are binned in one call: slice ``r`` uses bins ``[r*K, (r+1)*K)``
of a flat ``[B*C*K]`` buffer (index offset ``r*K``). The ops stay the
reference's: one-hot target, ``scatter_reduce(mean)`` for hard binning, six
``scatter_add`` for soft binning, boolean-mask NaN assignment. Isolates the
effect of removing the launch-bound loop.
"""

import torch

from ace_kernels.common import Prepared, finalize, hard_edges, prepare, soft_assign
from ace_kernels.config import ACEConfig
from ace_kernels.registry import register


def _row_offsets(prep: Prepared, num_bins: int) -> torch.Tensor:
    """``[B, C, 1]``: first flat bin index of each (b, c) slice."""
    b, c = prep.probs.shape[:2]
    rows = torch.arange(b * c, device=prep.probs.device).view(b, c, 1)
    return rows * num_bins


def _hard(prep: Prepared, config: ACEConfig) -> torch.Tensor:
    p, y = prep.probs, prep.onehot
    assert y is not None
    b, c, _ = p.shape
    k = config.num_bins
    idx = torch.bucketize(p, hard_edges(k, p.device), right=config.right)
    flat = (idx + _row_offsets(prep, k)).flatten()
    size = b * c * k
    counts = torch.zeros(size, device=p.device).scatter_add(
        0, flat, torch.ones_like(p).flatten()
    )
    mean_p = torch.empty(size, device=p.device).scatter_reduce(
        0, flat, p.flatten(), reduce="mean", include_self=False
    )
    mean_gt = torch.empty(size, device=p.device).scatter_reduce(
        0, flat, y.flatten(), reduce="mean", include_self=False
    )
    # copy before masking, like the reference's slice assignment: scatter_reduce
    # needs its untouched output for backward
    mean_p, mean_gt = mean_p.clone(), mean_gt.clone()
    mean_p[counts == 0] = torch.nan
    mean_gt[counts == 0] = torch.nan
    return (mean_p - mean_gt).abs().view(b, c, k)


def _soft(prep: Prepared, config: ACEConfig) -> torch.Tensor:
    p, y = prep.probs, prep.onehot
    assert y is not None
    b, c, _ = p.shape
    k = config.num_bins
    a = soft_assign(p, k, config.right)
    left_idx, right_idx, left_w, right_w = a.left_idx, a.right_idx, a.left_w, a.right_w

    offsets = _row_offsets(prep, k)
    left_flat = (left_idx + offsets).flatten()
    right_flat = (right_idx + offsets).flatten()
    size = b * c * k

    def bin_sum(values_left: torch.Tensor, values_right: torch.Tensor) -> torch.Tensor:
        zeros = torch.zeros(size, device=p.device)
        left = zeros.scatter_add(0, left_flat, values_left.flatten())
        right = zeros.scatter_add(0, right_flat, values_right.flatten())
        return left + right

    weights = bin_sum(left_w, right_w)
    sum_p = bin_sum(left_w * p, right_w * p)
    sum_gt = bin_sum(left_w * y, right_w * y)
    mean_p = sum_p / weights
    mean_gt = sum_gt / weights
    empty = weights < config.empty_weight
    mean_p[empty] = torch.nan
    mean_gt[empty] = torch.nan
    return (mean_p - mean_gt).abs().view(b, c, k)


@register("v1_vectorized", description="Reference ops, B x C loop removed")
def v1_vectorized(
    input: torch.Tensor, target: torch.Tensor, config: ACEConfig
) -> torch.Tensor:
    prep = prepare(input, target, config)
    errors = _hard(prep, config) if config.binning == "hard" else _soft(prep, config)
    return finalize(errors, prep, config)
