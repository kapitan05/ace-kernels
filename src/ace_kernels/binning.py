"""Per-bin statistics shared by the PyTorch implementations (v1_clean, v2).

For every (b, c) slice and bin k:
    W_k = sum_i w_ik          (count for hard binning, weight sum for soft)
    P_k = sum_i w_ik p_i
    G_k = sum_i w_ik y_i
stored flat as ``[B, C, K]``. Bins of slice ``r = b * C + c`` live at flat
indices ``[r * K, (r + 1) * K)`` so one ``scatter_add`` covers all slices.
"""

from dataclasses import dataclass

import torch

from ace_kernels.common import Prepared, hard_edges, soft_assign
from ace_kernels.config import ACEConfig


@dataclass
class BinStats:
    weight: torch.Tensor  # W, [B, C, K]
    sum_p: torch.Tensor  # P
    sum_gt: torch.Tensor  # G
    empty: torch.Tensor  # bool: excluded from the mean (count 0 / below empty_weight)


def rows(prep: Prepared) -> torch.Tensor:
    """``[B, C, 1]`` flat row index ``b * C + c`` of each slice."""
    b, c = prep.probs.shape[:2]
    return torch.arange(b * c, device=prep.probs.device).view(b, c, 1)


def true_class(prep: Prepared) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """For index targets: true-class channel ``[B, 1, N]``, its flat row, validity.

    Voxels whose class is not among the channels (background dropped) get
    validity 0 and contribute nothing.
    """
    assert prep.labels is not None
    b, c = prep.probs.shape[:2]
    valid = (prep.labels >= 0).unsqueeze(1).to(prep.probs.dtype)
    channel = prep.labels.clamp_min(0).unsqueeze(1)
    batch = torch.arange(b, device=channel.device).view(b, 1, 1)
    return channel, batch * c + channel, valid


def hard_bin_index(prep: Prepared, config: ACEConfig) -> torch.Tensor:
    """``[B, C, N]`` int64 bin of each probability (reference bucketize semantics)."""
    p = prep.probs
    edges = hard_edges(config.num_bins, p.device).to(p.dtype)
    return torch.bucketize(p, edges, right=config.right)


def hard_stats(prep: Prepared, config: ACEConfig) -> BinStats:
    p = prep.probs
    b, c, _ = p.shape
    k = config.num_bins
    idx = hard_bin_index(prep, config)
    flat = (idx + rows(prep) * k).flatten()
    zeros = p.new_zeros(b * c * k)
    counts = zeros.scatter_add(0, flat, p.new_ones(()).expand(flat.shape))
    sum_p = zeros.scatter_add(0, flat, p.flatten())
    if prep.onehot is not None:
        sum_gt = zeros.scatter_add(0, flat, prep.onehot.flatten())
    else:
        channel, row, valid = true_class(prep)
        true_idx = idx.gather(1, channel)
        sum_gt = zeros.scatter_add(0, (true_idx + row * k).flatten(), valid.flatten())
    shape = (b, c, k)
    counts = counts.view(shape)
    return BinStats(counts, sum_p.view(shape), sum_gt.view(shape), counts == 0)


def soft_stats(prep: Prepared, config: ACEConfig) -> BinStats:
    p = prep.probs
    b, c, _ = p.shape
    k = config.num_bins
    a = soft_assign(p, k, config.right)
    offsets = rows(prep) * k
    left = (a.left_idx + offsets).flatten()
    right = (a.right_idx + offsets).flatten()
    zeros = p.new_zeros(b * c * k)

    def scatter(
        idx_l: torch.Tensor,
        val_l: torch.Tensor,
        idx_r: torch.Tensor,
        val_r: torch.Tensor,
    ) -> torch.Tensor:
        # two scatters into one buffer; a single scatter over cat'ed left/right
        # would make autograd keep 2N-sized copies (measured: 15.8x vs 11x input)
        return zeros.scatter_add(0, idx_l, val_l).scatter_add(0, idx_r, val_r)

    weights = scatter(left, a.left_w.flatten(), right, a.right_w.flatten())
    sum_p = scatter(left, (a.left_w * p).flatten(), right, (a.right_w * p).flatten())
    if prep.onehot is not None:
        y = prep.onehot
        sum_gt = scatter(
            left, (a.left_w * y).flatten(), right, (a.right_w * y).flatten()
        )
    else:
        channel, row, valid = true_class(prep)
        sum_gt = scatter(
            (a.left_idx.gather(1, channel) + row * k).flatten(),
            (a.left_w.gather(1, channel) * valid).flatten(),
            (a.right_idx.gather(1, channel) + row * k).flatten(),
            (a.right_w.gather(1, channel) * valid).flatten(),
        )
    shape = (b, c, k)
    weights = weights.view(shape)
    return BinStats(
        weights,
        sum_p.view(shape),
        sum_gt.view(shape),
        weights < config.empty_weight,
    )


def bin_stats(prep: Prepared, config: ACEConfig) -> BinStats:
    return (
        hard_stats(prep, config)
        if config.binning == "hard"
        else soft_stats(prep, config)
    )


def bin_difference(stats: BinStats) -> torch.Tensor:
    """``mean_p - mean_gt`` per bin (0 for empty bins), computed like the reference
    (two divisions, then subtract) so its sign at exact ties matches."""
    safe = torch.where(stats.empty, torch.ones_like(stats.weight), stats.weight)
    diff = stats.sum_p / safe - stats.sum_gt / safe
    return torch.where(stats.empty, torch.zeros_like(diff), diff)


def per_bin_error(stats: BinStats) -> torch.Tensor:
    """``[B, C, K]`` |mean_p - mean_gt|, NaN for empty bins (as the reference)."""
    error = bin_difference(stats).abs()
    return torch.where(stats.empty, torch.full_like(error, torch.nan), error)
