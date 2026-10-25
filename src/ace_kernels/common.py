"""Pre/post-processing shared by the optimized implementations.

Reproduces the reference semantics (``ace_reference.HardL1ACELoss.forward``):
activation, optional one-hot target, background removal, empty-class masking,
class weights and reduction. Only the binning in between differs per impl.
"""

from dataclasses import dataclass

import torch

from ace_kernels.config import ACEConfig

EPS32 = torch.finfo(torch.float32).eps


def hard_edges(num_bins: int, device: torch.device) -> torch.Tensor:
    """Upper bin edges, bit-identical to the reference (float32 linspace)."""
    edges = torch.linspace(0.0, 1.0 + EPS32, num_bins + 1, device=device)
    return edges[1:]


def soft_half_boundaries(num_bins: int, device: torch.device) -> torch.Tensor:
    """Bin centres and edges interleaved, bit-identical to the reference."""
    return torch.linspace(0.0, 1.0 + EPS32, 2 * num_bins + 1, device=device)


@dataclass
class SoftAssignment:
    """Soft binning of each probability to its two neighbouring bin centres.

    Same arithmetic as the reference (including its +-10000 sentinels outside
    the first/last centre, where both neighbours are the same end bin).
    """

    left_idx: torch.Tensor  # bin index per probability, int64
    right_idx: torch.Tensor
    left_w: torch.Tensor  # interpolation weights, differentiable w.r.t. p
    right_w: torch.Tensor


def soft_assign(p: torch.Tensor, num_bins: int, right: bool) -> SoftAssignment:
    half = soft_half_boundaries(num_bins, p.device)
    half_idx = torch.bucketize(p, half[1:], right=right)
    left_idx = torch.clamp(half_idx - 1, min=0) // 2
    right_idx = torch.clamp(half_idx + 1, max=2 * num_bins - 1) // 2
    sentinel = torch.tensor([10000.0], device=p.device)
    repl = torch.cat((-sentinel, half[1::2].repeat_interleave(2), sentinel))
    left_dist = p - repl[half_idx]
    right_dist = repl[half_idx + 2] - p
    sum_dist = left_dist + right_dist
    return SoftAssignment(
        left_idx, right_idx, right_dist / sum_dist, left_dist / sum_dist
    )


@dataclass
class Prepared:
    """Inputs after activation / one-hot / background handling.

    probs:  ``[B, C, N]`` probabilities.
    onehot: ``[B, C, N]`` float targets, or None when ``labels`` is used.
    labels: ``[B, N]`` class index per voxel, already shifted when the background
            channel was dropped (dropped voxels get -1), or None.
    """

    probs: torch.Tensor
    onehot: torch.Tensor | None
    labels: torch.Tensor | None
    spatial_ndim: int


def _activate(x: torch.Tensor, config: ACEConfig) -> torch.Tensor:
    if config.activation == "sigmoid":
        return torch.sigmoid(x)
    if config.activation == "softmax" and x.shape[1] > 1:
        return torch.softmax(x, 1)
    return x  # "none", or softmax on one channel (the reference ignores it)


def prepare(
    input: torch.Tensor,
    target: torch.Tensor,
    config: ACEConfig,
    keep_labels: bool = False,
) -> Prepared:
    """Apply the reference's input handling.

    With ``keep_labels`` and an index target (``to_onehot_y``), the one-hot tensor
    is never materialized; ``labels`` is returned instead.
    """
    if input.dim() < 3:
        raise ValueError(f"input must be [B, C, *spatial], got {tuple(input.shape)}")
    probs = _activate(input, config)
    n_ch = input.shape[1]
    drop_bg = not config.include_background and n_ch > 1
    use_labels = config.to_onehot_y and n_ch > 1

    if use_labels and keep_labels:
        labels = target.reshape(target.shape[0], -1).long()
        if drop_bg:
            labels = labels - 1  # background (0) -> -1: belongs to no channel
            probs = probs[:, 1:]
        if labels.shape[-1] != probs[0, 0].numel():
            raise AssertionError("target and input spatial sizes differ")
        return Prepared(probs.flatten(2).float(), None, labels, input.dim() - 2)

    if use_labels:
        idx = target.long()
        onehot = torch.zeros_like(probs, dtype=torch.float32).scatter_(1, idx, 1.0)
    else:
        onehot = target.float()
    if drop_bg:
        probs, onehot = probs[:, 1:], onehot[:, 1:]
    if onehot.shape != probs.shape:
        raise AssertionError(
            f"ground truth has different shape ({tuple(onehot.shape)}) from input "
            f"({tuple(probs.shape)})"
        )
    return Prepared(probs.flatten(2).float(), onehot.flatten(2), None, input.dim() - 2)


def nonempty_classes(prep: Prepared) -> torch.Tensor:
    """``[B, C]`` bool: the class occurs in the target (reference: sum > 0)."""
    if prep.onehot is not None:
        return prep.onehot.sum(-1) > 0
    assert prep.labels is not None
    b, c = prep.probs.shape[:2]
    valid = prep.labels >= 0
    counts = torch.zeros(b, c, device=prep.labels.device)
    counts.scatter_add_(1, prep.labels.clamp_min(0), valid.float())
    return counts > 0


def finalize(
    per_bin_error: torch.Tensor, prep: Prepared, config: ACEConfig
) -> torch.Tensor:
    """``[B, C, K]`` |mean_p - mean_gt| (NaN = empty bin) -> reduced loss."""
    f = torch.nanmean(per_bin_error, dim=-1)
    if config.ignore_empty_classes:
        f = f * nonempty_classes(prep).float()
    n_ch = f.shape[1]
    if config.weight is not None and n_ch != 1:
        # checked on the Python config values: no device -> host sync
        weights = (
            [config.weight] * n_ch
            if isinstance(config.weight, float)
            else list(config.weight)
        )
        if len(weights) != n_ch:
            raise ValueError(
                "the length of the `weight` sequence should be the same as the "
                "number of classes (excluding background if include_background=False)"
            )
        if min(weights) < 0:
            raise ValueError(
                "the value/values of the `weight` should be no less than 0."
            )
        f = f * torch.tensor(weights, dtype=f.dtype, device=f.device)
    if config.reduction == "mean":
        return f.mean()
    if config.reduction == "sum":
        return f.sum()
    return f.view(*f.shape[:2], *([1] * prep.spatial_ndim))
