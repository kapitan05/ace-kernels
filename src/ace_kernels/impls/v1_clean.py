"""v1_clean: v1_vectorized plus removal of avoidable work (still default autograd).

Changes on top of v1_vectorized, each removing memory traffic or a host sync:
- Per-bin statistics are plain sums (``scatter_add``): sum of p, sum of y and the
  count / weight sum. Means are taken on the tiny ``[B, C, K]`` result. Hard
  binning no longer uses ``scatter_reduce(mean)``.
- With an index target (``to_onehot_y``) no one-hot tensor is built: each voxel
  adds its bin weight only to the row of its true class (gather + scatter on
  ``[B, N]`` instead of ``[B, C, N]``).
- Empty bins are masked with ``torch.where`` instead of boolean-index assignment;
  divisions are guarded so empty bins give zero, not NaN, gradients. No
  host<->device copies or syncs anywhere, so fwd+bwd is CUDA-graph capturable
  (tests/test_sync.py).
"""

import torch

from ace_kernels.common import (
    Prepared,
    finalize,
    hard_edges,
    prepare,
    soft_assign,
)
from ace_kernels.config import ACEConfig
from ace_kernels.registry import register


def _rows(prep: Prepared) -> torch.Tensor:
    """``[B, C, 1]`` flat row index ``b * C + c`` of each slice."""
    b, c = prep.probs.shape[:2]
    return torch.arange(b * c, device=prep.probs.device).view(b, c, 1)


def _true_class(prep: Prepared) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """For index targets: true-class channel ``[B, 1, N]``, its flat row, validity.

    Voxels whose class is not among the channels (background dropped) get
    validity 0 and contribute nothing.
    """
    assert prep.labels is not None
    b, c = prep.probs.shape[:2]
    valid = (prep.labels >= 0).unsqueeze(1).float()
    channel = prep.labels.clamp_min(0).unsqueeze(1)
    batch = torch.arange(b, device=channel.device).view(b, 1, 1)
    return channel, batch * c + channel, valid


def _masked_error(
    sum_p: torch.Tensor, sum_gt: torch.Tensor, norm: torch.Tensor, empty: torch.Tensor
) -> torch.Tensor:
    safe = torch.where(empty, torch.ones_like(norm), norm)
    error = (sum_p / safe - sum_gt / safe).abs()
    return torch.where(empty, torch.full_like(error, torch.nan), error)


def _hard(prep: Prepared, config: ACEConfig) -> torch.Tensor:
    p = prep.probs
    b, c, _ = p.shape
    k = config.num_bins
    size = b * c * k
    idx = torch.bucketize(p, hard_edges(k, p.device), right=config.right)
    flat = (idx + _rows(prep) * k).flatten()
    zeros = p.new_zeros(size)
    counts = zeros.scatter_add(0, flat, p.new_ones(()).expand(flat.shape))
    sum_p = zeros.scatter_add(0, flat, p.flatten())
    if prep.onehot is not None:
        sum_gt = zeros.scatter_add(0, flat, prep.onehot.flatten())
    else:
        channel, row, valid = _true_class(prep)
        true_idx = idx.gather(1, channel)
        sum_gt = zeros.scatter_add(0, (true_idx + row * k).flatten(), valid.flatten())
    error = _masked_error(sum_p, sum_gt, counts, counts == 0)
    return error.view(b, c, k)


def _soft(prep: Prepared, config: ACEConfig) -> torch.Tensor:
    p = prep.probs
    b, c, _ = p.shape
    k = config.num_bins
    size = b * c * k
    a = soft_assign(p, k, config.right)
    offsets = _rows(prep) * k
    left = (a.left_idx + offsets).flatten()
    right = (a.right_idx + offsets).flatten()
    zeros = p.new_zeros(size)

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
        channel, row, valid = _true_class(prep)
        sum_gt = scatter(
            (a.left_idx.gather(1, channel) + row * k).flatten(),
            (a.left_w.gather(1, channel) * valid).flatten(),
            (a.right_idx.gather(1, channel) + row * k).flatten(),
            (a.right_w.gather(1, channel) * valid).flatten(),
        )
    error = _masked_error(sum_p, sum_gt, weights, weights < config.empty_weight)
    return error.view(b, c, k)


@register("v1_clean", description="Vectorized + plain sums, no one-hot, no host sync")
def v1_clean(
    input: torch.Tensor, target: torch.Tensor, config: ACEConfig
) -> torch.Tensor:
    prep = prepare(input, target, config, keep_labels=True)
    errors = _hard(prep, config) if config.binning == "hard" else _soft(prep, config)
    return finalize(errors, prep, config)


_compiled = torch.compile(v1_clean)


@register(
    "v1_clean_compile",
    description="v1_clean under torch.compile (default mode)",
    slow=True,
)
def v1_clean_compile(
    input: torch.Tensor, target: torch.Tensor, config: ACEConfig
) -> torch.Tensor:
    loss: torch.Tensor = _compiled(input, target, config)
    return loss
