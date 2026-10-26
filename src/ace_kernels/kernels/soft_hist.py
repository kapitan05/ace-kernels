"""Soft-binning statistics with per-block private histograms (Triton).

Each probability contributes to its two neighbouring bin centres with linear
interpolation weights (docs/derivation.md), exactly as ``common.soft_assign``:

- half-bin index ``h`` = number of half boundaries ``< p`` (``<= p`` if ``right``),
  counted against the reference's float32 boundary tensor: bit-identical bins;
- left / right bins ``max(h - 1, 0) // 2`` and ``min(h + 1, 2K - 1) // 2``;
- distances to the replicated centres table (with the reference's +-10000
  sentinels), weights ``lw = right_dist / S``, ``rw = left_dist / S`` with IEEE
  round-to-nearest division (``tl.div_rn``) like PyTorch's ``/``.

A block's contributions are reduced into private ``[K]`` histograms through the
``[BLOCK, K]`` weight matrix ``hit_left * lw + hit_right * rw`` (end bins, where
left == right, simply add up), then one atomic add per bin and statistic.
"""

import torch
import triton
import triton.language as tl

from ace_kernels.binning import BinStats
from ace_kernels.common import Prepared, soft_half_boundaries
from ace_kernels.config import ACEConfig
from ace_kernels.kernels.act import (
    ACT_NONE,
    act_code,
    activate_tl,
    channel_pow2,
    softmax_stats,
)

_CONFIGS = [
    triton.Config({"BLOCK": block}, num_warps=warps)
    for block in (128, 256, 512)
    for warps in (4, 8)
]


@triton.jit
def soft_assign_tl(p, half_ptr, repl_ptr, K: tl.constexpr, RIGHT: tl.constexpr):
    """Left/right bins, weights and span S for probabilities ``p`` (any shape)."""
    h = tl.zeros(p.shape, dtype=tl.int32)
    for j in tl.static_range(2 * K):  # half boundaries half[1:], 2K values
        bound = tl.load(half_ptr + 1 + j)
        if RIGHT:
            h += (p >= bound).to(tl.int32)
        else:
            h += (p > bound).to(tl.int32)
    left = tl.maximum(h - 1, 0) // 2
    right = tl.minimum(h + 1, 2 * K - 1) // 2
    left_dist = p - tl.load(repl_ptr + h)
    right_dist = tl.load(repl_ptr + h + 2) - p
    span = left_dist + right_dist
    lw = tl.div_rn(right_dist, span)
    rw = tl.div_rn(left_dist, span)
    return left, right, lw, rw, span


@triton.autotune(configs=_CONFIGS, key=["N", "C", "K"], reset_to_zero=["out_ptr"])
@triton.jit
def _soft_hist_kernel(
    x_ptr,  # [B, C, N] probabilities (ACT none) or logits, any float dtype
    labels_ptr,  # [B, N] int64 class per voxel, indexing x's channels; -1: none
    onehot_ptr,  # [B, C, N] float targets aligned with x; if not USE_LABELS
    half_ptr,  # [2K + 1] float32 half boundaries (reference linspace)
    repl_ptr,  # [2K + 2] float32: -10000, c0, c0, c1, c1, ..., 10000
    out_ptr,  # [3, B, C - OFFSET, K] float32, zeroed: weight sum, sum wp, sum wy
    N,
    C,
    BCK,  # B * (C - OFFSET) * K
    K: tl.constexpr,
    KP: tl.constexpr,
    CP: tl.constexpr,
    OFFSET: tl.constexpr,
    ACT: tl.constexpr,
    RIGHT: tl.constexpr,
    USE_LABELS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(1)  # voxel blocks on axis 0: axis 1 is limited to 65535
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offs < N
    bins = tl.arange(0, KP)
    bin_ok = bins < K
    if USE_LABELS:
        label = tl.load(labels_ptr + b * N + offs, mask=valid, other=-1)
    m = tl.zeros([BLOCK], dtype=tl.float32)
    s = tl.full([BLOCK], 1.0, dtype=tl.float32)
    if ACT == 1:
        m, s = softmax_stats(x_ptr, b, offs, valid, N, C, CP)
    for c in range(OFFSET, C):
        base = (b * C + c) * N
        x = tl.load(x_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)
        p = tl.where(valid, activate_tl(x, m, s, ACT), 0.0)
        left, right, lw, rw, _ = soft_assign_tl(p, half_ptr, repl_ptr, K, RIGHT)
        lw = tl.where(valid, lw, 0.0)
        rw = tl.where(valid, rw, 0.0)
        if USE_LABELS:
            y = ((label == c) & valid).to(tl.float32)
        else:
            y = tl.load(onehot_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)
        w = (left[:, None] == bins[None, :]).to(tl.float32) * lw[:, None] + (
            right[:, None] == bins[None, :]
        ).to(tl.float32) * rw[:, None]
        out = out_ptr + (b * (C - OFFSET) + c - OFFSET) * K + bins
        tl.atomic_add(out, tl.sum(w, axis=0), mask=bin_ok)
        tl.atomic_add(out + BCK, tl.sum(w * p[:, None], axis=0), mask=bin_ok)
        tl.atomic_add(out + 2 * BCK, tl.sum(w * y[:, None], axis=0), mask=bin_ok)


def soft_tables(
    num_bins: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Half boundaries and the replicated-centres table, as in the reference."""
    half = soft_half_boundaries(num_bins, device).contiguous()
    sentinel = half.new_full((1,), 10000.0)
    repl = torch.cat((-sentinel, half[1::2].repeat_interleave(2), sentinel))
    return half, repl.contiguous()


def soft_stats_triton(prep: Prepared, config: ACEConfig) -> BinStats:
    """Same result as ``binning.soft_stats`` (float32, CUDA)."""
    p = prep.probs.contiguous()
    if p.dtype != torch.float32 or not p.is_cuda:
        raise ValueError("soft_stats_triton needs float32 CUDA probabilities")
    b, c, n = p.shape
    k = config.num_bins
    out = torch.zeros((3, b, c, k), device=p.device, dtype=torch.float32)
    use_labels = prep.labels is not None
    labels = prep.labels.contiguous() if prep.labels is not None else p
    onehot = p if prep.onehot is None else prep.onehot.contiguous().float()
    half, repl = soft_tables(k, p.device)

    def grid(meta: dict[str, int]) -> tuple[int, int]:
        return (triton.cdiv(n, meta["BLOCK"]), b)

    _soft_hist_kernel[grid](
        p,
        labels,
        onehot,
        half,
        repl,
        out,
        n,
        c,
        b * c * k,
        K=k,
        KP=max(2, triton.next_power_of_2(k)),
        CP=2,
        OFFSET=0,
        ACT=ACT_NONE,
        RIGHT=config.right,
        USE_LABELS=use_labels,
    )
    weight, sum_p, sum_y = out.unbind(0)
    return BinStats(weight, sum_p, sum_y, weight < config.empty_weight)


def soft_stats_fused(
    x: torch.Tensor, target: torch.Tensor, config: ACEConfig
) -> BinStats:
    """Soft statistics straight from logits ``x`` [B, C, N]; see hard_stats_fused."""
    b, c, n = x.shape
    k = config.num_bins
    offset = 1 if (not config.include_background and c > 1) else 0
    out = torch.zeros((3, b, c - offset, k), device=x.device, dtype=torch.float32)
    use_labels = target.dim() == 2
    half, repl = soft_tables(k, x.device)

    def grid(meta: dict[str, int]) -> tuple[int, int]:
        return (triton.cdiv(n, meta["BLOCK"]), b)

    _soft_hist_kernel[grid](
        x,
        target if use_labels else x,
        x if use_labels else target,
        half,
        repl,
        out,
        n,
        c,
        b * (c - offset) * k,
        K=k,
        KP=max(2, triton.next_power_of_2(k)),
        CP=channel_pow2(c),
        OFFSET=offset,
        ACT=act_code(config, c),
        RIGHT=config.right,
        USE_LABELS=use_labels,
    )
    weight, sum_p, sum_y = out.unbind(0)
    return BinStats(weight, sum_p, sum_y, weight < config.empty_weight)
