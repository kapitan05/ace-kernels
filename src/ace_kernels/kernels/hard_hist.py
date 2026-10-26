"""Hard-binning statistics with per-block private histograms (Triton).

Replaces ``binning.hard_stats``' three contended ``scatter_add`` calls (millions of
voxels adding into K addresses per slice) with: each program bins a block of
voxels for every channel, reduces its block into a private ``[K]`` histogram in
registers (``[BLOCK, K]`` compare mask + ``tl.sum``), and issues one atomic add per
bin and statistic. Atomics per slice drop from N to ``3 K * N / BLOCK``.

Bins are counted against the reference's float32 edge tensor with the same
comparison as ``torch.bucketize``: ``right=False`` -> number of edges < p,
``right=True`` -> number of edges <= p. Bit-identical bin assignment.
"""

import torch
import triton
import triton.language as tl

from ace_kernels.binning import BinStats
from ace_kernels.common import Prepared, hard_edges
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


@triton.autotune(configs=_CONFIGS, key=["N", "C", "K"], reset_to_zero=["out_ptr"])
@triton.jit
def _hard_hist_kernel(
    x_ptr,  # [B, C, N] probabilities (ACT none) or logits, any float dtype
    labels_ptr,  # [B, N] int64 class per voxel, indexing x's channels; -1: none
    onehot_ptr,  # [B, C, N] float targets aligned with x; if not USE_LABELS
    edges_ptr,  # [K] float32 upper bin edges (reference linspace)
    out_ptr,  # [3, B, C - OFFSET, K] float32, zeroed: count, sum p, sum y
    N,
    C,  # channels of x
    BCK,  # B * (C - OFFSET) * K: stride between the three statistics
    K: tl.constexpr,
    KP: tl.constexpr,  # next power of two >= K (tl.arange needs a power of 2)
    CP: tl.constexpr,  # next power of two >= C (softmax tile)
    OFFSET: tl.constexpr,  # 1: channel 0 (background) is not in the loss
    ACT: tl.constexpr,  # activation applied in-kernel (act.py)
    RIGHT: tl.constexpr,
    USE_LABELS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # voxel blocks on grid axis 0 (up to 2^31 - 1 programs); axis 1 is limited to
    # 65535, which 256^3 volumes exceed (measured: "invalid argument" on a T4)
    b = tl.program_id(1)
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
        idx = tl.zeros([BLOCK], dtype=tl.int32)
        for k in tl.static_range(K):
            edge = tl.load(edges_ptr + k)
            if RIGHT:
                idx += (p >= edge).to(tl.int32)
            else:
                idx += (p > edge).to(tl.int32)
        if USE_LABELS:
            y = ((label == c) & valid).to(tl.float32)
        else:
            y = tl.load(onehot_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)
        hit = ((idx[:, None] == bins[None, :]) & valid[:, None]).to(tl.float32)
        count = tl.sum(hit, axis=0)
        sum_p = tl.sum(hit * p[:, None], axis=0)
        sum_y = tl.sum(hit * y[:, None], axis=0)
        out = out_ptr + (b * (C - OFFSET) + c - OFFSET) * K + bins
        tl.atomic_add(out, count, mask=bin_ok)
        tl.atomic_add(out + BCK, sum_p, mask=bin_ok)
        tl.atomic_add(out + 2 * BCK, sum_y, mask=bin_ok)


def hard_stats_triton(prep: Prepared, config: ACEConfig) -> BinStats:
    """Same result as ``binning.hard_stats`` (float32, CUDA)."""
    p = prep.probs.contiguous()
    if p.dtype != torch.float32 or not p.is_cuda:
        raise ValueError("hard_stats_triton needs float32 CUDA probabilities")
    b, c, n = p.shape
    k = config.num_bins
    out = torch.zeros((3, b, c, k), device=p.device, dtype=torch.float32)
    use_labels = prep.labels is not None
    labels = prep.labels.contiguous() if prep.labels is not None else p
    onehot = p if prep.onehot is None else prep.onehot.contiguous().float()
    edges = hard_edges(k, p.device).contiguous()

    def grid(meta: dict[str, int]) -> tuple[int, int]:
        return (triton.cdiv(n, meta["BLOCK"]), b)

    _hard_hist_kernel[grid](
        p,
        labels,
        onehot,
        edges,
        out,
        n,
        c,
        b * c * k,
        K=k,
        KP=max(2, triton.next_power_of_2(k)),  # arange of size 1 is degenerate
        CP=2,
        OFFSET=0,
        ACT=ACT_NONE,
        RIGHT=config.right,
        USE_LABELS=use_labels,
    )
    count, sum_p, sum_y = out.unbind(0)
    return BinStats(count, sum_p, sum_y, count == 0)


def hard_stats_fused(
    x: torch.Tensor, target: torch.Tensor, config: ACEConfig
) -> BinStats:
    """Hard statistics straight from logits ``x`` [B, C, N] (activation in-kernel).

    ``target``: ``[B, N]`` class indices (original numbering) when ``to_onehot_y``
    and C > 1, else ``[B, C, N]`` targets. Covers the loss channels (C - 1 when the
    background is dropped).
    """
    b, c, n = x.shape
    k = config.num_bins
    offset = 1 if (not config.include_background and c > 1) else 0
    out = torch.zeros((3, b, c - offset, k), device=x.device, dtype=torch.float32)
    use_labels = target.dim() == 2

    def grid(meta: dict[str, int]) -> tuple[int, int]:
        return (triton.cdiv(n, meta["BLOCK"]), b)

    _hard_hist_kernel[grid](
        x,
        target if use_labels else x,
        x if use_labels else target,
        hard_edges(k, x.device).contiguous(),
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
    count, sum_p, sum_y = out.unbind(0)
    return BinStats(count, sum_p, sum_y, count == 0)
