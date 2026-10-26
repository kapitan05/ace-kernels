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

_CONFIGS = [
    triton.Config({"BLOCK": block}, num_warps=warps)
    for block in (128, 256, 512)
    for warps in (4, 8)
]


@triton.autotune(configs=_CONFIGS, key=["N", "C", "K"], reset_to_zero=["out_ptr"])
@triton.jit
def _hard_hist_kernel(
    probs_ptr,  # [B, C, N] float32, contiguous
    labels_ptr,  # [B, N] int64 class per voxel (-1: no channel); if USE_LABELS
    onehot_ptr,  # [B, C, N] float32 targets; if not USE_LABELS
    edges_ptr,  # [K] float32 upper bin edges (reference linspace)
    out_ptr,  # [3, B, C, K] float32, zeroed: count, sum p, sum y
    N,
    C,
    BCK,  # B * C * K: stride between the three statistics
    K: tl.constexpr,
    KP: tl.constexpr,  # next power of two >= K (tl.arange needs a power of 2)
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
    for c in range(C):
        base = (b * C + c) * N
        p = tl.load(probs_ptr + base + offs, mask=valid, other=0.0)
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
            y = tl.load(onehot_ptr + base + offs, mask=valid, other=0.0)
        hit = ((idx[:, None] == bins[None, :]) & valid[:, None]).to(tl.float32)
        count = tl.sum(hit, axis=0)
        sum_p = tl.sum(hit * p[:, None], axis=0)
        sum_y = tl.sum(hit * y[:, None], axis=0)
        out = out_ptr + (b * C + c) * K + bins
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
        RIGHT=config.right,
        USE_LABELS=use_labels,
    )
    count, sum_p, sum_y = out.unbind(0)
    return BinStats(count, sum_p, sum_y, count == 0)
