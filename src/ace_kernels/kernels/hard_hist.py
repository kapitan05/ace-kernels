"""Hard-binning statistics with per-block private histograms (Triton).

Replaces ``binning.hard_stats``' three contended ``scatter_add`` calls (millions of
voxels adding into K addresses per slice) with: each program bins a block of
voxels for every channel, reduces its block into a private ``[K]`` histogram in
registers (``[BLOCK, K]`` compare mask + ``tl.sum``), and issues one atomic add per
bin and statistic. Atomics per slice drop from N to ``3 K * N / BLOCK``.

Bins are counted against the reference's float32 edge tensor with the same
comparison as ``torch.bucketize``: ``right=False`` -> number of edges < p,
``right=True`` -> number of edges <= p. Bit-identical bin assignment.

Sprint 11 options (constexpr, ``bins.py``): ``DIRECT`` computes that count in O(1)
instead of K compares; ``DOT`` reduces the block with a tensor-core matmul
(one-hot ``[KP, BLOCK]`` @ values ``[BLOCK, 16]``) instead of three
``[BLOCK, KP]`` multiply-and-sum passes. ``DOT`` needs p in [0, 1] and 0/1 targets
(activation + class-index labels), where it is exact up to fp32 summation order.
"""

import torch
import triton
import triton.language as tl

from ace_kernels.binning import BinStats
from ace_kernels.common import Prepared
from ace_kernels.config import ACEConfig
from ace_kernels.kernels.act import (
    ACT_NONE,
    act_code,
    activate_tl,
    channel_pow2,
    softmax_stats,
)
from ace_kernels.kernels.bins import (
    DOT_COLS,
    column,
    dot_bins,
    edge_table,
    join3,
    prune_dot_configs,
    put3,
    sorted_count,
)

_CONFIGS = [
    triton.Config({"BLOCK": block}, num_warps=warps)
    for block in (128, 256, 512)
    for warps in (4, 8)
]


@triton.autotune(
    configs=_CONFIGS,
    key=["N", "C", "K", "DOT"],
    reset_to_zero=["out_ptr"],
    prune_configs_by={"early_config_prune": prune_dot_configs},
)
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
    DIRECT: tl.constexpr,  # O(1) bin index (bins.sorted_count)
    DOT: tl.constexpr,  # tensor-core reduction (bins.put3 / join3)
    PRECISE: tl.constexpr,  # libdevice exp in the activation
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
        m, s = softmax_stats(x_ptr, b, offs, valid, N, C, CP, PRECISE)
    for c in range(OFFSET, C):
        base = (b * C + c) * N
        x = tl.load(x_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)
        p = tl.where(valid, activate_tl(x, m, s, ACT, PRECISE), 0.0)
        if DIRECT:
            idx = sorted_count(p, edges_ptr, K, RIGHT)
        else:
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
        if DOT:
            col = tl.arange(0, DOT_COLS)[None, :]
            vals = tl.where(col == 0, 1.0, tl.where(col == 1, y[:, None], 0.0))
            vals = put3(vals, col, 2, p)
            onehot_t = (bins[:, None] == idx[None, :]) & valid[None, :]
            acc = tl.dot(onehot_t.to(tl.float16), vals.to(tl.float16))  # [KP, 16]
            count = column(acc, col, 0)
            sum_y = column(acc, col, 1)
            sum_p = join3(acc, col, 2)
        else:
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
    edges = edge_table(k, p.device)

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
        DIRECT=False,
        DOT=False,
        PRECISE=False,
    )
    count, sum_p, sum_y = out.unbind(0)
    return BinStats(count, sum_p, sum_y, count == 0)


def hard_stats_fused(
    x: torch.Tensor,
    target: torch.Tensor,
    config: ACEConfig,
    direct: bool = False,
    dot: bool = False,
    precise: bool = False,
) -> BinStats:
    """``hard_hist_fused`` as ``BinStats`` (count, sum p, sum y, empty)."""
    count, sum_p, sum_y = hard_hist_fused(
        x, target, config, direct, dot, precise
    ).unbind(0)
    return BinStats(count, sum_p, sum_y, count == 0)


def hard_hist_fused(
    x: torch.Tensor,
    target: torch.Tensor,
    config: ACEConfig,
    direct: bool = False,
    dot: bool = False,
    precise: bool = False,
) -> torch.Tensor:
    """``[3, B, C', K]`` count, sum p, sum y straight from logits ``x`` [B, C, N]
    (activation in-kernel).

    ``target``: ``[B, N]`` class indices (original numbering) when ``to_onehot_y``
    and C > 1, else ``[B, C, N]`` targets. Covers the loss channels (C - 1 when the
    background is dropped). ``direct`` / ``dot`` / ``precise``: Sprint 11 options
    (``dot`` is used only where it is exact: activation and class-index labels).
    """
    b, c, n = x.shape
    k = config.num_bins
    offset = 1 if (not config.include_background and c > 1) else 0
    out = torch.zeros((3, b, c - offset, k), device=x.device, dtype=torch.float32)
    use_labels = target.dim() == 2
    act = act_code(config, c)
    dot = dot and use_labels and act != ACT_NONE

    def grid(meta: dict[str, int]) -> tuple[int, int]:
        return (triton.cdiv(n, meta["BLOCK"]), b)

    _hard_hist_kernel[grid](
        x,
        target if use_labels else x,
        x if use_labels else target,
        edge_table(k, x.device),
        out,
        n,
        c,
        b * (c - offset) * k,
        K=k,
        KP=dot_bins(k) if dot else max(2, triton.next_power_of_2(k)),
        CP=channel_pow2(c),
        OFFSET=offset,
        ACT=act,
        RIGHT=config.right,
        USE_LABELS=use_labels,
        DIRECT=direct,
        DOT=dot,
        PRECISE=precise,
    )
    return out
