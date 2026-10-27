"""Sprint 11 building blocks: O(1) bin index and tensor-core histogram reduction.

``sorted_count`` replaces the O(K) compare loop: ``floor(p * n)`` is within one of
``torch.bucketize``'s index for the reference's float32 edge tables (entry j is
(j + 1) / n up to a few ulp), and two correction rounds against the same table make
it exact (tests check every float32 within 8 ulp of every edge).

``split3`` + ``tl.dot`` replace the ``[BLOCK, K]`` multiply-and-sum reduction: a
block's histogram is ``onehot[K, BLOCK] @ values[BLOCK, 16]``. The one-hot matrix is
exact in fp16, and each fp32 value is split into three fp16 parts, so the fp16
tensor-core products (fp32 accumulation) reproduce the fp32 sums.
"""

import functools

import torch
import triton
import triton.language as tl

from ace_kernels.common import hard_edges

DOT_COLS = tl.constexpr(16)  # value columns of the reduction matmul
# split scales 2^11 (fp16 significand bits) and 2^22 appear as literals below:
# Triton only reads module globals wrapped in tl.constexpr


@triton.jit
def _counted(p, t, RIGHT: tl.constexpr):
    if RIGHT:  # noqa: SIM108 (constexpr branch in a Triton function)
        hit = p >= t
    else:
        hit = p > t
    return hit


@triton.jit
def sorted_count(p, table_ptr, n: tl.constexpr, RIGHT: tl.constexpr):
    """Number of entries counted for ``p`` (``t < p``, or ``t <= p`` if ``RIGHT``)
    in a sorted table with entry j ~= (j + 1) / n: ``torch.bucketize``'s index.
    NaN -> 0 (no comparison holds), +inf -> n, as the compare loop."""
    g = tl.minimum(tl.maximum(p * n, 0.0), n * 1.0)
    g = tl.where(p == p, g, 0.0).to(tl.int32)  # non-negative: truncation = floor
    for _ in tl.static_range(2):
        up = tl.load(table_ptr + tl.minimum(g, n - 1))
        g += ((g < n) & _counted(p, up, RIGHT)).to(tl.int32)
        lo = tl.load(table_ptr + tl.maximum(g - 1, 0))
        g -= ((g > 0) & ~_counted(p, lo, RIGHT)).to(tl.int32)
    return g


@triton.jit
def split3(v):
    """fp32 ``v`` = a + b / 2^11 + c / 2^22 (error ~2^-33 |v|), a, b, c exact in fp16
    (returned as fp32)."""
    a = v.to(tl.float16).to(tl.float32)
    r = v - a
    b = (r * 2048.0).to(tl.float16).to(tl.float32)
    r = r - b * (1.0 / 2048.0)
    c = (r * 4194304.0).to(tl.float16).to(tl.float32)
    return a, b, c


@triton.jit
def put3(vals, col, j: tl.constexpr, v):
    """Write ``split3(v)`` ([BLOCK]) into columns j, j+1, j+2 of ``vals``."""
    a, b, c = split3(v)
    vals = tl.where(col == j, a[:, None], vals)
    vals = tl.where(col == j + 1, b[:, None], vals)
    return tl.where(col == j + 2, c[:, None], vals)


@triton.jit
def column(acc, col, j: tl.constexpr):
    """Column j of a ``[KP, DOT_COLS]`` accumulator: [KP]."""
    return tl.sum(tl.where(col == j, acc, 0.0), axis=1)


@triton.jit
def join3(acc, col, j: tl.constexpr):
    """Sum of a value written by ``put3`` at column j."""
    small = column(acc, col, j + 1) * (1.0 / 2048.0) + column(acc, col, j + 2) * (
        1.0 / 4194304.0
    )
    return column(acc, col, j) + small


def dot_bins(k: int) -> int:
    """Bin tile of the matmul path: power of two, >= 16 (tl.dot minimum)."""
    return max(16, int(triton.next_power_of_2(k)))


@functools.lru_cache(maxsize=64)
def edge_table(num_bins: int, device: torch.device) -> torch.Tensor:
    """``hard_edges``, built once per device: no linspace launch per call."""
    return hard_edges(num_bins, device).contiguous()


def prune_dot_configs(
    configs: list[triton.Config], named_args: dict[str, object], **kwargs: object
) -> list[triton.Config]:
    """Matmul path: drop tiles whose one-hot operand exceeds shared memory (T4:
    64 KiB; measured: BLOCK 512 x 64 bins needs 128 KiB)."""
    meta = {**named_args, **kwargs}
    if not meta.get("DOT"):
        return configs
    bins = int(str(meta["KP"]))
    fits = [c for c in configs if c.kwargs["BLOCK"] * bins <= 8192]
    return fits or configs[:1]
