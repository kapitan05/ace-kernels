"""Sprint 11: the per-bin epilogue in one Triton program.

Replaces ~50 small PyTorch launches on ``[B, C, K]`` tensors (``coefficients``,
``reduce_classes``, the autograd-built reduction jacobian) with one launch: from
the bin statistics it computes, per (b, c) row and bin, ``d = mean p - mean y``,
``coef = sign(d) / (K_ne W)``, the per-class error ``f`` (mean over non-empty
bins, NaN if none), class presence (``sum_k G_k > 0``), the class weights and the
reduction, plus ``jac = d loss / d f``. Same formulas and operation order as
``v2_autograd.coefficients`` / ``common.reduce_classes`` (IEEE division); only the
summation order over bins and classes differs (fp32 rounding).
"""

import functools

import torch
import triton
import triton.language as tl

from ace_kernels.common import class_weights
from ace_kernels.config import ACEConfig


@triton.jit
def _epilogue_kernel(
    weight_ptr,  # [R, K] float32 bin weight (count for hard); R = B * C'
    sum_p_ptr,  # [R, K]
    sum_y_ptr,  # [R, K]
    cls_w_ptr,  # [C'] float32 class weights (ones when unweighted)
    coef_ptr,  # [R, K] out
    d_ptr,  # [R, K] out
    jac_ptr,  # [R] out: d loss / d f
    f_ptr,  # [R] out: f * present * class weight (reduction "none")
    loss_ptr,  # [1] out: mean / sum of f_ptr
    R,
    CL,  # loss channels C'
    EMPTY_W,  # soft: a bin is empty below this weight
    K: tl.constexpr,
    KP: tl.constexpr,
    RP: tl.constexpr,
    HARD: tl.constexpr,
    IGNORE_EMPTY: tl.constexpr,
    MEAN: tl.constexpr,
):
    rows = tl.arange(0, RP)
    bins = tl.arange(0, KP)
    row_ok = rows < R
    inside = row_ok[:, None] & (bins < K)[None, :]
    at = rows[:, None] * K + bins[None, :]
    weight = tl.load(weight_ptr + at, mask=inside, other=0.0)
    sum_p = tl.load(sum_p_ptr + at, mask=inside, other=0.0)
    sum_y = tl.load(sum_y_ptr + at, mask=inside, other=0.0)
    if HARD:  # noqa: SIM108 (constexpr branch in a Triton function)
        filled = weight != 0.0
    else:
        filled = weight >= EMPTY_W
    nonempty = filled & inside
    safe = tl.where(nonempty, weight, 1.0)
    d = tl.where(nonempty, tl.div_rn(sum_p, safe) - tl.div_rn(sum_y, safe), 0.0)
    k_ne = tl.sum(nonempty.to(tl.float32), axis=1)  # [RP]
    err = tl.sum(tl.where(nonempty, tl.abs(d), 0.0), axis=1)
    f = tl.where(k_ne > 0, tl.div_rn(err, tl.maximum(k_ne, 1.0)), float("nan"))
    sign = tl.where(d > 0, 1.0, tl.where(d < 0, -1.0, 0.0))
    denom = tl.maximum(k_ne, 1.0)[:, None] * safe
    coef = tl.where(nonempty, tl.div_rn(sign, denom), 0.0)
    tl.store(coef_ptr + at, coef, mask=inside)
    tl.store(d_ptr + at, d, mask=inside)

    cls_w = tl.load(cls_w_ptr + rows % CL, mask=row_ok, other=0.0)
    if IGNORE_EMPTY:
        present = (tl.sum(sum_y, axis=1) > 0).to(tl.float32)
        f = f * present
        scale = present * cls_w
    else:
        scale = cls_w
    f = f * cls_w
    total = tl.sum(tl.where(row_ok, f, 0.0), axis=0)
    if MEAN:
        total = total / R
        scale = scale / R
    tl.store(f_ptr + rows, f, mask=row_ok)
    tl.store(jac_ptr + rows, scale, mask=row_ok)
    tl.store(loss_ptr, total)


@functools.lru_cache(maxsize=64)
def _class_weight_tensor(
    weights: tuple[float, ...], device: torch.device
) -> torch.Tensor:
    """Cached on the device: no host->device copy per call (graph capturable)."""
    return torch.tensor(weights, dtype=torch.float32, device=device)


def epilogue_triton(
    hist: torch.Tensor, config: ACEConfig, spatial_ndim: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(loss, coef, d, jac) from the kernels' ``[3, B, C', K]`` statistics (weight,
    sum p, sum y), as v5's PyTorch path."""
    weight, sum_p, sum_y = hist.unbind(0)
    _, b, cl, k = hist.shape
    rows = b * cl
    weights = class_weights(config, cl) or [1.0] * cl
    coef = torch.empty_like(weight)
    d = torch.empty_like(weight)
    jac = torch.empty((b, cl), device=coef.device, dtype=torch.float32)
    f = torch.empty_like(jac)
    loss = torch.empty((), device=coef.device, dtype=torch.float32)
    _epilogue_kernel[(1,)](
        weight,
        sum_p,
        sum_y,
        _class_weight_tensor(tuple(weights), coef.device),
        coef,
        d,
        jac,
        f,
        loss,
        rows,
        cl,
        config.empty_weight,
        K=k,
        KP=max(2, int(triton.next_power_of_2(k))),
        RP=max(2, int(triton.next_power_of_2(rows))),
        HARD=config.binning == "hard",
        IGNORE_EMPTY=config.ignore_empty_classes,
        MEAN=config.reduction == "mean",
    )
    if config.reduction == "none":
        loss = f.view(b, cl, *([1] * spatial_ndim))
    return loss, coef, d, jac
