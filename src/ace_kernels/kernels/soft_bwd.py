"""Soft-binning backward in one Triton pass (docs/derivation.md).

Per voxel and loss channel, with the two bins l, r the probability contributes to:

    dL/dp = g_bc * ( coef[l] * (lw - (p - y - d[l]) / S)
                   + coef[r] * (rw + (p - y - d[r]) / S) )

(``w'_l = -1/S``, ``w'_r = +1/S``), then the activation backward across channels in
registers, as in ``hard_bwd``. Bins and weights come from the same
``soft_assign_tl`` as the forward kernel, on the same ``torch.softmax`` output.
"""

import torch
import triton
import triton.language as tl

from ace_kernels.config import ACEConfig
from ace_kernels.kernels.act import activate_tl, softmax_stats
from ace_kernels.kernels.hard_bwd import ACT_NONE, ACT_SIGMOID, ACT_SOFTMAX
from ace_kernels.kernels.soft_hist import soft_assign_tl, soft_tables

_CONFIGS = [
    triton.Config({"BLOCK": block}, num_warps=warps)
    for block in (128, 256, 512)
    for warps in (4, 8)
]


@triton.autotune(configs=_CONFIGS, key=["N", "C", "K"])
@triton.jit
def _soft_bwd_kernel(
    x_ptr,  # [B, C, N] probabilities, or logits if FROM_LOGITS (any float dtype)
    labels_ptr,  # [B, N] int64 original class index (unshifted); if USE_LABELS
    onehot_ptr,  # [B, C, N] float32 targets over ALL channels; if not USE_LABELS
    coef_ptr,  # [B, C - OFFSET, K] float32
    d_ptr,  # [B, C - OFFSET, K] float32: mean p - mean y per bin
    g_ptr,  # [B, C - OFFSET] float32: dL/df_bc
    half_ptr,
    repl_ptr,
    dx_ptr,  # [B, C, N] float32 output
    N,
    C,
    OFFSET: tl.constexpr,
    K: tl.constexpr,
    CP: tl.constexpr,
    RIGHT: tl.constexpr,
    ACT: tl.constexpr,
    FROM_LOGITS: tl.constexpr,  # recompute the activation in-kernel (fused)
    DIRECT: tl.constexpr,  # O(1) bin index, as the forward (bins.sorted_count)
    PRECISE: tl.constexpr,  # libdevice exp, as the forward
    USE_LABELS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(1)
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offs < N
    chan = tl.arange(0, CP)
    chan_ok = chan < C
    mask = chan_ok[:, None] & valid[None, :]
    ptrs = b * C * N + chan[:, None] * N + offs[None, :]
    if FROM_LOGITS:  # same functions as the forward kernels -> same bins
        m, s = softmax_stats(x_ptr, b, offs, valid, N, C, CP, PRECISE)
        x = tl.load(x_ptr + ptrs, mask=mask, other=0.0).to(tl.float32)
        p = tl.where(mask, activate_tl(x, m[None, :], s[None, :], ACT, PRECISE), 0.0)
    else:
        p = tl.load(x_ptr + ptrs, mask=mask, other=0.0).to(tl.float32)  # [CP, BLOCK]
    if USE_LABELS:
        label = tl.load(labels_ptr + b * N + offs, mask=valid, other=-1)
        y = (label[None, :] == chan[:, None]).to(tl.float32)
    else:
        y = tl.load(onehot_ptr + ptrs, mask=mask, other=0.0).to(tl.float32)

    left, right, lw, rw, span = soft_assign_tl(p, half_ptr, repl_ptr, K, RIGHT, DIRECT)
    inv_span = tl.div_rn(1.0, span)

    c_loss = C - OFFSET
    row = chan - OFFSET
    in_loss = (row >= 0) & chan_ok
    tab = (b * c_loss + row)[:, None] * K
    lmask = in_loss[:, None] & valid[None, :]
    coef_l = tl.load(coef_ptr + tab + left, mask=lmask, other=0.0)
    coef_r = tl.load(coef_ptr + tab + right, mask=lmask, other=0.0)
    d_l = tl.load(d_ptr + tab + left, mask=lmask, other=0.0)
    d_r = tl.load(d_ptr + tab + right, mask=lmask, other=0.0)
    g = tl.load(g_ptr + b * c_loss + row, mask=in_loss, other=0.0)

    resid = p - y
    gp = coef_l * (lw - inv_span * (resid - d_l)) + coef_r * (
        rw + inv_span * (resid - d_r)
    )
    gp = gp * g[:, None]

    if ACT == 1:
        dx = p * (gp - tl.sum(p * gp, axis=0)[None, :])
    elif ACT == 2:
        dx = gp * p * (1.0 - p)
    else:
        dx = gp
    tl.store(dx_ptr + ptrs, dx.to(dx_ptr.dtype.element_ty), mask=mask)


def soft_input_grad_triton(
    probs: torch.Tensor,
    target: torch.Tensor,
    coef: torch.Tensor,
    d: torch.Tensor,
    g: torch.Tensor,
    config: ACEConfig,
    from_logits: bool = False,
    direct: bool = False,
    precise: bool = False,
) -> torch.Tensor:
    """``dL/d(input)`` [B, C, N] for soft binning.

    ``target``: ``[B, 1, ...]`` class indices when ``config.to_onehot_y`` (and C > 1),
    else targets with the input's channels. ``coef``, ``d``, ``g`` cover the loss
    channels (C - 1 when the background is dropped). With ``from_logits``,
    ``probs`` holds the logits and the activation is recomputed in-kernel.
    """
    probs = probs.contiguous()
    if not probs.is_cuda or (probs.dtype != torch.float32 and not from_logits):
        raise ValueError("soft_input_grad_triton needs float32 CUDA probabilities")
    b, c, n = probs.shape
    k = config.num_bins
    offset = c - coef.shape[1]
    use_labels = config.to_onehot_y and c > 1
    labels = target.reshape(b, n).long().contiguous() if use_labels else probs
    onehot = probs if use_labels else target.reshape(b, c, n).float().contiguous()
    if config.activation == "sigmoid":
        act = ACT_SIGMOID
    elif config.activation == "softmax" and c > 1:
        act = ACT_SOFTMAX
    else:
        act = ACT_NONE
    half, repl = soft_tables(k, probs.device)
    dx = torch.empty_like(probs)

    def grid(meta: dict[str, int]) -> tuple[int, int]:
        return (triton.cdiv(n, meta["BLOCK"]), b)

    _soft_bwd_kernel[grid](
        probs,
        labels,
        onehot,
        coef.contiguous().float(),
        d.contiguous().float(),
        g.contiguous().float(),
        half,
        repl,
        dx,
        n,
        c,
        OFFSET=offset,
        K=k,
        CP=max(2, triton.next_power_of_2(c)),
        RIGHT=config.right,
        ACT=act,
        FROM_LOGITS=from_logits,
        DIRECT=direct,
        PRECISE=precise,
        USE_LABELS=use_labels,
    )
    return dx
