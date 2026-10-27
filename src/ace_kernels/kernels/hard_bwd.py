"""Hard-binning backward in one Triton pass (docs/derivation.md).

Per voxel and channel: ``dL/dp = g_bc * coef[b, c, bin(p)]`` with the tiny
``coef`` table from the forward, then the activation backward across channels in
registers (softmax: ``dx_c = p_c (gp_c - sum_c' p_c' gp_c')``). Reads the
probabilities once and writes the input gradient once: no voxel-sized
temporaries (bin indices, gathers) as in the PyTorch backward.

The probabilities come from the same ``torch.softmax`` as the forward's, so every
voxel lands in the same bin in both passes (a Triton re-implementation of exp could
differ by 1 ulp and move exact-edge voxels).
"""

import torch
import triton
import triton.language as tl

from ace_kernels.config import ACEConfig
from ace_kernels.kernels.act import activate_tl, softmax_stats
from ace_kernels.kernels.bins import edge_table, sorted_count

ACT_NONE, ACT_SOFTMAX, ACT_SIGMOID = 0, 1, 2

_CONFIGS = [
    triton.Config({"BLOCK": block}, num_warps=warps)
    for block in (128, 256, 512, 1024)
    for warps in (4, 8)
]


@triton.autotune(configs=_CONFIGS, key=["N", "C", "K"])
@triton.jit
def _hard_bwd_kernel(
    x_ptr,  # [B, C, N] probabilities, or logits if FROM_LOGITS (any float dtype)
    coef_ptr,  # [B, C - OFFSET, K] float32: sign(d) / (K_ne W), 0 for empty bins
    g_ptr,  # [B, C - OFFSET] float32: dL/df_bc
    edges_ptr,  # [K] float32
    dx_ptr,  # [B, C, N] float32 output: dL/d(logits)
    N,
    C,
    OFFSET: tl.constexpr,  # 1 when the background channel is excluded from the loss
    K: tl.constexpr,
    CP: tl.constexpr,  # next power of two >= C
    RIGHT: tl.constexpr,
    ACT: tl.constexpr,
    FROM_LOGITS: tl.constexpr,  # recompute the activation in-kernel (fused)
    DIRECT: tl.constexpr,  # O(1) bin index, as the forward (bins.sorted_count)
    PRECISE: tl.constexpr,  # libdevice exp, as the forward
    BLOCK: tl.constexpr,
):
    # voxel blocks on grid axis 0 (up to 2^31 - 1 programs); axis 1 is limited to
    # 65535, which 256^3 volumes exceed (measured: "invalid argument" on a T4)
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

    if DIRECT:
        idx = sorted_count(p, edges_ptr, K, RIGHT)
    else:
        idx = tl.zeros([CP, BLOCK], dtype=tl.int32)
        for k in tl.static_range(K):
            edge = tl.load(edges_ptr + k)
            if RIGHT:
                idx += (p >= edge).to(tl.int32)
            else:
                idx += (p > edge).to(tl.int32)

    c_loss = C - OFFSET  # channels that enter the loss
    row = chan - OFFSET
    in_loss = (row >= 0) & chan_ok
    coef = tl.load(
        coef_ptr + (b * c_loss + row)[:, None] * K + idx,
        mask=in_loss[:, None] & valid[None, :],
        other=0.0,
    )
    g = tl.load(g_ptr + b * c_loss + row, mask=in_loss, other=0.0)
    gp = coef * g[:, None]  # dL/dp, 0 for the dropped background channel

    if ACT == 1:  # softmax over channels
        dx = p * (gp - tl.sum(p * gp, axis=0)[None, :])
    elif ACT == 2:  # sigmoid
        dx = gp * p * (1.0 - p)
    else:
        dx = gp
    tl.store(dx_ptr + ptrs, dx.to(dx_ptr.dtype.element_ty), mask=mask)


def hard_input_grad_triton(
    probs: torch.Tensor,
    coef: torch.Tensor,
    g: torch.Tensor,
    config: ACEConfig,
    from_logits: bool = False,
    direct: bool = False,
    precise: bool = False,
) -> torch.Tensor:
    """``dL/d(input)`` [B, C, N] from all-channel ``probs`` and the forward's ``coef``.

    With ``from_logits``, ``probs`` holds the logits (fp32/bf16/fp16) and the
    activation is recomputed in-kernel; the gradient has the logits' dtype.

    ``coef`` and ``g`` cover the loss channels (C - 1 when the background is
    dropped). Softmax with one channel is the identity, as in the reference.
    """
    probs = probs.contiguous()
    if not probs.is_cuda or (probs.dtype != torch.float32 and not from_logits):
        raise ValueError("hard_input_grad_triton needs float32 CUDA probabilities")
    b, c, n = probs.shape
    k = config.num_bins
    offset = c - coef.shape[1]
    if config.activation == "sigmoid":
        act = ACT_SIGMOID
    elif config.activation == "softmax" and c > 1:
        act = ACT_SOFTMAX
    else:
        act = ACT_NONE
    dx = torch.empty_like(probs)

    def grid(meta: dict[str, int]) -> tuple[int, int]:
        return (triton.cdiv(n, meta["BLOCK"]), b)

    _hard_bwd_kernel[grid](
        probs,
        coef.contiguous().float(),
        g.contiguous().float(),
        edge_table(k, probs.device),
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
    )
    return dx
