"""Activations computed inside the kernels (Sprint 8: fused softmax).

Forward and backward kernels use these same functions on the same logits, so a
voxel always lands in the same bin in both passes. Values can differ from
``torch.softmax`` by ~1 ulp (different exp implementation): bins can differ from
the reference only for probabilities within ~1e-7 of a bin edge.
"""

import torch
import triton
import triton.language as tl

from ace_kernels.config import ACEConfig

ACT_NONE, ACT_SOFTMAX, ACT_SIGMOID = 0, 1, 2


def act_code(config: ACEConfig, channels: int) -> int:
    """Kernel activation id; softmax over one channel is the identity (reference)."""
    if config.activation == "sigmoid":
        return ACT_SIGMOID
    if config.activation == "softmax" and channels > 1:
        return ACT_SOFTMAX
    return ACT_NONE


@triton.jit
def softmax_stats(x_ptr, b, offs, valid, N, C, CP: tl.constexpr):
    """Per-voxel max and sum of exp over the C channels of a block: [BLOCK] each."""
    chan = tl.arange(0, CP)
    mask = (chan < C)[:, None] & valid[None, :]
    x = tl.load(
        x_ptr + (b * C + chan)[:, None] * N + offs[None, :],
        mask=mask,
        other=float("-inf"),
    ).to(tl.float32)
    m = tl.where(valid, tl.max(x, axis=0), 0.0)
    s = tl.sum(tl.exp(x - m[None, :]), axis=0)  # masked rows: exp(-inf) = 0
    return m, tl.where(valid, s, 1.0)


@triton.jit
def activate_tl(x, m, s, ACT: tl.constexpr):
    """Probability from logits ``x`` (``m``, ``s`` from ``softmax_stats``)."""
    if ACT == 1:
        p = tl.div_rn(tl.exp(x - m), s)
    elif ACT == 2:
        p = tl.div_rn(1.0, 1.0 + tl.exp(-x))
    else:
        p = x
    return p


def channel_pow2(channels: int) -> int:
    return max(2, int(triton.next_power_of_2(channels)))


def as_kernel_input(x: torch.Tensor) -> torch.Tensor:
    """``[B, C, N]`` contiguous view of logits/probabilities (fp32/bf16/fp16 kept)."""
    if x.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        x = x.float()
    return x.reshape(x.shape[0], x.shape[1], -1).contiguous()
