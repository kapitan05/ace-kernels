"""v5_triton (Sprint 8): fused activation, logits -> loss -> d logits.

Both Triton kernels read the logits and compute softmax (or sigmoid) in registers
(``kernels/act.py``): the separate ``torch.softmax`` passes (31% of v4's GPU time,
run in the forward and again in the backward) and the probability tensor
disappear. Logits may be float32, bfloat16 or float16 (read natively, math in
float32); the input gradient has the logits' dtype. Saves the inputs (held by
the caller) and per-bin ``[B, C, K]`` tensors only.

Probabilities can differ from ``torch.softmax`` by ~1 ulp, so a probability within
~1e-7 of a bin edge may land in the neighbouring bin compared with the reference.
"""

from typing import Any

import torch

from ace_kernels.common import Prepared, reduce_classes
from ace_kernels.config import ACEConfig
from ace_kernels.impls.v2_autograd import _apply, _reduction_jacobian, coefficients
from ace_kernels.kernels import HAS_TRITON
from ace_kernels.registry import register


def _kernel_target(
    target: torch.Tensor, config: ACEConfig, channels: int
) -> torch.Tensor:
    """``[B, N]`` int64 class indices (one-hot conversion fused), or ``[B, C, N]``."""
    b = target.shape[0]
    if config.to_onehot_y and channels > 1:
        return target.reshape(b, -1).long().contiguous()
    return target.reshape(b, channels, -1).contiguous()


class _FusedACELoss(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any, input: torch.Tensor, target: torch.Tensor, config: ACEConfig
    ) -> torch.Tensor:
        from ace_kernels.kernels.act import as_kernel_input
        from ace_kernels.kernels.hard_hist import hard_stats_fused
        from ace_kernels.kernels.soft_hist import soft_stats_fused

        x = as_kernel_input(input)
        tgt = _kernel_target(target, config, x.shape[1])
        fused = hard_stats_fused if config.binning == "hard" else soft_stats_fused
        f, present, coef, d = coefficients(fused(x, tgt, config))
        # only spatial_ndim is read when class presence is given
        shape_only = Prepared(x, None, None, input.dim() - 2)
        loss = reduce_classes(f, shape_only, config, present)
        jac = _reduction_jacobian(f, shape_only, config, present)
        ctx.config = config
        ctx.save_for_backward(input, target, coef, d, jac)
        return loss

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[Any, ...]:
        from ace_kernels.kernels.act import as_kernel_input
        from ace_kernels.kernels.hard_bwd import hard_input_grad_triton
        from ace_kernels.kernels.soft_bwd import soft_input_grad_triton

        input, target, coef, d, jac = ctx.saved_tensors
        config: ACEConfig = ctx.config
        g = jac * (
            grad_output if grad_output.dim() == 0 else grad_output.view(jac.shape)
        )
        x = as_kernel_input(input)
        if config.binning == "hard":
            dx = hard_input_grad_triton(x, coef, g, config, from_logits=True)
        else:
            dx = soft_input_grad_triton(x, target, coef, d, g, config, from_logits=True)
        return dx.view(input.shape).to(input.dtype), None, None


if HAS_TRITON:

    @register(
        "v5_triton",
        description="Triton kernels with fused softmax: logits in, d logits out",
        devices=frozenset({"cuda"}),
    )
    def v5_triton(
        input: torch.Tensor, target: torch.Tensor, config: ACEConfig
    ) -> torch.Tensor:
        if input.dtype == torch.float64:  # gradcheck path: kernels are fp32 inside
            return _apply(input, target, config, "recompute", "torch")
        loss: torch.Tensor = _FusedACELoss.apply(input, target, config)
        return loss
