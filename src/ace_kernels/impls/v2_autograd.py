"""v2: the whole loss as one ``autograd.Function`` with a handwritten backward.

Implements docs/derivation.md. The forward runs without autograd, so no
voxel-sized intermediate (softmax output, bin indices, soft weights) is kept for
backward. Two variants:

- ``v2_recompute``: saves only the inputs (held by the caller anyway) and two
  ``[B, C, K]`` tensors per bin (``coef = sign(d) / (K_ne W)`` and ``d``); the
  backward recomputes softmax and bin assignment, then applies the formula.
- ``v2_grad_in_fwd``: Liger-style. The forward also computes the input gradient
  for a unit upstream gradient and saves it (1x input); the backward only scales
  it. Possible for mean/sum reductions (scalar upstream gradient); with
  ``reduction="none"`` it falls back to recompute.

Class mask, weights and reduction (``reduce_classes``) are differentiated with
autograd on a ``[B, C]`` tensor, so they match the reference by construction.
"""

from typing import Any, Literal

import torch

from ace_kernels.binning import (
    bin_difference,
    bin_stats,
    hard_bin_index,
    per_bin_error,
)
from ace_kernels.common import (
    Prepared,
    compute_dtype,
    prepare,
    reduce_classes,
    soft_assign,
)
from ace_kernels.config import ACEConfig
from ace_kernels.registry import register

Mode = Literal["recompute", "grad_in_fwd"]


def _bin_coefficients(
    prep: Prepared, config: ACEConfig
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-class error ``f`` [B, C] and the backward's per-bin ``coef``, ``d``."""
    stats = bin_stats(prep, config)
    d = bin_difference(stats)
    f = torch.nanmean(per_bin_error(stats), dim=-1)
    nonempty = ~stats.empty
    k_ne = nonempty.sum(-1, keepdim=True).clamp_min(1).to(d.dtype)
    safe_w = torch.where(nonempty, stats.weight, torch.ones_like(stats.weight))
    coef = torch.where(nonempty, torch.sign(d) / (k_ne * safe_w), torch.zeros_like(d))
    return f, coef, d


def _true_target(prep: Prepared) -> torch.Tensor:
    """``[B, C, N]`` y in the probabilities' dtype (built only for the backward)."""
    if prep.onehot is not None:
        return prep.onehot
    assert prep.labels is not None
    channels = torch.arange(prep.probs.shape[1], device=prep.probs.device)
    return (prep.labels.unsqueeze(1) == channels.view(1, -1, 1)).to(prep.probs.dtype)


def _df_dprobs(
    prep: Prepared, coef: torch.Tensor, d: torch.Tensor, config: ACEConfig
) -> torch.Tensor:
    """``[B, C, N]``: d f_bc / d p_bci (derivation: sum over the voxel's bins)."""
    if config.binning == "hard":
        return coef.gather(2, hard_bin_index(prep, config))
    p = prep.probs
    a = soft_assign(p, config.num_bins, config.right)
    inv_span = 1.0 / a.span
    resid = p - _true_target(prep)  # p_i - y_i
    left = coef.gather(2, a.left_idx) * (
        a.left_w - inv_span * (resid - d.gather(2, a.left_idx))
    )
    right = coef.gather(2, a.right_idx) * (
        a.right_w + inv_span * (resid - d.gather(2, a.right_idx))
    )
    return left + right


def _activation_backward(
    input: torch.Tensor, grad_probs: torch.Tensor, config: ACEConfig
) -> torch.Tensor:
    """Chain ``dL/dp`` (all channels, input shape) through the activation."""
    dtype = compute_dtype(input)
    if config.activation == "sigmoid":
        p = torch.sigmoid(input).to(dtype)
        return grad_probs * p * (1 - p)
    if config.activation == "softmax" and input.shape[1] > 1:
        p = torch.softmax(input, 1).to(dtype)
        return p * (grad_probs - (grad_probs * p).sum(1, keepdim=True))
    return grad_probs


def _upstream(
    f_shape: torch.Size,
    prep: Prepared,
    config: ACEConfig,
    grad_output: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """``dL/df_bc`` [B, C] from mask, weights and reduction (via autograd)."""
    with torch.enable_grad():
        f_var = torch.zeros(f_shape, device=device, dtype=dtype, requires_grad=True)
        out = reduce_classes(f_var, prep, config)
        (g,) = torch.autograd.grad(out, f_var, grad_output.to(dtype))
    return g


def _input_grad(
    input: torch.Tensor,
    target: torch.Tensor,
    config: ACEConfig,
    coef: torch.Tensor,
    d: torch.Tensor,
    grad_output: torch.Tensor,
) -> torch.Tensor:
    prep = prepare(input, target, config, keep_labels=True)
    g = _upstream(coef.shape[:2], prep, config, grad_output, input.device, coef.dtype)
    grad_p = g.unsqueeze(-1) * _df_dprobs(prep, coef, d, config)  # [B, C', N]
    if grad_p.shape[1] != input.shape[1]:  # background channel was dropped
        grad_p = torch.cat((torch.zeros_like(grad_p[:, :1]), grad_p), dim=1)
    grad_p = grad_p.view(input.shape)
    return _activation_backward(input, grad_p, config).to(input.dtype)


class _ACELoss(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        input: torch.Tensor,
        target: torch.Tensor,
        config: ACEConfig,
        mode: Mode,
    ) -> torch.Tensor:
        prep = prepare(input, target, config, keep_labels=True)
        f, coef, d = _bin_coefficients(prep, config)
        loss = reduce_classes(f, prep, config)
        ctx.config = config
        ctx.precomputed = mode == "grad_in_fwd" and config.reduction != "none"
        if ctx.precomputed:  # gradient for upstream 1, scaled in backward
            unit = torch.ones((), device=loss.device, dtype=loss.dtype)
            ctx.save_for_backward(_input_grad(input, target, config, coef, d, unit))
        else:
            ctx.save_for_backward(input, target, coef, d)
        return loss

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[Any, ...]:
        if ctx.precomputed:
            (unit_grad,) = ctx.saved_tensors
            return grad_output * unit_grad, None, None, None
        input, target, coef, d = ctx.saved_tensors
        grad = _input_grad(input, target, ctx.config, coef, d, grad_output)
        return grad, None, None, None


def _apply(
    input: torch.Tensor, target: torch.Tensor, config: ACEConfig, mode: Mode
) -> torch.Tensor:
    loss: torch.Tensor = _ACELoss.apply(input, target, config, mode)
    return loss


@register(
    "v2_recompute",
    description="Handwritten backward; saves inputs + [B,C,K] stats, recomputes",
)
def v2_recompute(
    input: torch.Tensor, target: torch.Tensor, config: ACEConfig
) -> torch.Tensor:
    return _apply(input, target, config, "recompute")


@register(
    "v2_grad_in_fwd",
    description="Handwritten backward; gradient computed in forward (Liger-style)",
)
def v2_grad_in_fwd(
    input: torch.Tensor, target: torch.Tensor, config: ACEConfig
) -> torch.Tensor:
    return _apply(input, target, config, "grad_in_fwd")
