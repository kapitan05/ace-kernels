"""v6 (Sprint 11): closing the gaps of v5, one registry entry per change.

- ``v6_index``:  v5 + O(1) bin index in all kernels (``kernels/bins.sorted_count``).
- ``v6_triton``: + one-launch epilogue (``kernels/epilogue.py``) instead of ~50 small
  PyTorch ops, registered as ``torch.library`` custom ops so ``torch.compile`` of a
  whole training step treats the loss as two opaque nodes (no graph break).
- ``v6_precise``: v6_triton with libdevice ``exp`` instead of Triton's fast one
  (accuracy vs speed; autograd.Function, not the custom op).
- ``v6_dot``:    side branch, v6_index + tensor-core histogram reduction
  (``kernels/bins.put3``): 3.6-4x slower on a T4 (no tensor-core ``tl.dot`` on
  sm_75); kept to re-measure on sm_80+ GPUs.

Like v5: logits in (fp32/bf16/fp16), d logits out; saves the inputs and per-bin
``[B, C, K]`` tensors only.
"""

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import torch

from ace_kernels.common import Prepared, reduce_classes
from ace_kernels.config import ACEConfig
from ace_kernels.impls.v2_autograd import _apply, _reduction_jacobian, coefficients
from ace_kernels.impls.v5_triton import _kernel_target
from ace_kernels.kernels import HAS_TRITON
from ace_kernels.registry import register

Outputs = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class Options:
    direct: bool = True  # O(1) bin index
    dot: bool = False  # tensor-core reduction
    epilogue: bool = False  # one-launch Triton epilogue
    precise: bool = False  # libdevice exp


def _forward(
    input: torch.Tensor, target: torch.Tensor, config: ACEConfig, opt: Options
) -> Outputs:
    """(loss, coef, d, jac) for logits ``input``."""
    from ace_kernels.kernels.act import as_kernel_input
    from ace_kernels.kernels.epilogue import epilogue_triton
    from ace_kernels.kernels.hard_hist import hard_hist_fused, hard_stats_fused
    from ace_kernels.kernels.soft_hist import soft_hist_fused, soft_stats_fused

    x = as_kernel_input(input)
    tgt = _kernel_target(target, config, x.shape[1])
    args = (x, tgt, config, opt.direct, opt.dot, opt.precise)
    hard = config.binning == "hard"
    if opt.epilogue:
        hist = hard_hist_fused(*args) if hard else soft_hist_fused(*args)
        return epilogue_triton(hist, config, input.dim() - 2)
    f, present, coef, d = coefficients(
        hard_stats_fused(*args) if hard else soft_stats_fused(*args)
    )
    shape_only = Prepared(x, None, None, input.dim() - 2)
    loss = reduce_classes(f, shape_only, config, present)
    jac = _reduction_jacobian(f, shape_only, config, present)
    return loss, coef, d, jac


def _backward(
    input: torch.Tensor,
    target: torch.Tensor,
    coef: torch.Tensor,
    d: torch.Tensor,
    jac: torch.Tensor,
    grad_output: torch.Tensor,
    config: ACEConfig,
    opt: Options,
) -> torch.Tensor:
    from ace_kernels.kernels.act import as_kernel_input
    from ace_kernels.kernels.hard_bwd import hard_input_grad_triton
    from ace_kernels.kernels.soft_bwd import soft_input_grad_triton

    g = jac * (grad_output if grad_output.dim() == 0 else grad_output.view(jac.shape))
    x = as_kernel_input(input)
    if config.binning == "hard":
        dx = hard_input_grad_triton(x, coef, g, config, True, opt.direct, opt.precise)
    else:
        dx = soft_input_grad_triton(
            x, target, coef, d, g, config, True, opt.direct, opt.precise
        )
    return dx.view(input.shape).to(input.dtype)


class _V6Loss(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        input: torch.Tensor,
        target: torch.Tensor,
        config: ACEConfig,
        opt: Options,
    ) -> torch.Tensor:
        loss, coef, d, jac = _forward(input, target, config, opt)
        ctx.config, ctx.opt = config, opt
        ctx.save_for_backward(input, target, coef, d, jac)
        return loss

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[Any, ...]:
        input, target, coef, d, jac = ctx.saved_tensors
        dx = _backward(input, target, coef, d, jac, grad_output, ctx.config, ctx.opt)
        return dx, None, None, None


# v6_triton: no tensor-core reduction. Measured on a T4 (sm_75, no Triton
# tensor-core path): v6_dot 3.6-4x *slower* than v6_index. To re-test on sm_80+.
FULL = Options(direct=True, epilogue=True)


@functools.lru_cache(maxsize=64)
def _config(config_json: str) -> ACEConfig:
    return ACEConfig.model_validate_json(config_json)


def _dump(config: ACEConfig) -> str:
    return config.model_dump_json()


# The op's config argument; constant for ``torch.compile`` (pydantic is not
# traceable), so a compiled function must not swap configs between calls.
# (torch.compiler's decorators are unannotated: typed through ``cast``.)
config_arg = cast(
    Callable[[ACEConfig], str],
    torch.compiler.assume_constant_result(_dump),  # type: ignore[no-untyped-call]
)


@torch.library.custom_op("ace_kernels::v6_forward", mutates_args=())
def v6_forward(input: torch.Tensor, target: torch.Tensor, config_json: str) -> Outputs:
    """mL1-ACE forward: (loss, coef, d, jac); see ``_forward``."""
    loss, coef, d, jac = _forward(input, target, _config(config_json), FULL)
    # custom ops may not return views of internal buffers: copy the "none" view
    return loss.clone() if loss.dim() else loss, coef, d, jac


@v6_forward.register_fake
def _(input: torch.Tensor, target: torch.Tensor, config_json: str) -> Outputs:
    config = _config(config_json)
    b, c = input.shape[:2]
    cl = c - (1 if (not config.include_background and c > 1) else 0)
    per_bin = input.new_empty((b, cl, config.num_bins), dtype=torch.float32)
    loss_shape = (b, cl, *([1] * (input.dim() - 2)))
    loss = input.new_empty(
        loss_shape if config.reduction == "none" else (), dtype=torch.float32
    )
    jac = input.new_empty((b, cl), dtype=torch.float32)
    return loss, per_bin, torch.empty_like(per_bin), jac


@torch.library.custom_op("ace_kernels::v6_backward", mutates_args=())
def v6_backward(
    input: torch.Tensor,
    target: torch.Tensor,
    coef: torch.Tensor,
    d: torch.Tensor,
    jac: torch.Tensor,
    grad_output: torch.Tensor,
    config_json: str,
) -> torch.Tensor:
    """d loss / d input; see ``_backward``."""
    return _backward(
        input, target, coef, d, jac, grad_output, _config(config_json), FULL
    )


@v6_backward.register_fake
def _(
    input: torch.Tensor,
    target: torch.Tensor,
    coef: torch.Tensor,
    d: torch.Tensor,
    jac: torch.Tensor,
    grad_output: torch.Tensor,
    config_json: str,
) -> torch.Tensor:
    return torch.empty_like(input)


def _setup_context(ctx: Any, inputs: tuple[Any, ...], output: Outputs) -> None:
    input, target, config_json = inputs
    _, coef, d, jac = output
    ctx.config_json = config_json
    ctx.set_materialize_grads(False)  # coef / d / jac get no gradient: no zero fills
    ctx.save_for_backward(input, target, coef, d, jac)


def _op_backward(
    ctx: Any, grad_loss: torch.Tensor, *_unused: torch.Tensor
) -> tuple[torch.Tensor | None, None, None]:
    input, target, coef, d, jac = ctx.saved_tensors
    dx = v6_backward(input, target, coef, d, jac, grad_loss, ctx.config_json)
    return dx, None, None


v6_forward.register_autograd(_op_backward, setup_context=_setup_context)


def _entry(opt: Options | None) -> Any:
    def impl(
        input: torch.Tensor, target: torch.Tensor, config: ACEConfig
    ) -> torch.Tensor:
        if input.dtype == torch.float64:  # gradcheck path: kernels are fp32 inside
            return _apply(input, target, config, "recompute", "torch")
        if opt is None:
            loss: torch.Tensor = v6_forward(input, target, config_arg(config))[0]
        else:
            loss = _V6Loss.apply(input, target, config, opt)
        return loss

    return impl


if HAS_TRITON:
    _CUDA = frozenset({"cuda"})
    register(
        "v6_index",
        description="v5 + O(1) bin index (floor + exact correction)",
        devices=_CUDA,
    )(_entry(Options(direct=True)))
    register(
        "v6_precise",
        description="v6_triton with libdevice exp (as torch.softmax) in the kernels",
        devices=_CUDA,
    )(_entry(Options(direct=True, epilogue=True, precise=True)))
    register(
        "v6_dot",
        description="v6_index + tensor-core histogram reduction (side branch)",
        devices=_CUDA,
    )(_entry(Options(direct=True, dot=True)))
    register(
        "v6_triton",
        description="v6_index + one-launch epilogue, as torch.library custom ops",
        devices=_CUDA,
    )(_entry(None))
