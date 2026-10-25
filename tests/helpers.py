"""Shared test helpers: tolerances, devices, running an implementation fwd + bwd."""

import warnings
from dataclasses import dataclass

import pytest
import torch

from ace_kernels import ACEConfig, ace_loss, get_impl
from ace_kernels.config import Binning

# fp32 tolerances for "same loss" between implementations. Summation order
# differs between implementations (and atomics on GPU), so exact equality is not
# expected; the reference itself agrees with the float64 oracle to ~1e-6.
LOSS_RTOL, LOSS_ATOL = 1e-5, 1e-6
GRAD_RTOL, GRAD_ATOL = 1e-4, 1e-7

# Cases run for implementations marked ``slow`` (e.g. torch.compile recompiles
# for every new shape): the paper setting plus the most fragile edge cases.
SMOKE_CASES = frozenset(
    {
        "paper-3d-hard",
        "paper-3d-soft",
        "empty-class-ignored-hard",
        "bin-edges-right-hard",
        "reduction-none-soft",
    }
)

# Cases with exact ties: probabilities exactly on edges / constant / 0-1, so some
# bins have mean(p) - mean(y) == 0 exactly, where |x| has a kink. Eager PyTorch
# returns the subgradient sign(0) = 0; a compiled kernel reorders the float ops,
# gets +-1e-9 and returns +-1/N. Both are valid subgradients (measured on a T4:
# v1_clean_compile, bin-edges-right-hard). For compiled impls these cases check
# the loss exactly and the gradient only for finiteness.
TIE_CASES = frozenset(
    {
        f"{name}-{b}"
        for name in (
            "bin-edges-left",
            "bin-edges-right",
            "constant-probs",
            "extreme-probs",
        )
        for b in ("hard", "soft")
    }
)

DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=[
            pytest.mark.cuda,
            pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA"),
        ],
    ),
]


@dataclass
class Result:
    loss: torch.Tensor
    grad: torch.Tensor


def run(impl: str, x: torch.Tensor, y: torch.Tensor, config: ACEConfig) -> Result:
    """Loss and d(loss.sum())/d(input) on a fresh leaf copy of ``x``."""
    leaf = x.detach().clone().requires_grad_(True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # reference warns for C=1
        loss = ace_loss(leaf, y, config, impl=impl)
    torch.autograd.backward(loss.sum())
    assert leaf.grad is not None
    return Result(loss.detach(), leaf.grad)


def assert_same(actual: Result, expected: Result) -> None:
    torch.testing.assert_close(
        actual.loss,
        expected.loss.to(actual.loss.device),
        rtol=LOSS_RTOL,
        atol=LOSS_ATOL,
    )
    torch.testing.assert_close(
        actual.grad,
        expected.grad.to(actual.grad.device),
        rtol=GRAD_RTOL,
        atol=GRAD_ATOL,
    )


def skip_unless_supported(
    impl: str, binning: Binning, device: str, case_name: str | None = None
) -> None:
    """Skip the current test if ``impl`` cannot or should not run this case."""
    spec = get_impl(impl)
    if not spec.supports(binning, device):
        pytest.skip(f"{impl} does not support {binning} on {device}")
    if spec.slow and (case_name is None or case_name not in SMOKE_CASES):
        pytest.skip(f"{impl} is slow: smoke cases only")
