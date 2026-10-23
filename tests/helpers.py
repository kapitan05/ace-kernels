"""Shared test helpers: tolerances, devices, running an implementation fwd + bwd."""

import warnings
from dataclasses import dataclass

import pytest
import torch

from ace_kernels import ACEConfig, ace_loss

# fp32 tolerances for "same loss" between implementations. Summation order
# differs between implementations (and atomics on GPU), so exact equality is not
# expected; the reference itself agrees with the float64 oracle to ~1e-6.
LOSS_RTOL, LOSS_ATOL = 1e-5, 1e-6
GRAD_RTOL, GRAD_ATOL = 1e-4, 1e-7

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
