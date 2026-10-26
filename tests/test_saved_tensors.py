"""Memory saved for backward: the instrumentation itself and the reference baseline."""

import gc
import weakref

import pytest
import torch

from ace_kernels import ACEConfig, ace_loss
from ace_kernels.cases import InputSpec, make_inputs
from ace_kernels.config import Binning
from ace_kernels.instrumentation import measure_saved_tensors


def test_instrumentation_counts_saved_input() -> None:
    x = torch.randn(1000, requires_grad=True)
    _, stats = measure_saved_tensors(lambda: (x * x).sum(), exclude=[x])
    assert stats.unique_bytes == x.numel() * 4  # x * x saves x (twice, one storage)
    assert stats.excluded_bytes == stats.unique_bytes
    assert stats.extra_bytes == 0


def test_instrumentation_counts_new_activation() -> None:
    x = torch.randn(1000, requires_grad=True)
    _, stats = measure_saved_tensors(lambda: torch.sigmoid(x).sum(), exclude=[x])
    assert stats.extra_bytes == x.numel() * 4  # sigmoid saves its output


@pytest.mark.parametrize("binning", ["hard", "soft"])
def test_reference_saves_per_voxel_tensors(binning: Binning) -> None:
    """Baseline fact: the reference keeps several voxel-sized tensors for backward.

    Optimized implementations must bring ``extra_bytes`` down to O(B * C * K).
    """
    config = ACEConfig(binning=binning, to_onehot_y=True)
    x, y = make_inputs(InputSpec(2, 4, (16, 16, 16)), config)
    x.requires_grad_(True)
    _, stats = measure_saved_tensors(lambda: ace_loss(x, y, config), exclude=[x])
    input_bytes = x.numel() * x.element_size()
    assert stats.extra_bytes > input_bytes, stats.extra_bytes


def test_measurement_does_not_leak_saved_outputs() -> None:
    """Regression: the pack hook must not keep saved outputs alive (ref cycle)."""
    x = torch.randn(1000, requires_grad=True)
    refs: list[weakref.ref[torch.Tensor]] = []

    def f() -> torch.Tensor:
        y = torch.softmax(x, 0)  # softmax saves its output for backward
        refs.append(weakref.ref(y))
        return y.sum()

    out, _ = measure_saved_tensors(f)
    del out
    gc.collect()
    assert refs[0]() is None


def test_measured_graph_still_backpropagates() -> None:
    x = torch.randn(1000, requires_grad=True)
    out, _ = measure_saved_tensors(lambda: torch.softmax(x, 0)[0] * 3)
    (grad,) = torch.autograd.grad(out, x)
    (expected,) = torch.autograd.grad(torch.softmax(x, 0)[0] * 3, x)
    torch.testing.assert_close(grad, expected)
