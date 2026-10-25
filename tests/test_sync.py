"""Host-device synchronization: v1_clean must never block the CPU on the GPU.

A sync inside the loss stalls the training step's CPU-side kernel queueing and
breaks CUDA-graph capture. Runs only with CUDA.
"""

import pytest
import torch

from ace_kernels import ACEConfig, ace_loss
from ace_kernels.cases import InputSpec, make_inputs

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA"),
]


@pytest.mark.parametrize("binning", ["hard", "soft"])
@pytest.mark.parametrize("include_background", [True, False])
def test_v1_clean_has_no_host_sync(binning: str, include_background: bool) -> None:
    config = ACEConfig.model_validate(
        {
            "binning": binning,
            "to_onehot_y": True,
            "include_background": include_background,
            "weight": (1.0, 2.0, 0.5, 1.0) if include_background else None,
        }
    )
    x, y = make_inputs(InputSpec(2, 4, (16, 16, 16)), config, "cuda")
    x.requires_grad_(True)
    ace_loss(x, y, config, impl="v1_clean")  # warm up lazy init outside the check
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        loss = ace_loss(x, y, config, impl="v1_clean")
        (grad,) = torch.autograd.grad(loss, x)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert torch.isfinite(grad).all()


def test_reference_does_sync() -> None:
    """Documents the baseline: the reference's boolean-mask NaN fill syncs."""
    config = ACEConfig(to_onehot_y=True)
    x, y = make_inputs(InputSpec(1, 2, (8, 8, 8)), config, "cuda")
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        with pytest.raises(RuntimeError, match="synchroniz"):
            ace_loss(x, y, config, impl="reference")
    finally:
        torch.cuda.set_sync_debug_mode("default")


@pytest.mark.parametrize("binning", ["hard", "soft"])
def test_v1_clean_captures_in_cuda_graph(binning: str) -> None:
    """No host<->device copies or syncs: fwd+bwd can be captured and replayed."""
    config = ACEConfig.model_validate(
        {"binning": binning, "to_onehot_y": True, "weight": (1.0, 2.0, 0.5, 1.0)}
    )
    x, y = make_inputs(InputSpec(2, 4, (16, 16, 16)), config, "cuda")
    x.requires_grad_(True)

    def step() -> torch.Tensor:
        (grad,) = torch.autograd.grad(ace_loss(x, y, config, impl="v1_clean"), x)
        return grad

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            expected = step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = step()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured, expected)
