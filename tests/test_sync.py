"""Host<->device traffic: v1_clean must never sync or copy between CPU and GPU.

A sync stalls the training step's CPU-side kernel queueing; syncs and
host->device copies break CUDA-graph capture. Runs only with CUDA.
"""

import pytest
import torch

from ace_kernels import ACEConfig, ace_loss
from ace_kernels.cases import InputSpec, make_inputs
from ace_kernels.config import Binning
from ace_kernels.kernels import HAS_TRITON, TRITON_IMPLS
from tests.helpers import skip_unless_supported

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA"),
]


GRAPH_SAFE = ["v1_clean", "v2_recompute", "v2_grad_in_fwd"] + (
    sorted(TRITON_IMPLS) if HAS_TRITON else []
)


@pytest.mark.parametrize("impl", GRAPH_SAFE)
@pytest.mark.parametrize("binning", ["hard", "soft"])
@pytest.mark.parametrize("include_background", [True, False])
def test_has_no_host_sync(
    impl: str, binning: Binning, include_background: bool
) -> None:
    skip_unless_supported(impl, binning, "cuda", "paper-3d-hard")
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
    ace_loss(x, y, config, impl=impl)  # warm up lazy init outside the check
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        loss = ace_loss(x, y, config, impl=impl)
        (grad,) = torch.autograd.grad(loss, x)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert torch.isfinite(grad).all()


def test_reference_soft_is_not_graph_capturable() -> None:
    """Documents the baseline: the reference soft loss builds its sentinels with
    ``torch.tensor(..., device=cuda)``, a host->device copy that CUDA-graph capture
    rejects. (Its boolean-mask NaN fill does *not* sync on CUDA: measured.)"""
    config = ACEConfig(binning="soft", to_onehot_y=True)
    x, y = make_inputs(InputSpec(1, 2, (8, 8, 8)), config, "cuda")
    ace_loss(x, y, config, impl="reference")
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with (
        pytest.raises(RuntimeError, match="CUDA graph capture"),
        torch.cuda.graph(graph),
    ):
        ace_loss(x, y, config, impl="reference")


@pytest.mark.parametrize("impl", GRAPH_SAFE)
@pytest.mark.parametrize("binning", ["hard", "soft"])
def test_captures_in_cuda_graph(impl: str, binning: Binning) -> None:
    skip_unless_supported(impl, binning, "cuda", "paper-3d-hard")
    """No host<->device copies or syncs: fwd+bwd can be captured and replayed."""
    config = ACEConfig.model_validate(
        {"binning": binning, "to_onehot_y": True, "weight": (1.0, 2.0, 0.5, 1.0)}
    )
    x, y = make_inputs(InputSpec(2, 4, (16, 16, 16)), config, "cuda")
    x.requires_grad_(True)

    def step() -> torch.Tensor:
        (grad,) = torch.autograd.grad(ace_loss(x, y, config, impl=impl), x)
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
