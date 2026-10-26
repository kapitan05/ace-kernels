"""v2 handwritten backward: math checks beyond "matches the reference in fp32"."""

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from ace_kernels import ACEConfig, ace_loss
from ace_kernels.cases import InputSpec, make_inputs
from ace_kernels.config import Binning
from ace_kernels.instrumentation import measure_saved_tensors

V2 = ["v2_recompute", "v2_grad_in_fwd"]

CONFIGS: dict[str, dict[str, object]] = {
    "paper": {"to_onehot_y": True},
    "no-background-weights": {
        "to_onehot_y": True,
        "include_background": False,
        "weight": (2.0, 0.5, 1.0),
    },
    "reduction-none": {"to_onehot_y": True, "reduction": "none"},
    "reduction-sum-kept-empty": {
        "to_onehot_y": True,
        "reduction": "sum",
        "ignore_empty_classes": False,
    },
    "sigmoid": {"activation": "sigmoid"},
    "onehot-target": {},
}


def _inputs(
    name: str, binning: Binning
) -> tuple[ACEConfig, torch.Tensor, torch.Tensor]:
    config = ACEConfig.model_validate({"binning": binning, **CONFIGS[name]})
    channels = 1 if name == "sigmoid" else 4
    spec = InputSpec(2, channels, (5, 5, 4), seed=7, labels_as_index=config.to_onehot_y)
    x, y = make_inputs(spec, config)
    return config, x.double(), y


@pytest.mark.parametrize("impl", V2)
@pytest.mark.parametrize("binning", ["hard", "soft"])
@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_gradcheck_float64(impl: str, binning: Binning, name: str) -> None:
    """Analytic gradient == finite differences (independent of the reference)."""
    config, x, y = _inputs(name, binning)
    x.requires_grad_(True)
    assert torch.autograd.gradcheck(
        lambda inp: ace_loss(inp, y, config, impl=impl),
        (x,),
        eps=1e-7,
        atol=1e-6,
        rtol=1e-4,
    )


@pytest.mark.parametrize("impl", V2)
@settings(max_examples=30, deadline=None, derandomize=True)
@given(
    binning=st.sampled_from(["hard", "soft"]),
    channels=st.integers(2, 5),
    voxels=st.integers(1, 200),
    num_bins=st.integers(1, 25),
    include_background=st.booleans(),
    seed=st.integers(0, 2**31 - 1),
)
def test_matches_autograd_in_float64(
    impl: str,
    binning: Binning,
    channels: int,
    voxels: int,
    num_bins: int,
    include_background: bool,
    seed: int,
) -> None:
    """Handwritten formula == autograd through v1_clean, to float64 precision."""
    config = ACEConfig(
        binning=binning,
        num_bins=num_bins,
        to_onehot_y=True,
        include_background=include_background,
    )
    x, y = make_inputs(InputSpec(2, channels, (voxels,), seed=seed), config)
    x = x.double()
    grads = []
    for name in (impl, "v1_clean"):
        leaf = x.clone().requires_grad_(True)
        loss = ace_loss(leaf, y, config, impl=name)
        grads.append((loss.detach(), torch.autograd.grad(loss, leaf)[0]))
    (loss_v2, grad_v2), (loss_ref, grad_ref) = grads
    torch.testing.assert_close(loss_v2, loss_ref, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(grad_v2, grad_ref, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("binning", ["hard", "soft"])
def test_recompute_saves_only_per_bin_stats(binning: Binning) -> None:
    config = ACEConfig(binning=binning, to_onehot_y=True)
    b, c, k = 2, 4, config.num_bins
    x, y = make_inputs(InputSpec(b, c, (32, 32, 32)), config)
    x.requires_grad_(True)
    _, stats = measure_saved_tensors(
        lambda: ace_loss(x, y, config, impl="v2_recompute"), exclude=[x, y]
    )
    # coef and d (float32 [B, C, K]) + class presence (bool [B, C])
    assert stats.extra_bytes == 2 * b * c * k * 4 + b * c


@pytest.mark.parametrize("binning", ["hard", "soft"])
def test_grad_in_fwd_saves_one_input_gradient(binning: Binning) -> None:
    config = ACEConfig(binning=binning, to_onehot_y=True)
    x, y = make_inputs(InputSpec(2, 4, (32, 32, 32)), config)
    x.requires_grad_(True)
    _, stats = measure_saved_tensors(
        lambda: ace_loss(x, y, config, impl="v2_grad_in_fwd"), exclude=[x, y]
    )
    input_bytes = x.numel() * x.element_size()
    # + a few [B, C] tensors from differentiating the class reduction in forward
    assert input_bytes <= stats.extra_bytes <= input_bytes + 1024


def test_grad_in_fwd_scales_with_upstream_gradient() -> None:
    config = ACEConfig(binning="soft", to_onehot_y=True)
    x, y = make_inputs(InputSpec(2, 4, (6, 6, 6)), config)
    x = (
        x.double()
    )  # fp32 would differ by rounding: scale-then-chain vs chain-then-scale
    grads = []
    for impl in V2:
        leaf = x.clone().requires_grad_(True)
        loss = 3.5 * ace_loss(leaf, y, config, impl=impl)
        grads.append(torch.autograd.grad(loss, leaf)[0])
    torch.testing.assert_close(grads[0], grads[1], rtol=1e-10, atol=1e-14)


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_reduction_jacobian_matches_autograd_upstream(name: str) -> None:
    """Triton backward's shortcut dL/df = grad_output * jac == v2's autograd path."""
    from ace_kernels.common import prepare
    from ace_kernels.impls.v2_autograd import _reduction_jacobian, _upstream

    config, x, y = _inputs(name, "hard")
    prep = prepare(x, y, config, keep_labels=True)
    b, c = prep.probs.shape[:2]
    f = torch.rand(b, c, dtype=x.dtype)
    if config.reduction == "none":
        grad_output = torch.randn(b, c, *([1] * prep.spatial_ndim), dtype=x.dtype)
        shortcut = _reduction_jacobian(f, prep, config) * grad_output.view(b, c)
    else:
        grad_output = torch.tensor(2.5, dtype=x.dtype)
        shortcut = _reduction_jacobian(f, prep, config) * grad_output
    expected = _upstream(f.shape, prep, config, grad_output, f.device, f.dtype)
    torch.testing.assert_close(shortcut, expected, rtol=1e-12, atol=1e-15)


def test_class_presence_from_stats_matches_target_sum() -> None:
    """present = sum_k G_k > 0 must equal the reference's target.sum() > 0."""
    from ace_kernels.binning import bin_stats, class_present
    from ace_kernels.cases import CASES
    from ace_kernels.common import nonempty_classes, prepare

    for case in CASES:
        x, y = make_inputs(case.spec, case.config)
        for keep_labels in (True, False):
            prep = prepare(x, y, case.config, keep_labels=keep_labels)
            stats = bin_stats(prep, case.config)
            assert torch.equal(class_present(stats), nonempty_classes(prep)), case.name
