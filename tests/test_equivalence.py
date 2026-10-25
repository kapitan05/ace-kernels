"""Every implementation equals the reference on random shapes and options.

Only ``reference`` is registered in Sprint 1, so this is a self-check for now;
each new implementation is picked up automatically via the registry.
"""

import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from ace_kernels import ACEConfig, available_impls
from ace_kernels.cases import InputSpec, make_inputs
from tests.helpers import DEVICES, assert_same, run, skip_unless_supported


@st.composite
def random_case(draw: st.DrawFn) -> tuple[InputSpec, ACEConfig]:
    channels = draw(st.integers(1, 5))
    spatial = tuple(draw(st.lists(st.integers(1, 6), min_size=1, max_size=3)))
    spec = InputSpec(
        batch=draw(st.integers(1, 3)),
        channels=channels,
        spatial=spatial,
        seed=draw(st.integers(0, 2**31 - 1)),
        logit_scale=draw(st.sampled_from([0.5, 2.0, 10.0])),
    )
    config = ACEConfig(
        binning=draw(st.sampled_from(["hard", "soft"])),
        num_bins=draw(st.integers(1, 30)),
        activation="softmax" if channels > 1 else "sigmoid",
        to_onehot_y=channels > 1,
        include_background=draw(st.booleans()) or channels == 1,
        ignore_empty_classes=draw(st.booleans()),
        reduction=draw(st.sampled_from(["mean", "sum", "none"])),
        right=draw(st.booleans()),
    )
    return spec, config


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("impl", [s.name for s in available_impls()])
@settings(
    max_examples=40,
    deadline=None,
    derandomize=True,  # same examples every run: the gate must be reproducible
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(case=random_case())
def test_impl_matches_reference(
    impl: str, device: str, case: tuple[InputSpec, ACEConfig]
) -> None:
    spec, config = case
    skip_unless_supported(impl, config.binning, device)
    x, y = make_inputs(spec, config, device)
    expected = run("reference", x.cpu(), y.cpu(), config)
    assert_same(run(impl, x, y, config), expected)


def test_voxel_order_does_not_matter() -> None:
    """Binning losses depend only on the multiset of (p, y) per slice.

    Guards future kernels whose accumulation order differs (tiles, atomics).
    """
    spec = InputSpec(2, 4, (6, 6, 6), seed=3)
    for binning in ("hard", "soft"):
        config = ACEConfig(binning=binning, to_onehot_y=True)
        x, y = make_inputs(spec, config)
        perm = torch.randperm(6 * 6 * 6, generator=torch.Generator().manual_seed(0))
        xp = x.flatten(2)[..., perm].reshape(x.shape)
        yp = y.flatten(2)[..., perm].reshape(y.shape)
        a = run("reference", x, y, config).loss
        b = run("reference", xp, yp, config).loss
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


def test_loss_is_bounded_and_finite() -> None:
    """Unweighted mean mL1-ACE lies in [0, 1]; gradients are finite."""
    for seed in range(5):
        for binning in ("hard", "soft"):
            config = ACEConfig(binning=binning, to_onehot_y=True)
            x, y = make_inputs(InputSpec(2, 4, (5, 5, 5), seed=seed), config)
            result = run("reference", x, y, config)
            assert 0.0 <= result.loss.item() <= 1.0
            assert torch.isfinite(result.grad).all()
