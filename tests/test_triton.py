"""Triton kernels vs the PyTorch per-bin statistics (CUDA + Triton only)."""

import pytest
import torch

from ace_kernels import ACEConfig
from ace_kernels.binning import BinStats, bin_stats, hard_stats, soft_stats
from ace_kernels.cases import InputKind, InputSpec, make_inputs
from ace_kernels.common import Prepared, prepare
from ace_kernels.config import Binning
from ace_kernels.kernels import HAS_TRITON

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(
        not (HAS_TRITON and torch.cuda.is_available()), reason="needs Triton + CUDA"
    ),
]

SPATIAL = {"n1": (1,), "n7": (7,), "n100003": (100003,), "cube": (24, 24, 24)}


@pytest.mark.parametrize("spatial", sorted(SPATIAL))
@pytest.mark.parametrize("channels", [1, 4, 16])
@pytest.mark.parametrize("num_bins", [1, 20, 50])
@pytest.mark.parametrize("right", [False, True])
@pytest.mark.parametrize("kind", ["logits", "bin_edges", "extreme"])
def test_hard_stats_match_torch(
    spatial: str, channels: int, num_bins: int, right: bool, kind: InputKind
) -> None:
    from ace_kernels.kernels.hard_hist import hard_stats_triton

    labels = channels > 1 and kind == "logits"
    config = ACEConfig(
        binning="hard",
        num_bins=num_bins,
        right=right,
        to_onehot_y=labels,
        # probabilities must lie in [0, 1]: softmax is ignored for one channel
        # (as in the reference), so single-channel logits need a sigmoid
        activation=(
            "none" if kind != "logits" else "sigmoid" if channels == 1 else "softmax"
        ),
    )
    spec = InputSpec(
        2, channels, SPATIAL[spatial], seed=3, kind=kind, labels_as_index=labels
    )
    x, y = make_inputs(spec, config, "cuda")
    prep = prepare(x, y, config, keep_labels=True)
    expected = hard_stats(prep, config)
    actual = hard_stats_triton(prep, config)
    torch.testing.assert_close(actual.weight, expected.weight, rtol=0, atol=0)
    torch.testing.assert_close(actual.sum_gt, expected.sum_gt, rtol=0, atol=0)
    assert torch.equal(actual.empty, expected.empty)
    _assert_sum_p_accurate(prep, config, actual.sum_p, expected.sum_p)


def _truth64(prep: Prepared, config: ACEConfig) -> BinStats:
    """Per-bin statistics in float64 (same bins): the accuracy reference."""
    prep64 = Prepared(
        prep.probs.double(),
        None if prep.onehot is None else prep.onehot.double(),
        prep.labels,
        prep.spatial_ndim,
    )
    return bin_stats(prep64, config)


def _assert_accurate(
    triton_val: torch.Tensor, torch_val: torch.Tensor, truth: torch.Tensor
) -> None:
    """Within 1e-5 of the float64 truth and no less accurate than PyTorch fp32."""
    scale = truth.abs().clamp_min(1.0)
    err_triton = ((triton_val.double() - truth).abs() / scale).max().item()
    err_torch = ((torch_val.double() - truth).abs() / scale).max().item()
    assert err_triton <= 1e-5, (err_triton, err_torch)
    assert err_triton <= max(err_torch, 1e-6), (err_triton, err_torch)


def _assert_sum_p_accurate(
    prep: Prepared, config: ACEConfig, triton_sum: torch.Tensor, torch_sum: torch.Tensor
) -> None:
    """Sum of p per bin vs a float64 ground truth (same bins).

    fp32 sums of ~1e5 values differ by summation order (measured on a T4: up to
    4.9e-5 relative between the kernel and scatter_add); the kernel sums each block
    as a tree before one atomic add, scatter_add accumulates one value at a time.
    """
    _assert_accurate(triton_sum, torch_sum, _truth64(prep, config).sum_p)


@pytest.mark.parametrize("include_background", [True, False])
def test_hard_stats_index_target_background(include_background: bool) -> None:
    from ace_kernels.kernels.hard_hist import hard_stats_triton

    config = ACEConfig(to_onehot_y=True, include_background=include_background)
    x, y = make_inputs(InputSpec(2, 5, (40, 40, 9), seed=5), config, "cuda")
    prep = prepare(x, y, config, keep_labels=True)
    expected = hard_stats(prep, config)
    actual = hard_stats_triton(prep, config)
    torch.testing.assert_close(actual.weight, expected.weight, rtol=0, atol=0)
    torch.testing.assert_close(actual.sum_gt, expected.sum_gt, rtol=0, atol=0)
    _assert_sum_p_accurate(prep, config, actual.sum_p, expected.sum_p)


BWD_CONFIGS: dict[str, dict[str, object]] = {
    "paper": {"to_onehot_y": True},
    "no-background-weights": {
        "to_onehot_y": True,
        "include_background": False,
        "weight": (2.0, 0.5, 1.0, 1.0),
    },
    "reduction-none": {"to_onehot_y": True, "reduction": "none"},
    "kept-empty-sum": {
        "to_onehot_y": True,
        "reduction": "sum",
        "ignore_empty_classes": False,
    },
    "fifty-bins-right": {"to_onehot_y": True, "num_bins": 50, "right": True},
    "one-bin": {"to_onehot_y": True, "num_bins": 1},
    "onehot-target": {},
}


@pytest.mark.parametrize("name", sorted(BWD_CONFIGS))
@pytest.mark.parametrize("spatial", [(7,), (33, 17, 5), (64, 64, 64)])
def test_v4_gradient_matches_v2(name: str, spatial: tuple[int, ...]) -> None:
    """Triton backward == handwritten PyTorch backward (same forward stats)."""
    from ace_kernels import ace_loss

    config = ACEConfig.model_validate({"binning": "hard", **BWD_CONFIGS[name]})
    channels = 5
    spec = InputSpec(2, channels, spatial, seed=11, labels_as_index=config.to_onehot_y)
    x, y = make_inputs(spec, config, "cuda")
    grads = []
    for impl in ("v2_recompute", "v4_triton"):
        leaf = x.clone().requires_grad_(True)
        loss = ace_loss(leaf, y, config, impl=impl)
        upstream = torch.linspace(0.5, 2.0, loss.numel(), device="cuda").view_as(loss)
        grads.append(torch.autograd.grad(loss, leaf, upstream)[0])
    torch.testing.assert_close(grads[1], grads[0], rtol=1e-4, atol=1e-8)


def test_v4_sigmoid_single_channel() -> None:
    from ace_kernels import ace_loss

    config = ACEConfig(binning="hard", activation="sigmoid")
    x, y = make_inputs(InputSpec(2, 1, (40, 40, 8), seed=2), config, "cuda")
    grads = []
    for impl in ("v2_recompute", "v4_triton"):
        leaf = x.clone().requires_grad_(True)
        grads.append(torch.autograd.grad(ace_loss(leaf, y, config, impl=impl), leaf)[0])
    torch.testing.assert_close(grads[1], grads[0], rtol=1e-4, atol=1e-8)


def test_large_volume_grid() -> None:
    """Regression: > 65535 voxel blocks (256^3) failed with "invalid argument" when
    the voxel blocks were on grid axis 1 (limit 65535)."""
    from ace_kernels import ace_loss
    from ace_kernels.kernels.hard_hist import hard_stats_triton

    config = ACEConfig(binning="hard", to_onehot_y=True)
    x, y = make_inputs(InputSpec(1, 2, (256, 256, 256), seed=1), config, "cuda")
    prep = prepare(x, y, config, keep_labels=True)
    torch.testing.assert_close(
        hard_stats_triton(prep, config).weight, hard_stats(prep, config).weight
    )
    leaf = x.clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(ace_loss(leaf, y, config, impl="v4_triton"), leaf)
    assert torch.isfinite(grad).all()


@pytest.mark.parametrize("spatial", sorted(SPATIAL))
@pytest.mark.parametrize("channels", [1, 4, 16])
@pytest.mark.parametrize("num_bins", [1, 20, 50])
@pytest.mark.parametrize("right", [False, True])
@pytest.mark.parametrize("kind", ["logits", "bin_edges", "extreme"])
def test_soft_stats_match_torch(
    spatial: str, channels: int, num_bins: int, right: bool, kind: InputKind
) -> None:
    from ace_kernels.kernels.soft_hist import soft_stats_triton

    labels = channels > 1 and kind == "logits"
    config = ACEConfig(
        binning="soft",
        num_bins=num_bins,
        right=right,
        to_onehot_y=labels,
        activation=(
            "none" if kind != "logits" else "sigmoid" if channels == 1 else "softmax"
        ),
    )
    spec = InputSpec(
        2, channels, SPATIAL[spatial], seed=3, kind=kind, labels_as_index=labels
    )
    x, y = make_inputs(spec, config, "cuda")
    prep = prepare(x, y, config, keep_labels=True)
    expected = soft_stats(prep, config)
    actual = soft_stats_triton(prep, config)
    truth = _truth64(prep, config)
    _assert_accurate(actual.weight, expected.weight, truth.weight)
    _assert_accurate(actual.sum_p, expected.sum_p, truth.sum_p)
    _assert_accurate(actual.sum_gt, expected.sum_gt, truth.sum_gt)
    # empty = weight < empty_weight; may only differ within rounding of the threshold
    near = (truth.weight - config.empty_weight).abs() < 1e-4
    assert torch.equal(actual.empty | near, expected.empty | near)


@pytest.mark.parametrize("name", sorted(BWD_CONFIGS))
@pytest.mark.parametrize("spatial", [(7,), (33, 17, 5), (64, 64, 64)])
def test_v4_soft_gradient_matches_v2(name: str, spatial: tuple[int, ...]) -> None:
    from ace_kernels import ace_loss

    config = ACEConfig.model_validate({"binning": "soft", **BWD_CONFIGS[name]})
    spec = InputSpec(2, 5, spatial, seed=11, labels_as_index=config.to_onehot_y)
    x, y = make_inputs(spec, config, "cuda")
    grads = []
    for impl in ("v2_recompute", "v4_triton"):
        leaf = x.clone().requires_grad_(True)
        loss = ace_loss(leaf, y, config, impl=impl)
        upstream = torch.linspace(0.5, 2.0, loss.numel(), device="cuda").view_as(loss)
        grads.append(torch.autograd.grad(loss, leaf, upstream)[0])
    torch.testing.assert_close(grads[1], grads[0], rtol=1e-4, atol=1e-8)


def test_v4_soft_sigmoid_single_channel() -> None:
    from ace_kernels import ace_loss

    config = ACEConfig(binning="soft", activation="sigmoid")
    x, y = make_inputs(InputSpec(2, 1, (40, 40, 8), seed=2), config, "cuda")
    grads = []
    for impl in ("v2_recompute", "v4_triton"):
        leaf = x.clone().requires_grad_(True)
        grads.append(torch.autograd.grad(ace_loss(leaf, y, config, impl=impl), leaf)[0])
    torch.testing.assert_close(grads[1], grads[0], rtol=1e-4, atol=1e-8)


def test_large_volume_grid_soft() -> None:
    from ace_kernels import ace_loss

    config = ACEConfig(binning="soft", to_onehot_y=True)
    x, y = make_inputs(InputSpec(1, 2, (256, 256, 256), seed=1), config, "cuda")
    leaf = x.clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(ace_loss(leaf, y, config, impl="v4_triton"), leaf)
    assert torch.isfinite(grad).all()


@pytest.mark.parametrize("binning", ["hard", "soft"])
@pytest.mark.parametrize("name", sorted(BWD_CONFIGS))
@pytest.mark.parametrize("spatial", [(7,), (33, 17, 5), (64, 64, 64)])
def test_v5_matches_v4(binning: str, name: str, spatial: tuple[int, ...]) -> None:
    """Fused activation == separate torch activation (loss and gradient)."""
    from ace_kernels import ace_loss

    config = ACEConfig.model_validate({"binning": binning, **BWD_CONFIGS[name]})
    spec = InputSpec(2, 5, spatial, seed=11, labels_as_index=config.to_onehot_y)
    x, y = make_inputs(spec, config, "cuda")
    out = []
    for impl in ("v4_triton", "v5_triton"):
        leaf = x.clone().requires_grad_(True)
        loss = ace_loss(leaf, y, config, impl=impl)
        upstream = torch.linspace(0.5, 2.0, loss.numel(), device="cuda").view_as(loss)
        out.append((loss.detach(), torch.autograd.grad(loss, leaf, upstream)[0]))
    torch.testing.assert_close(out[1][0], out[0][0], rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(out[1][1], out[0][1], rtol=1e-4, atol=1e-8)


HALF = [
    torch.float16,
    pytest.param(
        torch.bfloat16,
        marks=pytest.mark.skipif(
            not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
            reason="GPU without native bfloat16 (e.g. T4, sm_75)",
        ),
    ),
]


@pytest.mark.parametrize("dtype", HALF)
@pytest.mark.parametrize("binning", ["hard", "soft"])
def test_v5_half_precision_logits(dtype: torch.dtype, binning: Binning) -> None:
    """Half logits are read natively; all math is fp32 after the load."""
    from ace_kernels import ace_loss

    config = ACEConfig(binning=binning, to_onehot_y=True)
    x, y = make_inputs(InputSpec(2, 4, (48, 48, 32), seed=4), config, "cuda")
    x_half = x.to(dtype)
    results = []
    for inp in (x_half, x_half.float()):  # same values, half vs fp32 storage
        leaf = inp.clone().requires_grad_(True)
        loss = ace_loss(leaf, y, config, impl="v5_triton")
        (grad,) = torch.autograd.grad(loss, leaf)
        assert grad.dtype == inp.dtype
        results.append((loss.detach(), grad.float()))
    torch.testing.assert_close(results[0][0], results[1][0], rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(results[0][1], results[1][1], rtol=2e-2, atol=1e-6)


@pytest.mark.parametrize("binning", ["hard", "soft"])
def test_v5_saves_only_per_bin_tensors(binning: Binning) -> None:
    from ace_kernels import ace_loss
    from ace_kernels.instrumentation import measure_saved_tensors

    config = ACEConfig(binning=binning, to_onehot_y=True)
    b, c, k = 2, 4, config.num_bins
    x, y = make_inputs(InputSpec(b, c, (32, 32, 32)), config, "cuda")
    x.requires_grad_(True)
    _, stats = measure_saved_tensors(
        lambda: ace_loss(x, y, config, impl="v5_triton"), exclude=[x, y]
    )
    # coef and d (float32 [B, C, K]) + reduction jacobian (float32 [B, C])
    assert stats.extra_bytes == 2 * b * c * k * 4 + b * c * 4
