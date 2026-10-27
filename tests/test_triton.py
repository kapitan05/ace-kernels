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
    """Within 1e-5 of the float64 truth and not meaningfully less accurate than
    PyTorch fp32. At noise level the two swap places between runs (atomic order;
    measured on a T4: 1.56e-6 vs 1.53e-6), hence the factor 2."""
    scale = truth.abs().clamp_min(1.0)
    err_triton = ((triton_val.double() - truth).abs() / scale).max().item()
    err_torch = ((torch_val.double() - truth).abs() / scale).max().item()
    assert err_triton <= 1e-5, (err_triton, err_torch)
    assert err_triton <= max(2 * err_torch, 1e-6), (err_triton, err_torch)


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


def _near_boundary_voxels(x: torch.Tensor, config: ACEConfig) -> torch.Tensor:
    """[B, N] bool: some channel's probability lies within 1e-6 of a boundary
    where its bin assignment changes (hard: bin edges; soft: half boundaries)."""
    from ace_kernels.common import activate, hard_edges, soft_half_boundaries

    p = activate(x, config).float().flatten(2)  # [B, C, N]
    if config.binning == "hard":
        bounds = hard_edges(config.num_bins, x.device)
    else:
        bounds = soft_half_boundaries(config.num_bins, x.device)[1:]
    dist = (p.unsqueeze(-1) - bounds).abs().amin(-1)  # [B, C, N]
    return (dist < 1e-6).any(1)


@pytest.mark.parametrize("binning", ["hard", "soft"])
@pytest.mark.parametrize("name", sorted(BWD_CONFIGS))
@pytest.mark.parametrize("spatial", [(7,), (33, 17, 5), (64, 64, 64)])
def test_v5_matches_v4(binning: str, name: str, spatial: tuple[int, ...]) -> None:
    """Fused activation == separate torch.softmax, except at boundary voxels.

    In-kernel softmax can differ from torch.softmax by ~1 ulp, so a probability
    within ~1e-7 of a bin boundary can be binned differently (measured on a T4: one
    voxel of 2.6M at 64^3, gradient changed on all its channels via the softmax
    coupling). Gradients may therefore differ only at voxels near a boundary, and
    those must be rare; the loss must match tightly.
    """
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
    g5, g4 = out[1][1].flatten(2), out[0][1].flatten(2)
    # rtol 1e-3 / atol 1e-7: few-voxels-per-bin soft cases amplify ulp differences
    mismatch = ~torch.isclose(g5, g4, rtol=1e-3, atol=1e-7)  # [B, C, N]
    bad_voxels = mismatch.any(1)
    near = _near_boundary_voxels(x, config)
    assert not (bad_voxels & ~near).any(), "gradient differs away from bin boundaries"
    # mismatches themselves must be rare (being *near* a boundary is not: with a
    # +-1e-6 window, C * K edges and ~uniform probabilities, ~1e-3 of voxels are)
    assert bad_voxels.float().mean().item() <= 1e-3


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
    # coef and d (float32 [B, C, K]) + reduction jacobian (float32 [B, C]); plus
    # a [B, C] temporary from differentiating the class reduction in forward
    expected = 2 * b * c * k * 4 + b * c * 4
    assert expected <= stats.extra_bytes <= expected + 1024


# --- Sprint 11: v6 (direct index, tensor-core reduction, fused epilogue) ---


def _near_edge_values(table: torch.Tensor, ulps: int = 8) -> torch.Tensor:
    """Every float32 within ``ulps`` of every table entry, plus 0, 1 and 1e5 uniform."""
    vals = [table, torch.tensor([0.0, -0.0, 1.0, 2.0, -1.0], device=table.device)]
    up, down = table.clone(), table.clone()
    for _ in range(ulps):
        up = torch.nextafter(up, torch.full_like(up, 2.0))
        down = torch.nextafter(down, torch.full_like(down, -1.0))
        vals += [up, down]
    gen = torch.Generator(device=table.device).manual_seed(0)
    vals.append(torch.rand(100_000, generator=gen, device=table.device))
    return torch.cat(vals)


@pytest.mark.parametrize("num_bins", [1, 2, 3, 7, 20, 33, 50, 100])
@pytest.mark.parametrize("right", [False, True])
@pytest.mark.parametrize("binning", ["hard", "soft"])
def test_direct_index_exact(num_bins: int, right: bool, binning: Binning) -> None:
    """O(1) index == the K-compare loop (== torch.bucketize) at every near-edge
    float32. Hard: counts exact (one moved voxel changes a count by 1). Soft:
    weights within atomic-order rounding (one moved voxel: ~0.5 of ~1e3-1e5)."""
    from ace_kernels.common import hard_edges, soft_half_boundaries
    from ace_kernels.kernels.hard_hist import hard_stats_fused
    from ace_kernels.kernels.soft_hist import soft_stats_fused

    table = (
        hard_edges(num_bins, torch.device("cuda"))
        if binning == "hard"
        else soft_half_boundaries(num_bins, torch.device("cuda"))
    )
    p = _near_edge_values(table).view(1, 1, -1)
    config = ACEConfig(binning=binning, num_bins=num_bins, right=right)
    config = config.model_copy(update={"activation": "none"})
    target = (torch.arange(p.numel(), device="cuda") % 3 == 0).float().view_as(p)
    fused = hard_stats_fused if binning == "hard" else soft_stats_fused
    loop = fused(p, target, config, direct=False)
    direct = fused(p, target, config, direct=True)
    if binning == "hard":
        assert torch.equal(direct.weight, loop.weight)
    # float atomics add in a different order on every run: not bit-identical
    for a, b in ((direct.weight, loop.weight), (direct.sum_p, loop.sum_p)):
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-3)
    torch.testing.assert_close(direct.sum_gt, loop.sum_gt, rtol=1e-6, atol=1e-3)


@pytest.mark.parametrize("binning", ["hard", "soft"])
@pytest.mark.parametrize("channels", [2, 4, 16])
@pytest.mark.parametrize("num_bins", [1, 20, 50])
@pytest.mark.parametrize("spatial", [(7,), (100003,), (40, 40, 40)])
def test_dot_reduction_matches_sums(
    binning: Binning, channels: int, num_bins: int, spatial: tuple[int, ...]
) -> None:
    """Tensor-core reduction: counts / target sums exact, p sums as fp32 sums."""
    from ace_kernels.kernels.act import as_kernel_input
    from ace_kernels.kernels.hard_hist import hard_stats_fused
    from ace_kernels.kernels.soft_hist import soft_stats_fused

    config = ACEConfig(binning=binning, num_bins=num_bins, to_onehot_y=True)
    x, y = make_inputs(InputSpec(2, channels, spatial, seed=5), config, "cuda")
    xk, yk = as_kernel_input(x), y.reshape(2, -1)
    fused = hard_stats_fused if binning == "hard" else soft_stats_fused
    sums = fused(xk, yk, config, direct=True, dot=False)
    dot = fused(xk, yk, config, direct=True, dot=True)
    if binning == "hard":
        assert torch.equal(dot.weight, sums.weight)
        assert torch.equal(dot.sum_gt, sums.sum_gt)
    for a, b in ((dot.weight, sums.weight), (dot.sum_gt, sums.sum_gt)):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)
    # two fp32 summation orders over up to 1e5 values: ~1e-6 apart (measured 2.2e-6)
    torch.testing.assert_close(dot.sum_p, sums.sum_p, rtol=1e-5, atol=1e-5)


V6 = ["v6_index", "v6_dot", "v6_triton", "v6_precise"]


@pytest.mark.parametrize("impl", V6)
@pytest.mark.parametrize("binning", ["hard", "soft"])
@pytest.mark.parametrize("name", sorted(BWD_CONFIGS))
@pytest.mark.parametrize("spatial", [(7,), (33, 17, 5), (64, 64, 64)])
def test_v6_matches_v5(
    impl: str, binning: str, name: str, spatial: tuple[int, ...]
) -> None:
    """Same bins as v5 (exact index, same exp): loss tight; gradients equal up to
    fp32 summation order, except rare sign flips of |mean p - mean y| ~ 0 bins."""
    from ace_kernels import ace_loss

    config = ACEConfig.model_validate({"binning": binning, **BWD_CONFIGS[name]})
    spec = InputSpec(2, 5, spatial, seed=11, labels_as_index=config.to_onehot_y)
    x, y = make_inputs(spec, config, "cuda")
    out = []
    for name_ in ("v5_triton", impl):
        leaf = x.clone().requires_grad_(True)
        loss = ace_loss(leaf, y, config, impl=name_)
        upstream = torch.linspace(0.5, 2.0, loss.numel(), device="cuda").view_as(loss)
        out.append((loss.detach(), torch.autograd.grad(loss, leaf, upstream)[0]))
    torch.testing.assert_close(out[1][0], out[0][0], rtol=1e-5, atol=1e-7)
    mismatch = ~torch.isclose(out[1][1], out[0][1], rtol=1e-3, atol=1e-7)
    if impl == "v6_index":  # identical arithmetic
        assert not mismatch.any()
    assert mismatch.flatten(2).any(1).float().mean().item() <= 1e-3


@pytest.mark.parametrize("impl", ["v6_triton", "v6_precise"])
@pytest.mark.parametrize("binning", ["hard", "soft"])
def test_v6_launches(binning: Binning, impl: str) -> None:
    """Fused epilogue: a handful of launches per fwd + bwd (v5: ~55)."""
    from torch.profiler import ProfilerActivity, profile

    from ace_kernels import ace_loss

    config = ACEConfig(binning=binning, to_onehot_y=True)
    x, y = make_inputs(InputSpec(2, 4, (32, 32, 32)), config, "cuda")
    x.requires_grad_(True)
    for _ in range(2):  # autotune + compile outside the profile
        torch.autograd.backward(ace_loss(x, y, config, impl=impl))
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        torch.autograd.backward(ace_loss(x, y, config, impl=impl))
        torch.cuda.synchronize()
    kernels = [e for e in prof.events() if e.device_type.name == "CUDA"]
    assert len(kernels) <= 8, [e.name for e in kernels]


@pytest.mark.parametrize("impl", ["v6_triton", "v6_precise"])
@pytest.mark.parametrize("binning", ["hard", "soft"])
def test_v6_compiles_fullgraph(binning: Binning, impl: str) -> None:
    """torch.compile of network + loss without graph breaks; same result as eager."""
    from ace_kernels import ace_loss

    config = ACEConfig(binning=binning, to_onehot_y=True)
    x, y = make_inputs(InputSpec(2, 4, (16, 16, 16), seed=2), config, "cuda")
    net = torch.nn.Conv3d(4, 4, 3, padding=1).cuda()

    def step(inp: torch.Tensor) -> torch.Tensor:
        return ace_loss(net(inp), y, config, impl=impl)

    results = []
    for fn in (step, torch.compile(step, fullgraph=True)):
        net.zero_grad()
        loss = fn(x)
        torch.autograd.backward(loss)
        assert net.weight.grad is not None
        results.append((loss.detach(), net.weight.grad.clone()))
    torch.testing.assert_close(results[1][0], results[0][0], rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(results[1][1], results[0][1], rtol=1e-3, atol=1e-6)


@pytest.mark.parametrize("binning", ["hard", "soft"])
def test_v6_precise_exp_matches_torch_softmax(binning: Binning) -> None:
    """libdevice exp: in-kernel probabilities bin like torch.softmax (v4) far more
    often than the fast exp; measured, not required equal."""
    from ace_kernels.impls.v6_triton import Options, _forward

    config = ACEConfig(binning=binning, to_onehot_y=True)
    x, y = make_inputs(InputSpec(2, 4, (64, 64, 64), seed=3), config, "cuda")
    ref = _forward(x, y, config, Options(direct=True, precise=False))
    precise = _forward(x, y, config, Options(direct=True, precise=True))
    torch.testing.assert_close(precise[0], ref[0], rtol=1e-5, atol=1e-7)
