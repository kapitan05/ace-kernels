"""Triton kernels vs the PyTorch per-bin statistics (CUDA + Triton only)."""

import pytest
import torch

from ace_kernels import ACEConfig
from ace_kernels.binning import hard_stats
from ace_kernels.cases import InputKind, InputSpec, make_inputs
from ace_kernels.common import prepare
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
    torch.testing.assert_close(actual.sum_p, expected.sum_p, rtol=1e-5, atol=1e-4)
    torch.testing.assert_close(actual.sum_gt, expected.sum_gt, rtol=0, atol=0)
    assert torch.equal(actual.empty, expected.empty)


@pytest.mark.parametrize("include_background", [True, False])
def test_hard_stats_index_target_background(include_background: bool) -> None:
    from ace_kernels.kernels.hard_hist import hard_stats_triton

    config = ACEConfig(to_onehot_y=True, include_background=include_background)
    x, y = make_inputs(InputSpec(2, 5, (40, 40, 9), seed=5), config, "cuda")
    prep = prepare(x, y, config, keep_labels=True)
    expected = hard_stats(prep, config)
    actual = hard_stats_triton(prep, config)
    torch.testing.assert_close(actual.weight, expected.weight, rtol=0, atol=0)
    torch.testing.assert_close(actual.sum_p, expected.sum_p, rtol=1e-5, atol=1e-4)
    torch.testing.assert_close(actual.sum_gt, expected.sum_gt, rtol=0, atol=0)


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
    torch.testing.assert_close(grads[1], grads[0], rtol=1e-5, atol=1e-9)


def test_v4_sigmoid_single_channel() -> None:
    from ace_kernels import ace_loss

    config = ACEConfig(binning="hard", activation="sigmoid")
    x, y = make_inputs(InputSpec(2, 1, (40, 40, 8), seed=2), config, "cuda")
    grads = []
    for impl in ("v2_recompute", "v4_triton"):
        leaf = x.clone().requires_grad_(True)
        grads.append(torch.autograd.grad(ace_loss(leaf, y, config, impl=impl), leaf)[0])
    torch.testing.assert_close(grads[1], grads[0], rtol=1e-5, atol=1e-9)
