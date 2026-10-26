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
        activation="softmax" if kind == "logits" else "none",
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
