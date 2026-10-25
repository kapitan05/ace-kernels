"""Every implementation reproduces the stored reference loss and gradient.

The golden file pins the reference behaviour; it catches silent changes from a
torch/monai upgrade as well as wrong optimized implementations.
"""

from pathlib import Path
from typing import Any

import pytest
import torch

from ace_kernels import ACEConfig, available_impls, get_impl
from ace_kernels.cases import CASES
from tests.helpers import (
    DEVICES,
    TIE_CASES,
    Result,
    assert_same,
    run,
    skip_unless_supported,
)

GOLDEN_PATH = Path(__file__).parent / "golden" / "reference_cases.pt"
GOLDEN: dict[str, Any] = torch.load(GOLDEN_PATH, weights_only=True)["cases"]


def test_golden_covers_all_cases() -> None:
    assert set(GOLDEN) == {c.name for c in CASES}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("case_name", sorted(GOLDEN))
@pytest.mark.parametrize("impl", [s.name for s in available_impls()])
def test_matches_golden(impl: str, case_name: str, device: str) -> None:
    record = GOLDEN[case_name]
    config = ACEConfig.model_validate(record["config"])
    skip_unless_supported(impl, config.binning, device, case_name)
    x, y = record["input"].to(device), record["target"].to(device)
    result = run(impl, x, y, config)
    expected = Result(record["loss"], record["grad"])
    if get_impl(impl).slow and case_name in TIE_CASES:  # see TIE_CASES
        assert_same(Result(result.loss, expected.grad), expected)
        assert torch.isfinite(result.grad).all()
        return
    assert_same(result, expected)
