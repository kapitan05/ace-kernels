"""The reference agrees with an independent float64 restatement of the math.

This validates our reading of the loss (|P_k - G_k| / W_k per bin), which the
handwritten backward pass relies on.
"""

import numpy as np
import pytest
import torch

from ace_kernels.cases import CASES, Case, make_inputs
from tests.helpers import run
from tests.oracle import oracle_ace


def _one_hot_target(case: Case, y: torch.Tensor) -> torch.Tensor:
    if case.config.to_onehot_y and case.spec.channels > 1:
        return torch.zeros(case.spec.shape).scatter_(1, y, 1.0)
    return y


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_reference_matches_oracle(case: Case) -> None:
    x, y = make_inputs(case.spec, case.config)
    expected = oracle_ace(
        x.double().numpy(), _one_hot_target(case, y).double().numpy(), case.config
    )
    actual = run("reference", x, y, case.config).loss.double().numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
