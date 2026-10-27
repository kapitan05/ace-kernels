"""Drop-in modules vs the upstream ``HardL1ACELoss`` / ``SoftL1ACELoss`` classes."""

import inspect
import warnings

import pytest
import torch

from ace_kernels.cases import CASES, Case, make_inputs
from ace_kernels.module import HardL1ACELoss, SoftL1ACELoss, _ACELoss
from ace_reference.hardl1ace import HardL1ACELoss as RefHard
from ace_reference.softl1ace import SoftL1ACELoss as RefSoft
from tests.helpers import DEVICES, Result, assert_same

PAIRS = [(HardL1ACELoss, RefHard), (SoftL1ACELoss, RefSoft)]


@pytest.mark.parametrize(("ours", "ref"), PAIRS)
def test_same_signature(ours: type, ref: type) -> None:
    """Same parameter names, order and defaults; ``impl`` is extra, keyword-only."""
    mine = list(inspect.signature(ours).parameters.values())
    theirs = list(inspect.signature(ref).parameters.values())
    assert mine[-1].name == "impl" and mine[-1].kind is inspect.Parameter.KEYWORD_ONLY
    assert [(p.name, p.default) for p in mine[:-1]] == [
        (p.name, p.default) for p in theirs
    ]


def _kwargs(case: Case) -> dict[str, object]:
    c = case.config
    kwargs: dict[str, object] = {
        "num_bins": c.num_bins,
        "include_background": c.include_background,
        "to_onehot_y": c.to_onehot_y,
        "sigmoid": c.activation == "sigmoid",
        "softmax": c.activation == "softmax",
        "reduction": c.reduction,
        "weight": c.weight,
        "right": c.right,
        "ignore_empty_classes": c.ignore_empty_classes,
    }
    if c.binning == "soft":
        kwargs["empty_weight"] = c.empty_weight
    return kwargs


def _run(loss_fn: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> Result:
    leaf = x.detach().clone().requires_grad_(True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # reference warns for C=1
        loss = loss_fn(leaf, y)
    torch.autograd.backward(loss.sum())
    assert leaf.grad is not None
    return Result(loss.detach(), leaf.grad)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_module_matches_reference(case: Case, device: str) -> None:
    ours, ref = PAIRS[0] if case.config.binning == "hard" else PAIRS[1]
    x, y = make_inputs(case.spec, case.config, device)
    expected = _run(ref(**_kwargs(case)), x, y)
    assert_same(_run(ours(**_kwargs(case)), x, y), expected)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(("ours", "ref"), PAIRS)
def test_other_act(ours: type, ref: type, device: str) -> None:
    case = next(c for c in CASES if c.name == "paper-3d-hard")
    x, y = make_inputs(case.spec, case.config, device)
    act = torch.nn.Softmax(dim=1)
    expected = _run(ref(to_onehot_y=True, other_act=act), x, y)
    assert_same(_run(ours(to_onehot_y=True, other_act=act), x, y), expected)


def test_auto_impl_by_device() -> None:
    loss = HardL1ACELoss(impl="auto")
    assert loss.resolve_impl(torch.device("cpu")) == "v2_recompute"
    assert HardL1ACELoss(impl="reference").resolve_impl(torch.device("cpu")) == (
        "reference"
    )


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"sigmoid": True, "softmax": True}, ValueError),
        ({"other_act": 3}, TypeError),
        ({"impl": "no-such-impl"}, KeyError),
        ({"reduction": "max"}, ValueError),
    ],
)
def test_invalid_arguments(kwargs: dict[str, object], error: type) -> None:
    with pytest.raises(error):
        HardL1ACELoss(**kwargs)  # type: ignore[arg-type]


def test_tensor_weight() -> None:
    weights = torch.tensor([0.1, 1.0, 2.0, 0.5])
    module: _ACELoss = SoftL1ACELoss(weight=weights)
    assert module.config.weight == pytest.approx((0.1, 1.0, 2.0, 0.5))
