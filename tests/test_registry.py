import pytest
import torch
from pydantic import ValidationError

from ace_kernels import ACEConfig, ace_loss, available_impls, get_impl
from ace_kernels.registry import _REGISTRY, ImplSpec, register


def test_reference_registered_for_all_settings() -> None:
    spec = get_impl("reference")
    for binning in ("hard", "soft"):
        for device in ("cpu", "cuda"):
            assert spec.supports(binning, device)


def test_unknown_impl_raises() -> None:
    with pytest.raises(KeyError, match="unknown implementation"):
        get_impl("nope")


def test_duplicate_registration_raises() -> None:
    with pytest.raises(ValueError, match="already registered"):
        register("reference", description="dup")(get_impl("reference").fn)


def test_unsupported_binning_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def _impl(x: torch.Tensor, y: torch.Tensor, c: ACEConfig) -> torch.Tensor:
        return x.sum()

    spec = ImplSpec(
        "_hard_only_test", _impl, frozenset({"hard"}), frozenset({"cpu"}), "test"
    )
    monkeypatch.setitem(_REGISTRY, spec.name, spec)  # removed after the test
    x = torch.rand(1, 2, 4)
    with pytest.raises(ValueError, match="does not support"):
        ace_loss(x, x, ACEConfig(binning="soft"), impl="_hard_only_test")
    assert "_hard_only_test" in {s.name for s in available_impls(binning="hard")}
    assert "_hard_only_test" not in {s.name for s in available_impls(binning="soft")}


def test_config_rejects_unknown_and_invalid_fields() -> None:
    with pytest.raises(ValidationError):
        ACEConfig.model_validate({"num_binz": 20})
    with pytest.raises(ValidationError):
        ACEConfig(num_bins=0)
