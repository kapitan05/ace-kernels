"""Drop-in replacements for the reference ``HardL1ACELoss`` / ``SoftL1ACELoss``.

Same constructor (argument names, order, defaults, validation) and ``forward(input,
target)`` as the upstream classes, so switching is one import::

    from ace_kernels.module import HardL1ACELoss  # was: from losses.hardl1ace ...

Extra keyword-only argument ``impl``: ``"auto"`` (default) runs the fastest
implementation for the input's device (Triton ``v6_triton`` on CUDA, the
handwritten-backward PyTorch ``v2_recompute`` elsewhere); any registered name
(``ace_kernels.available_impls()``) can be forced, ``"reference"`` included.
"""

from collections.abc import Callable, Sequence

import torch
from monai.utils.enums import LossReduction
from torch.nn.modules.loss import _Loss

from ace_kernels.config import ACEConfig, Activation, Binning
from ace_kernels.kernels import HAS_TRITON
from ace_kernels.registry import get_impl

FAST_CUDA = "v6_triton"
FAST_OTHER = "v2_recompute"

Weight = Sequence[float] | float | int | torch.Tensor | None


def _weight(weight: Weight) -> tuple[float, ...] | float | None:
    if weight is None:
        return None
    if isinstance(weight, torch.Tensor):
        values = [float(w) for w in weight.flatten().tolist()]
        return values[0] if weight.dim() == 0 else tuple(values)
    if isinstance(weight, int | float):
        return float(weight)
    return tuple(float(w) for w in weight)


class _ACELoss(_Loss):
    binning: Binning

    def __init__(
        self,
        *,
        num_bins: int,
        include_background: bool,
        to_onehot_y: bool,
        sigmoid: bool,
        softmax: bool,
        other_act: Callable[[torch.Tensor], torch.Tensor] | None,
        reduction: LossReduction | str,
        weight: Weight,
        empty_weight: float,
        right: bool,
        ignore_empty_classes: bool,
        impl: str,
    ) -> None:
        super().__init__(reduction=LossReduction(reduction).value)
        if other_act is not None and not callable(other_act):
            raise TypeError(
                f"other_act must be None or callable but is {type(other_act).__name__}."
            )
        if int(sigmoid) + int(softmax) + int(other_act is not None) > 1:
            raise ValueError(
                "Incompatible values: more than 1 of [sigmoid=True, softmax=True, "
                "other_act is not None]."
            )
        activation: Activation = (
            "sigmoid" if sigmoid else "softmax" if softmax else "none"
        )
        self.other_act = other_act
        self.impl = impl
        self.config = ACEConfig(
            binning=self.binning,
            num_bins=num_bins,
            activation=activation,
            to_onehot_y=to_onehot_y,
            include_background=include_background,
            ignore_empty_classes=ignore_empty_classes,
            reduction=LossReduction(reduction).value,
            weight=_weight(weight),
            right=right,
            empty_weight=empty_weight,
        )
        if impl != "auto":
            get_impl(impl)  # fail early on an unknown name

    def resolve_impl(self, device: torch.device) -> str:
        """Implementation used for inputs on ``device``."""
        if self.impl != "auto":
            return self.impl
        return FAST_CUDA if device.type == "cuda" and HAS_TRITON else FAST_OTHER

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """``input``: logits ``[B, C, *spatial]`` (probabilities when no activation);
        ``target``: one-hot ``[B, C, ...]`` or class indices ``[B, 1, ...]`` with
        ``to_onehot_y``."""
        if self.other_act is not None:
            input = self.other_act(input)
        loss: torch.Tensor = get_impl(self.resolve_impl(input.device)).fn(
            input, target, self.config
        )
        return loss


class HardL1ACELoss(_ACELoss):
    """Hard-binned L1 average calibration error (mL1-ACE); upstream signature."""

    binning: Binning = "hard"

    def __init__(
        self,
        num_bins: int = 20,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Callable[[torch.Tensor], torch.Tensor] | None = None,
        reduction: LossReduction | str = LossReduction.MEAN,
        weight: Weight = None,
        right: bool = False,
        ignore_empty_classes: bool = True,
        *,
        impl: str = "auto",
    ) -> None:
        super().__init__(
            num_bins=num_bins,
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            sigmoid=sigmoid,
            softmax=softmax,
            other_act=other_act,
            reduction=reduction,
            weight=weight,
            empty_weight=0.01,  # unused by hard binning
            right=right,
            ignore_empty_classes=ignore_empty_classes,
            impl=impl,
        )


class SoftL1ACELoss(_ACELoss):
    """Soft-binned L1 average calibration error (mL1-ACE); upstream signature."""

    binning: Binning = "soft"

    def __init__(
        self,
        num_bins: int = 20,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Callable[[torch.Tensor], torch.Tensor] | None = None,
        reduction: LossReduction | str = LossReduction.MEAN,
        weight: Weight = None,
        empty_weight: float = 0.01,
        right: bool = False,
        ignore_empty_classes: bool = True,
        *,
        impl: str = "auto",
    ) -> None:
        super().__init__(
            num_bins=num_bins,
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            sigmoid=sigmoid,
            softmax=softmax,
            other_act=other_act,
            reduction=reduction,
            weight=weight,
            empty_weight=empty_weight,
            right=right,
            ignore_empty_classes=ignore_empty_classes,
            impl=impl,
        )
