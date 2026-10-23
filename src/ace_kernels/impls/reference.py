"""Adapter exposing the frozen upstream losses through the registry."""

import torch

from ace_kernels.config import ACEConfig
from ace_kernels.registry import register
from ace_reference import HardL1ACELoss, SoftL1ACELoss


@register("reference", description="Original upstream loss (Python loop, autograd)")
def reference_ace(
    input: torch.Tensor, target: torch.Tensor, config: ACEConfig
) -> torch.Tensor:
    weight = list(config.weight) if isinstance(config.weight, tuple) else config.weight
    common = dict(
        num_bins=config.num_bins,
        include_background=config.include_background,
        to_onehot_y=config.to_onehot_y,
        sigmoid=config.activation == "sigmoid",
        softmax=config.activation == "softmax",
        reduction=config.reduction,
        weight=weight,
        right=config.right,
        ignore_empty_classes=config.ignore_empty_classes,
    )
    # A fresh module per call: HardL1ACELoss mutates its class_weight buffer.
    loss_fn: torch.nn.Module
    if config.binning == "hard":
        loss_fn = HardL1ACELoss(**common)
    else:
        loss_fn = SoftL1ACELoss(**common, empty_weight=config.empty_weight)
    loss: torch.Tensor = loss_fn.to(input.device)(input, target)
    return loss
