"""Configuration shared by every mL1-ACE implementation."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Binning = Literal["hard", "soft"]
Activation = Literal["softmax", "sigmoid", "none"]
Reduction = Literal["mean", "sum", "none"]


class ACEConfig(BaseModel):
    """Options of the mL1-ACE loss, mirroring ``HardL1ACELoss`` / ``SoftL1ACELoss``.

    Defaults follow the paper's training configs
    (``bundle_templates/configs/loss/*l1ace*.yaml`` upstream), except that the
    target one-hot conversion is opt-in like in the reference loss classes.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    binning: Binning = "hard"
    num_bins: int = Field(default=20, ge=1)
    activation: Activation = "softmax"
    to_onehot_y: bool = False
    include_background: bool = True
    ignore_empty_classes: bool = True
    reduction: Reduction = "mean"
    weight: tuple[float, ...] | float | None = None
    right: bool = False
    empty_weight: float = Field(default=0.01, ge=0.0)
