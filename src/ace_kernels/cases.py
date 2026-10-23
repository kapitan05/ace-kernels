"""Deterministic input generation and the named correctness cases.

Shared by tests, the golden-file script and (later) benchmarks.
"""

from dataclasses import dataclass, field
from typing import Literal

import torch

from ace_kernels.config import ACEConfig

InputKind = Literal["logits", "probs", "constant", "bin_edges", "extreme"]


@dataclass(frozen=True)
class InputSpec:
    """How to build ``(input, target)`` for one case.

    kind:
        logits    - Gaussian logits times ``logit_scale`` (use with an activation).
        probs     - probabilities in (0, 1) given directly (activation "none").
        constant  - every voxel has probability ``constant_value``.
        bin_edges - probabilities exactly on the hard-bin edges k / num_bins.
        extreme   - probabilities exactly 0 or 1.
    labels_as_index:
        True  -> target is ``[B, 1, *spatial]`` class indices (for ``to_onehot_y``).
        False -> target is one-hot ``[B, C, *spatial]`` float.
    missing_classes: class indices that never appear in the target.
    """

    batch: int
    channels: int
    spatial: tuple[int, ...]
    seed: int = 0
    kind: InputKind = "logits"
    logit_scale: float = 2.0
    constant_value: float = 0.5
    labels_as_index: bool = True
    missing_classes: tuple[int, ...] = field(default_factory=tuple)

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.batch, self.channels, *self.spatial)


def _labels(spec: InputSpec, gen: torch.Generator) -> torch.Tensor:
    allowed = [c for c in range(spec.channels) if c not in spec.missing_classes]
    if not allowed:
        raise ValueError("at least one class must be present")
    shape = (spec.batch, 1, *spec.spatial)
    idx = torch.randint(0, len(allowed), shape, generator=gen)
    return torch.tensor(allowed)[idx]


def _input(spec: InputSpec, num_bins: int, gen: torch.Generator) -> torch.Tensor:
    shape = spec.shape
    if spec.kind == "logits":
        return torch.randn(shape, generator=gen) * spec.logit_scale
    if spec.kind == "probs":
        return torch.rand(shape, generator=gen).clamp(1e-4, 1 - 1e-4)
    if spec.kind == "constant":
        return torch.full(shape, spec.constant_value)
    if spec.kind == "bin_edges":
        edges = torch.arange(num_bins + 1, dtype=torch.float32) / num_bins
        idx = torch.randint(0, num_bins + 1, shape, generator=gen)
        return edges[idx]
    if spec.kind == "extreme":
        return torch.randint(0, 2, shape, generator=gen).float()
    raise ValueError(f"unknown input kind {spec.kind!r}")


def make_inputs(
    spec: InputSpec, config: ACEConfig, device: str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build ``(input, target)`` on ``device``; the input is a fresh leaf tensor."""
    gen = torch.Generator().manual_seed(spec.seed)
    x = _input(spec, config.num_bins, gen)
    labels = _labels(spec, gen)
    if spec.channels == 1:  # binary task: random foreground mask
        target = torch.randint(0, 2, labels.shape, generator=gen).float()
    elif spec.labels_as_index:
        target = labels
    else:
        target = torch.zeros(spec.shape).scatter_(1, labels, 1.0)
    return x.to(device), target.to(device)


@dataclass(frozen=True)
class Case:
    name: str
    spec: InputSpec
    config: ACEConfig


def _both(name: str, spec: InputSpec, **cfg: object) -> list[Case]:
    return [
        Case(f"{name}-{b}", spec, ACEConfig.model_validate({"binning": b, **cfg}))
        for b in ("hard", "soft")
    ]


_PAPER: dict[str, object] = {"to_onehot_y": True, "activation": "softmax"}
_PROBS: dict[str, object] = {"activation": "none"}
_BASE = InputSpec(batch=2, channels=4, spatial=(8, 8, 8))
_ONEHOT = InputSpec(batch=2, channels=4, spatial=(6, 6, 6), labels_as_index=False)


def _cases() -> list[Case]:
    cases: list[Case] = []
    cases += _both("paper-3d", _BASE, **_PAPER)
    cases += _both("2d", InputSpec(2, 3, (16, 16), seed=1), **_PAPER)
    cases += _both("1d", InputSpec(1, 2, (50,), seed=2), **_PAPER)
    cases += _both("many-classes", InputSpec(1, 16, (4, 4, 4), seed=3), **_PAPER)
    cases += _both("onehot-target", _ONEHOT, activation="softmax")
    cases += _both(
        "saturated-logits", InputSpec(2, 4, (6, 6, 6), 4, logit_scale=30.0), **_PAPER
    )
    empty = InputSpec(2, 4, (6, 6, 6), seed=5, missing_classes=(2,))
    cases += _both("empty-class-ignored", empty, **_PAPER)
    cases += _both("empty-class-kept", empty, **_PAPER, ignore_empty_classes=False)
    cases += _both("no-background", _BASE, **_PAPER, include_background=False)
    cases += _both("class-weights", _BASE, **_PAPER, weight=(0.1, 1.0, 2.0, 0.5))
    cases += _both("scalar-weight", _BASE, **_PAPER, weight=2.0)
    cases += _both("reduction-sum", _BASE, **_PAPER, reduction="sum")
    cases += _both("reduction-none", _BASE, **_PAPER, reduction="none")
    cases += _both(
        "sigmoid-binary", InputSpec(2, 1, (8, 8, 8), seed=6), activation="sigmoid"
    )
    cases += _both("one-bin", _BASE, **_PAPER, num_bins=1)
    cases += _both(
        "fifty-bins", InputSpec(1, 4, (12, 12, 12), seed=7), **_PAPER, num_bins=50
    )
    probs = InputSpec(2, 4, (6, 6, 6), seed=8, kind="probs", labels_as_index=False)
    cases += _both("raw-probs", probs, **_PROBS)
    const = InputSpec(2, 4, (6, 6, 6), seed=9, kind="constant", labels_as_index=False)
    cases += _both("constant-probs", const, **_PROBS)
    edges = InputSpec(2, 4, (6, 6, 6), seed=10, kind="bin_edges", labels_as_index=False)
    cases += _both("bin-edges-left", edges, **_PROBS)
    cases += _both("bin-edges-right", edges, **_PROBS, right=True)
    extreme = InputSpec(2, 4, (6, 6, 6), seed=11, kind="extreme", labels_as_index=False)
    cases += _both("extreme-probs", extreme, **_PROBS)
    # 10.0 drops several sparse bins as "empty" (0.01 drops none for this input)
    high_empty = {"binning": "soft", "empty_weight": 10.0, **_PAPER}
    cases.append(
        Case("soft-high-empty-weight", _BASE, ACEConfig.model_validate(high_empty))
    )
    return cases


CASES: list[Case] = _cases()
