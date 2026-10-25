"""Registry of mL1-ACE implementations.

Every implementation has the same signature ``fn(input, target, config) -> loss``
so tests and benchmarks can iterate over all of them.
"""

from collections.abc import Callable
from dataclasses import dataclass

import torch

from ace_kernels.config import ACEConfig, Binning

ACEFn = Callable[[torch.Tensor, torch.Tensor, ACEConfig], torch.Tensor]

ALL_BINNINGS: frozenset[Binning] = frozenset({"hard", "soft"})
ALL_DEVICES: frozenset[str] = frozenset({"cpu", "cuda"})


@dataclass(frozen=True)
class ImplSpec:
    """A registered implementation and the settings it supports."""

    name: str
    fn: ACEFn
    binnings: frozenset[Binning]
    devices: frozenset[str]
    description: str
    slow: bool = False  # e.g. torch.compile: tests run only a small case subset

    def supports(self, binning: Binning, device: str) -> bool:
        return binning in self.binnings and device in self.devices


_REGISTRY: dict[str, ImplSpec] = {}


def register(
    name: str,
    *,
    description: str,
    binnings: frozenset[Binning] = ALL_BINNINGS,
    devices: frozenset[str] = ALL_DEVICES,
    slow: bool = False,
) -> Callable[[ACEFn], ACEFn]:
    """Decorator adding an implementation to the registry under ``name``."""

    def decorator(fn: ACEFn) -> ACEFn:
        if name in _REGISTRY:
            raise ValueError(f"implementation {name!r} already registered")
        _REGISTRY[name] = ImplSpec(name, fn, binnings, devices, description, slow)
        return fn

    return decorator


def get_impl(name: str) -> ImplSpec:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown implementation {name!r}; available: {sorted(_REGISTRY)}"
        ) from None


def available_impls(
    binning: Binning | None = None, device: str | None = None
) -> list[ImplSpec]:
    """Registered implementations, optionally filtered by binning and device."""
    return [
        spec
        for spec in _REGISTRY.values()
        if (binning is None or binning in spec.binnings)
        and (device is None or device in spec.devices)
    ]


def ace_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    config: ACEConfig | None = None,
    impl: str = "reference",
) -> torch.Tensor:
    """Compute the mL1-ACE loss with the chosen implementation.

    Args:
        input: logits or probabilities, shape ``[B, C, *spatial]``.
        target: one-hot ``[B, C, *spatial]``, or labels ``[B, 1, *spatial]``
            when ``config.to_onehot_y`` is set.
        config: loss options; defaults to ``ACEConfig()``.
        impl: registered implementation name.
    """
    config = config if config is not None else ACEConfig()
    spec = get_impl(impl)
    if not spec.supports(config.binning, input.device.type):
        raise ValueError(
            f"{impl!r} does not support binning={config.binning!r} "
            f"on device={input.device.type!r}"
        )
    return spec.fn(input, target, config)
