"""Theoretical lower bounds used to judge measured times.

The loss is memory-bound (a few FLOPs per byte), so the floor is the time to
move the unavoidable bytes at peak DRAM bandwidth:

- fwd:     read input + read target once.
- fwd_bwd: the bin statistics depend on the whole volume, so the gradient needs
  a second pass: read input + target twice and write the input gradient once.
  (Holds for both "recompute in backward" and "gradient in forward pass 2".)

A second floor comes from kernel launches: ``kernels * launch_us``. A run far
above both floors is neither bandwidth- nor launch-bound, i.e. it does extra
traffic (temporaries, copies) or synchronizes.
"""

from typing import Literal

Mode = Literal["fwd", "fwd_bwd"]

# Typical CUDA launch overhead per kernel (5-10 us is the usual range).
LAUNCH_US = 5.0


def ideal_bytes(input_bytes: int, target_bytes: int, mode: Mode) -> int:
    if mode == "fwd":
        return input_bytes + target_bytes
    return 2 * (input_bytes + target_bytes) + input_bytes


def bandwidth_floor_ms(nbytes: int, peak_gbps: float) -> float:
    return nbytes / (peak_gbps * 1e9) * 1e3


def launch_floor_ms(kernels: int, launch_us: float = LAUNCH_US) -> float:
    return kernels * launch_us / 1e3


def effective_gbps(nbytes: int, ms: float) -> float:
    """Ideal bytes divided by measured time: the bandwidth the run "achieved"."""
    return nbytes / (ms / 1e3) / 1e9
