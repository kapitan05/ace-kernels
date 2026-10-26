"""v3_triton: v2_recompute with the forward's per-bin sums from a Triton kernel.

The forward's contended ``scatter_add`` (83-92% of GPU time in v1/v2, Sprint 3)
is replaced by per-block private histograms (``kernels/hard_hist.py``). The
backward is still v2's handwritten PyTorch backward (Triton backward: Sprint 6).
Hard binning, float32, CUDA only; registered only where Triton is installed.
"""

import torch

from ace_kernels.config import ACEConfig
from ace_kernels.impls.v2_autograd import _apply
from ace_kernels.kernels import HAS_TRITON
from ace_kernels.registry import register

if HAS_TRITON:

    @register(
        "v3_triton",
        description="Triton private-histogram forward + v2 handwritten backward",
        binnings=frozenset({"hard"}),
        devices=frozenset({"cuda"}),
    )
    def v3_triton(
        input: torch.Tensor, target: torch.Tensor, config: ACEConfig
    ) -> torch.Tensor:
        if input.dtype == torch.float64:  # gradcheck path: kernel is float32-only
            return _apply(input, target, config, "recompute", "torch")
        return _apply(input, target, config, "recompute", "triton")
