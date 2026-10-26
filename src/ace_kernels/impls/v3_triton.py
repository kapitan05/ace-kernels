"""Triton implementations (float32, CUDA; registered only where Triton exists).

- ``v3_triton`` (Sprint 5): v2_recompute with the forward's per-bin sums from
  per-block private histograms (``kernels/hard_hist.py``) instead of the contended
  ``scatter_add`` (83-92% of GPU time in v1/v2). Backward: v2's PyTorch backward.
- ``v4_triton`` (Sprint 6): + the backward as one Triton pass
  (``kernels/hard_bwd.py``): gather ``coef`` per voxel and the softmax backward in
  registers. Saves only ``coef``, ``d``, presence and a ``[B, C]`` jacobian.
- Sprint 7: both cover soft binning (``kernels/soft_hist.py``, ``soft_bwd.py``).

float64 inputs (gradcheck) fall back to the PyTorch paths.
"""

import torch

from ace_kernels.config import ACEConfig
from ace_kernels.impls.v2_autograd import _apply
from ace_kernels.kernels import HAS_TRITON
from ace_kernels.registry import register

if HAS_TRITON:

    @register(
        "v3_triton",
        description="Triton private-histogram forward + v2 PyTorch backward",
        devices=frozenset({"cuda"}),
    )
    def v3_triton(
        input: torch.Tensor, target: torch.Tensor, config: ACEConfig
    ) -> torch.Tensor:
        if input.dtype == torch.float64:  # gradcheck path: kernel is float32-only
            return _apply(input, target, config, "recompute", "torch")
        return _apply(input, target, config, "recompute", "triton")

    @register(
        "v4_triton",
        description="Triton private-histogram forward + Triton backward",
        devices=frozenset({"cuda"}),
    )
    def v4_triton(
        input: torch.Tensor, target: torch.Tensor, config: ACEConfig
    ) -> torch.Tensor:
        if input.dtype == torch.float64:
            return _apply(input, target, config, "recompute", "torch")
        return _apply(input, target, config, "triton_bwd", "triton")
