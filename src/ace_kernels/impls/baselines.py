"""Off-the-shelf speed/memory tricks applied to the unchanged reference.

They set the bar our own implementations must beat:
- ``reference_compile``: torch.compile (Inductor fuses what it can trace).
- ``reference_checkpoint``: torch.utils.checkpoint (saves only the inputs, reruns
  the forward during backward: less memory, more time).
CUDA graphs are a benchmark option (``cuda_graph: true``), not an implementation.
"""

import torch
from torch.utils.checkpoint import checkpoint

from ace_kernels.config import ACEConfig
from ace_kernels.impls.reference import reference_ace
from ace_kernels.registry import register

_compiled = torch.compile(reference_ace)


@register(
    "reference_compile",
    description="Reference under torch.compile (default mode)",
    slow=True,
)
def reference_compile(
    input: torch.Tensor, target: torch.Tensor, config: ACEConfig
) -> torch.Tensor:
    loss: torch.Tensor = _compiled(input, target, config)
    return loss


@register(
    "reference_checkpoint",
    description="Reference under torch.utils.checkpoint (recompute in backward)",
)
def reference_checkpoint(
    input: torch.Tensor, target: torch.Tensor, config: ACEConfig
) -> torch.Tensor:
    loss = checkpoint(reference_ace, input, target, config, use_reentrant=False)
    assert isinstance(loss, torch.Tensor)
    return loss
