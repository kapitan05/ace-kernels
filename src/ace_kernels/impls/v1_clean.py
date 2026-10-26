"""v1_clean: v1_vectorized plus removal of avoidable work (still default autograd).

Changes on top of v1_vectorized, each removing memory traffic or a host sync:
- Per-bin statistics are plain sums (``scatter_add``): sum of p, sum of y and the
  count / weight sum. Means are taken on the tiny ``[B, C, K]`` result. Hard
  binning no longer uses ``scatter_reduce(mean)``.
- With an index target (``to_onehot_y``) no one-hot tensor is built: each voxel
  adds its bin weight only to the row of its true class (gather + scatter on
  ``[B, N]`` instead of ``[B, C, N]``).
- Empty bins are masked with ``torch.where`` instead of boolean-index assignment;
  divisions are guarded so empty bins give zero, not NaN, gradients. No
  host<->device copies or syncs anywhere, so fwd+bwd is CUDA-graph capturable
  (tests/test_sync.py).
"""

import torch

from ace_kernels.binning import bin_stats, per_bin_error
from ace_kernels.common import finalize, prepare
from ace_kernels.config import ACEConfig
from ace_kernels.registry import register


@register("v1_clean", description="Vectorized + plain sums, no one-hot, no host sync")
def v1_clean(
    input: torch.Tensor, target: torch.Tensor, config: ACEConfig
) -> torch.Tensor:
    prep = prepare(input, target, config, keep_labels=True)
    return finalize(per_bin_error(bin_stats(prep, config)), prep, config)


_compiled = torch.compile(v1_clean)


@register(
    "v1_clean_compile",
    description="v1_clean under torch.compile (default mode)",
    slow=True,
)
def v1_clean_compile(
    input: torch.Tensor, target: torch.Tensor, config: ACEConfig
) -> torch.Tensor:
    loss: torch.Tensor = _compiled(input, target, config)
    return loss
