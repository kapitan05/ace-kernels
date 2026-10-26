"""Importing this package registers every implementation."""

from ace_kernels.impls import (
    baselines,
    reference,
    v1_clean,
    v1_vectorized,
    v2_autograd,
    v3_triton,
    v5_triton,
)

__all__ = [
    "baselines",
    "reference",
    "v1_clean",
    "v1_vectorized",
    "v2_autograd",
    "v3_triton",
    "v5_triton",
]
