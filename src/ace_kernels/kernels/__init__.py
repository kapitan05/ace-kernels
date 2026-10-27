"""Triton kernels (Linux + CUDA only; Triton has no macOS builds)."""

try:
    import triton  # noqa: F401

    HAS_TRITON = True
except ImportError:  # macOS / CPU-only installs
    HAS_TRITON = False

# Implementations registered only when Triton is available.
TRITON_IMPLS = frozenset(
    {
        "v3_triton",
        "v4_triton",
        "v5_triton",
        "v6_index",
        "v6_dot",
        "v6_triton",
        "v6_precise",
    }
)
