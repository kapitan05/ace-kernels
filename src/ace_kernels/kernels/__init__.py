"""Triton kernels (Linux + CUDA only; Triton has no macOS builds)."""

try:
    import triton  # noqa: F401

    HAS_TRITON = True
except ImportError:  # macOS / CPU-only installs
    HAS_TRITON = False
