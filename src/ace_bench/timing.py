"""Wall-clock timing of GPU/CPU work with warmup, sync and L2 flushing.

Same method as ``triton.testing.do_bench`` (CUDA events, cache flush between
reps, quantiles), without requiring Triton, so CPU and GPU share one code path.
"""

import statistics
import time
from collections.abc import Callable

import torch
from pydantic import BaseModel

# Larger than any current L2 (H100: 50 MB, RTX PRO 6000: 128 MB).
_L2_FLUSH_BYTES = 256 * 1024 * 1024


class TimingStats(BaseModel):
    """Milliseconds per call over ``reps`` measured calls."""

    p50_ms: float
    p20_ms: float
    p80_ms: float
    mean_ms: float
    std_ms: float
    min_ms: float
    reps: int


def _quantile(sorted_ms: list[float], q: float) -> float:
    idx = min(len(sorted_ms) - 1, max(0, round(q * (len(sorted_ms) - 1))))
    return sorted_ms[idx]


def _summarize(times_ms: list[float]) -> TimingStats:
    s = sorted(times_ms)
    return TimingStats(
        p50_ms=_quantile(s, 0.5),
        p20_ms=_quantile(s, 0.2),
        p80_ms=_quantile(s, 0.8),
        mean_ms=statistics.fmean(s),
        std_ms=statistics.stdev(s) if len(s) > 1 else 0.0,
        min_ms=s[0],
        reps=len(s),
    )


def measure(
    fn: Callable[[], object],
    device: str,
    warmup: int = 5,
    reps: int = 20,
    flush_l2: bool = True,
) -> TimingStats:
    """Time ``fn()``; each rep is synchronized and (on CUDA) starts with a cold L2."""
    for _ in range(warmup):
        fn()
    if device != "cuda":
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t0) * 1e3)
        return _summarize(times)

    flush = torch.empty(_L2_FLUSH_BYTES, dtype=torch.uint8, device="cuda")
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    torch.cuda.synchronize()
    for start, end in zip(starts, ends, strict=True):
        if flush_l2:
            flush.zero_()
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    return _summarize([s.elapsed_time(e) for s, e in zip(starts, ends, strict=True)])
