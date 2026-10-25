"""Measure one (impl, binning, shape, mode) combination -> one result row."""

import gc
import traceback
import warnings
from collections.abc import Callable
from typing import Literal

import polars as pl
import torch
from pydantic import BaseModel
from torch.profiler import ProfilerActivity, profile

from ace_bench import calculator
from ace_bench.calculator import Mode
from ace_bench.config import ShapeSpec
from ace_bench.env import EnvInfo
from ace_bench.timing import TimingStats, measure
from ace_kernels import ACEConfig, ace_loss, get_impl
from ace_kernels.cases import InputSpec, make_inputs
from ace_kernels.instrumentation import measure_saved_tensors

Status = Literal["ok", "oom", "error", "unsupported", "capture_failed"]


class Row(BaseModel):
    """One benchmark result. Byte and time fields are None when not measured."""

    impl: str
    binning: str
    mode: str
    cuda_graph: bool
    shape: str
    batch: int
    channels: int
    spatial: str
    voxels: int
    status: Status
    error: str | None = None
    p50_ms: float | None = None
    p20_ms: float | None = None
    p80_ms: float | None = None
    mean_ms: float | None = None
    std_ms: float | None = None
    reps: int | None = None
    input_bytes: int
    target_bytes: int
    ideal_bytes: int
    effective_gbps: float | None = None
    pct_peak_bw: float | None = None
    bw_floor_ms: float | None = None
    launch_floor_ms: float | None = None
    kernels_per_call: int | None = None
    aten_ops_per_call: int | None = None
    saved_extra_bytes: int | None = None
    peak_extra_bytes: int | None = None


_POLARS_TYPES: dict[str, type[pl.DataType]] = {
    "str": pl.String,
    "int": pl.Int64,
    "float": pl.Float64,
    "bool": pl.Boolean,
}


def rows_to_frame(rows: list[Row]) -> pl.DataFrame:
    """DataFrame with a fixed schema (all-None columns keep their numeric type)."""
    schema: dict[str, type[pl.DataType]] = {}
    for name, field in Row.model_fields.items():
        args = getattr(field.annotation, "__args__", (field.annotation,))
        base = next(a for a in args if a is not type(None))
        schema[name] = _POLARS_TYPES.get(getattr(base, "__name__", "str"), pl.String)
    return pl.DataFrame([r.model_dump() for r in rows], schema=schema)


def _step_fn(
    impl: str, x: torch.Tensor, y: torch.Tensor, config: ACEConfig, mode: Mode
) -> Callable[[], object]:
    def fwd() -> torch.Tensor:
        return ace_loss(x, y, config, impl=impl)

    def fwd_bwd() -> torch.Tensor:
        (grad,) = torch.autograd.grad(fwd().sum(), x)
        return grad

    return fwd if mode == "fwd" else fwd_bwd


def _graphed(fn: Callable[[], object]) -> Callable[[], object]:
    """Capture ``fn`` in a CUDA graph; the returned callable replays it."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):  # warm up on a side stream, as required for capture
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph.replay


def _count_ops(fn: Callable[[], object]) -> tuple[int, int]:
    """CUDA kernels and aten op calls launched by one call (CUDA only)."""
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    events = prof.events()
    kernels = sum(1 for e in events if e.device_type.name == "CUDA")
    aten = sum(1 for e in events if e.name.startswith("aten::"))
    return kernels, aten


def _peak_extra_bytes(fn: Callable[[], object], device: str) -> int | None:
    """Peak memory above what was allocated before the call (CUDA only)."""
    if device != "cuda":
        return None
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    out = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - before
    del out
    return peak


def _apply_timing(row: Row, stats: TimingStats, env: EnvInfo) -> None:
    row.p50_ms, row.p20_ms, row.p80_ms = stats.p50_ms, stats.p20_ms, stats.p80_ms
    row.mean_ms, row.std_ms, row.reps = stats.mean_ms, stats.std_ms, stats.reps
    row.effective_gbps = calculator.effective_gbps(row.ideal_bytes, stats.p50_ms)
    if env.peak_bandwidth_gbps is not None:
        row.pct_peak_bw = 100 * row.effective_gbps / env.peak_bandwidth_gbps


def measure_case(
    impl: str,
    shape: ShapeSpec,
    loss_config: ACEConfig,
    mode: Mode,
    env: EnvInfo,
    *,
    cuda_graph: bool = False,
    warmup: int = 3,
    reps: int = 10,
    seed: int = 0,
) -> Row:
    device = env.device
    spec = InputSpec(shape.batch, shape.channels, shape.spatial, seed=seed)
    x, y = make_inputs(spec, loss_config, device)
    x.requires_grad_(True)  # as in training: fwd also pays for saving tensors
    input_bytes = x.numel() * x.element_size()
    target_bytes = y.numel() * y.element_size()
    ideal = calculator.ideal_bytes(input_bytes, target_bytes, mode)
    row = Row(
        impl=impl,
        binning=loss_config.binning,
        mode=mode,
        cuda_graph=cuda_graph,
        shape=shape.name,
        batch=shape.batch,
        channels=shape.channels,
        spatial="x".join(map(str, shape.spatial)),
        voxels=shape.voxels,
        status="ok",
        input_bytes=input_bytes,
        target_bytes=target_bytes,
        ideal_bytes=ideal,
    )
    if env.peak_bandwidth_gbps is not None:
        row.bw_floor_ms = calculator.bandwidth_floor_ms(ideal, env.peak_bandwidth_gbps)
    if not get_impl(impl).supports(loss_config.binning, device):
        row.status = "unsupported"
        return row

    fn = _step_fn(impl, x, y, loss_config, mode)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            fn()  # first call: compilation / allocator warmup, untimed
            if cuda_graph:
                try:
                    fn = _graphed(fn)
                except Exception as exc:
                    row.status = "capture_failed"
                    row.error = f"{type(exc).__name__}: {exc}"[:500]
                    return row
            else:
                x_saved = x.detach().requires_grad_(True)
                _, saved = measure_saved_tensors(
                    lambda: ace_loss(x_saved, y, loss_config, impl=impl),
                    exclude=[x_saved, y],
                )
                row.saved_extra_bytes = saved.extra_bytes
            if device == "cuda":
                row.kernels_per_call, row.aten_ops_per_call = _count_ops(fn)
                row.launch_floor_ms = calculator.launch_floor_ms(row.kernels_per_call)
            row.peak_extra_bytes = _peak_extra_bytes(fn, device)
            _apply_timing(row, measure(fn, device, warmup=warmup, reps=reps), env)
    except torch.OutOfMemoryError as exc:
        row.status, row.error = "oom", str(exc).splitlines()[0][:500]
    except Exception as exc:
        row.status = "error"
        row.error = "".join(traceback.format_exception_only(exc)).strip()[:500]
    return row


def release_memory(device: str) -> None:
    """Free cached blocks between cases so one case's peak can't skew the next."""
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
