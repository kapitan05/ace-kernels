"""Measure one (impl, binning, shape, mode) combination -> one result row."""

import gc
import traceback
import warnings
from collections.abc import Callable
from dataclasses import dataclass
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
    dtype: str = "float32"
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
    # CUDA memory still allocated after the row and its cleanup: should stay at
    # the baseline; growth across rows means a leak (Sprint 3 run 4 lost 38 rows
    # to ~10 GiB leaked from earlier rows).
    allocated_after_bytes: int | None = None
    # Output check at benchmark size, against `checked_against` on the same input:
    # |L - L_ref| / |L_ref| and max|g - g_ref| / max|g_ref| (untimed).
    checked_against: str | None = None
    loss_rel_err: float | None = None
    grad_rel_err: float | None = None
    # The same errors against a float64 truth (v2_recompute in float64, formula
    # validated by gradcheck), and the reference's own errors against it: tells
    # "less accurate" apart from "differently rounded" (both fp32).
    loss_err_truth: float | None = None
    grad_err_truth: float | None = None
    ref_loss_err_truth: float | None = None
    ref_grad_err_truth: float | None = None
    # Relative L2 gradient errors ||g - g_ref|| / ||g_ref||. The max-norm errors
    # above are dominated by ill-conditioned voxels (|x| kink: bins whose mean_p -
    # mean_y is ~0 flip gradient sign under any reordering; hard-bin edge flips):
    # measured up to 0.54 even for the reference vs float64 truth.
    grad_l2_err: float | None = None
    grad_l2_err_truth: float | None = None
    ref_grad_l2_err_truth: float | None = None


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


# Above this relative error a benchmark row is flagged: fp32 summation-order noise
# is <= 1e-5 (tests); real bugs were >= 1e-2 (Sprint 1 mutants).
VERIFY_TOLERANCE = 1e-4
# Validated stand-in when the reference itself runs out of memory (e.g. soft 288^3).
FALLBACK_REFERENCE = "v2_recompute"


@dataclass
class Reference:
    """Loss and input gradient of a trusted implementation, kept on the CPU.

    ``truth_*``: float64 result on the same input (None if it did not fit);
    ``ref_*_err_truth``: the reference's own relative errors against it.
    """

    impl: str
    loss: torch.Tensor
    grad: torch.Tensor
    truth_loss: torch.Tensor | None = None
    truth_grad: torch.Tensor | None = None
    ref_loss_err_truth: float | None = None
    ref_grad_err_truth: float | None = None
    ref_grad_l2_err_truth: float | None = None


TRUTH_IMPL = "v2_recompute"  # evaluated in float64


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    """max |a - b| / max |b| (b: the trusted value)."""
    diff = (a.double() - b.double()).abs().max()
    return float(diff / b.double().abs().max().clamp_min(1e-12))


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    """||a - b||_2 / ||b||_2 (b: the trusted value)."""
    diff = torch.linalg.vector_norm(a.double() - b.double())
    return float(diff / torch.linalg.vector_norm(b.double()).clamp_min(1e-30))


def _loss_and_grad(
    impl: str, x: torch.Tensor, y: torch.Tensor, config: ACEConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    leaf = x.detach().requires_grad_(True)
    loss = ace_loss(leaf, y, config, impl=impl)
    (grad,) = torch.autograd.grad(loss.sum(), leaf)
    return loss.detach().float().cpu(), grad.float().cpu()


def compute_reference(
    shape: ShapeSpec,
    loss_config: ACEConfig,
    device: str,
    seed: int = 0,
    dtype: str = "float32",
) -> Reference | None:
    """The reference's result on the benchmark input (fallback: v2_recompute)."""
    spec = InputSpec(
        shape.batch,
        shape.channels,
        shape.spatial,
        seed=seed,
        logit_scale=shape.logit_scale,
        target_bias=shape.target_bias,
    )
    x, y = make_inputs(spec, loss_config, device)
    x = x.to(getattr(torch, dtype))
    ref: Reference | None = None
    for impl in ("reference", FALLBACK_REFERENCE):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                loss, grad = _loss_and_grad(impl, x, y, loss_config)
            ref = Reference(impl, loss, grad)
            break
        except torch.OutOfMemoryError:
            release_memory(device)
    if ref is None:
        return None
    try:  # float64 truth on the same (already rounded) input values
        leaf = x.detach().double().requires_grad_(True)
        loss64 = ace_loss(leaf, y, loss_config, impl=TRUTH_IMPL)
        (grad64,) = torch.autograd.grad(loss64.sum(), leaf)
        ref.truth_loss, ref.truth_grad = loss64.detach().cpu(), grad64.cpu()
        ref.ref_loss_err_truth = _rel(ref.loss, ref.truth_loss)
        ref.ref_grad_err_truth = _rel(ref.grad, ref.truth_grad)
        ref.ref_grad_l2_err_truth = _rel_l2(ref.grad, ref.truth_grad)
    except torch.OutOfMemoryError:
        pass  # rows then carry no truth comparison
    release_memory(device)
    return ref


def _check_output(
    row: "Row",
    impl: str,
    x: torch.Tensor,
    y: torch.Tensor,
    config: ACEConfig,
    ref: Reference,
) -> None:
    """Fill the row's error columns (vs the reference, and vs float64 truth)."""
    row.checked_against = ref.impl
    row.ref_loss_err_truth = ref.ref_loss_err_truth
    row.ref_grad_err_truth = ref.ref_grad_err_truth
    row.ref_grad_l2_err_truth = ref.ref_grad_l2_err_truth
    if impl == ref.impl:
        loss, grad = ref.loss, ref.grad
    else:
        loss, grad = _loss_and_grad(impl, x, y, config)
    row.loss_rel_err, row.grad_rel_err = _rel(loss, ref.loss), _rel(grad, ref.grad)
    row.grad_l2_err = _rel_l2(grad, ref.grad)
    if ref.truth_loss is not None and ref.truth_grad is not None:
        row.loss_err_truth = _rel(loss, ref.truth_loss)
        row.grad_err_truth = _rel(grad, ref.truth_grad)
        row.grad_l2_err_truth = _rel_l2(grad, ref.truth_grad)


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
    reference: Reference | None = None,
    dtype: str = "float32",
) -> Row:
    device = env.device
    spec = InputSpec(
        shape.batch,
        shape.channels,
        shape.spatial,
        seed=seed,
        logit_scale=shape.logit_scale,
        target_bias=shape.target_bias,
    )
    x, y = make_inputs(spec, loss_config, device)
    x = x.to(getattr(torch, dtype))
    x.requires_grad_(True)  # as in training: fwd also pays for saving tensors
    input_bytes = x.numel() * x.element_size()
    target_bytes = y.numel() * y.element_size()
    ideal = calculator.ideal_bytes(input_bytes, target_bytes, mode)
    row = Row(
        impl=impl,
        binning=loss_config.binning,
        mode=mode,
        cuda_graph=cuda_graph,
        dtype=dtype,
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
            if not cuda_graph:  # replaying a graph allocates nothing: peak meaningless
                row.peak_extra_bytes = _peak_extra_bytes(fn, device)
            _apply_timing(row, measure(fn, device, warmup=warmup, reps=reps), env)
            if reference is not None and not cuda_graph:  # untimed output check
                del fn  # free the timed closure's buffers first
                _check_output(row, impl, x, y, loss_config, reference)
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
