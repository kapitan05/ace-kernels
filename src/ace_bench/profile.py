"""Profile one (impl, binning, shape, mode): where the time and memory go.

    uv run python -m ace_bench.profile --impl reference --shape kits23
    nsys profile --capture-range=cudaProfilerApi --trace=cuda,nvtx -o ref \\
        python -m ace_bench.profile --impl reference --shape kits23 --nsys

Writes ``results/profile/<impl>-<binning>-<shape>-<mode>/``:
- ``trace.json``: torch.profiler trace (open in https://ui.perfetto.dev)
- ``ops.txt``: op/kernel table sorted by device time, kernel count per step
- ``saved_tensors.txt``: largest tensors autograd keeps for backward
- ``memory_snapshot.pickle`` (CUDA): open in https://pytorch.org/memory_viz
- ``dynamo_explain.txt``: graph breaks seen by torch.compile
"""

import argparse
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function, schedule

from ace_bench.config import ShapeSpec
from ace_bench.env import REPO_ROOT, collect_env
from ace_kernels import ACEConfig, ace_loss
from ace_kernels.cases import InputSpec, make_inputs
from ace_kernels.instrumentation import measure_saved_tensors

# Paper training shapes (bundle_templates/configs/data/*.yaml upstream, batch 2).
PAPER_SHAPES = {
    s.name: s
    for s in [
        ShapeSpec(name="kits23", batch=2, channels=4, spatial=(128, 128, 128)),
        ShapeSpec(name="brats21", batch=2, channels=4, spatial=(224, 224, 144)),
        ShapeSpec(name="acdc17", batch=2, channels=4, spatial=(256, 256, 16)),
        ShapeSpec(name="amos22", batch=2, channels=16, spatial=(176, 176, 64)),
        ShapeSpec(name="small", batch=2, channels=4, spatial=(32, 32, 32)),
    ]
}

ACTIVE_STEPS = 3


@contextmanager
def _range(name: str, nvtx: bool) -> Iterator[None]:
    """A torch.profiler range, plus an NVTX range for nsys when ``nvtx``."""
    with record_function(name):
        if nvtx:
            torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            if nvtx:
                torch.cuda.nvtx.range_pop()


def _step(
    x: torch.Tensor,
    y: torch.Tensor,
    config: ACEConfig,
    impl: str,
    mode: str,
    nvtx: bool,
) -> None:
    with _range("ace_fwd", nvtx):
        loss = ace_loss(x, y, config, impl=impl)
    if mode == "fwd_bwd":
        with _range("ace_bwd", nvtx):
            torch.autograd.grad(loss.sum(), x)


def _write_saved(
    out: Path, x: torch.Tensor, y: torch.Tensor, config: ACEConfig, impl: str
) -> None:
    _, stats = measure_saved_tensors(lambda: ace_loss(x, y, config, impl=impl), [x, y])
    lines = [
        f"saved tensors: {len(stats.tensors)}",
        f"unique bytes: {stats.unique_bytes / 2**20:.1f} MiB",
        f"excluded (input) bytes: {stats.excluded_bytes / 2**20:.1f} MiB",
        f"extra bytes: {stats.extra_bytes / 2**20:.1f} MiB "
        f"({stats.extra_bytes / (x.numel() * x.element_size()):.2f}x input size)",
        "",
        "largest (shape, dtype, MiB):",
        *(f"  {t.shape} {t.dtype} {t.nbytes / 2**20:.1f}" for t in stats.largest(15)),
    ]
    (out / "saved_tensors.txt").write_text("\n".join(lines) + "\n")


def _write_explain(
    out: Path, x: torch.Tensor, y: torch.Tensor, config: ACEConfig
) -> None:
    explain = torch._dynamo.explain(lambda a, b: ace_loss(a, b, config))(x, y)
    reasons = [f"- {r.reason}" for r in explain.break_reasons]
    text = (
        f"graphs: {explain.graph_count}\n"
        f"graph breaks: {explain.graph_break_count}\n"
        f"ops: {explain.op_count}\n" + "\n".join(reasons) + "\n"
    )
    (out / "dynamo_explain.txt").write_text(text)
    torch._dynamo.reset()


def profile_case(
    impl: str,
    shape: ShapeSpec,
    config: ACEConfig,
    mode: str,
    device: str,
    out_root: Path,
    nsys: bool,
) -> Path:
    out = out_root / f"{impl}-{config.binning}-{shape.name}-{mode}"
    out.mkdir(parents=True, exist_ok=True)
    x, y = make_inputs(
        InputSpec(shape.batch, shape.channels, shape.spatial), config, device
    )
    x.requires_grad_(True)
    warnings.simplefilter("ignore", UserWarning)
    for _ in range(2):  # compile / allocator warmup
        _step(x, y, config, impl, mode, nvtx=False)

    if nsys:  # nsys --capture-range=cudaProfilerApi records only this block
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        for _ in range(ACTIVE_STEPS):
            _step(x, y, config, impl, mode, nvtx=True)
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()
        return out

    activities = [ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(ProfilerActivity.CUDA)
    with profile(
        activities=activities,
        schedule=schedule(wait=1, warmup=1, active=ACTIVE_STEPS, repeat=1),
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        for _ in range(2 + ACTIVE_STEPS):
            _step(x, y, config, impl, mode, nvtx=False)
            if device == "cuda":
                torch.cuda.synchronize()
            prof.step()
    prof.export_chrome_trace(str(out / "trace.json"))
    kernels = sum(1 for e in prof.events() if e.device_type.name == "CUDA")
    sort = "self_cuda_time_total" if device == "cuda" else "self_cpu_time_total"
    table = prof.key_averages().table(sort_by=sort, row_limit=40)
    header = f"{impl} {config.binning} {shape} {mode} on {device}\n"
    if device == "cuda":
        header += f"CUDA kernels per step: {kernels / ACTIVE_STEPS:.0f}\n"
    (out / "ops.txt").write_text(header + "\n" + table + "\n")

    _write_saved(out, x, y, config, impl)
    if device == "cuda":
        torch.cuda.memory._record_memory_history(max_entries=200_000)
        _step(x, y, config, impl, mode, nvtx=False)
        torch.cuda.synchronize()
        torch.cuda.memory._dump_snapshot(str(out / "memory_snapshot.pickle"))
        torch.cuda.memory._record_memory_history(enabled=None)
    if impl == "reference":
        _write_explain(out, x.detach(), y, config)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile one implementation")
    parser.add_argument("--impl", default="reference")
    parser.add_argument("--binning", choices=["hard", "soft"], default="hard")
    parser.add_argument("--shape", choices=sorted(PAPER_SHAPES), default="small")
    parser.add_argument("--mode", choices=["fwd", "fwd_bwd"], default="fwd_bwd")
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "profile")
    parser.add_argument(
        "--nsys",
        action="store_true",
        help="only emit NVTX ranges + cudaProfilerApi capture",
    )
    args = parser.parse_args()
    config = ACEConfig(binning=args.binning, to_onehot_y=True)
    env = collect_env(args.device)
    out = profile_case(
        args.impl,
        PAPER_SHAPES[args.shape],
        config,
        args.mode,
        args.device,
        args.out,
        args.nsys,
    )
    print(f"{env.device_name}: wrote {out}")


if __name__ == "__main__":
    main()
