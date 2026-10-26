"""Run a benchmark experiment and write results.

    uv run python -m ace_bench.run bench/configs/quick_cpu.yaml
    uv run python -m ace_bench.run bench/configs/baseline.yaml --wandb

Writes ``results/<timestamp>-<name>/``: ``rows.parquet`` (one row per
impl x binning x shape x mode [x cuda_graph]), ``meta.json`` (config + env) and
``summary.md`` (tables).
"""

import argparse
import itertools
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
import torch
from pydantic import BaseModel

from ace_bench.config import BenchConfig, ShapeSpec
from ace_bench.env import REPO_ROOT, EnvInfo, collect_env
from ace_bench.measure import (
    Row,
    compute_reference,
    measure_case,
    release_memory,
    rows_to_frame,
)
from ace_bench.report import write_summary
from ace_bench.tracking import mirror_run
from ace_kernels.config import Binning


def _fmt(row: Row) -> str:
    if row.status != "ok":
        return f"{row.status}: {row.error or ''}"[:100]
    bw = f"{row.pct_peak_bw:5.1f}% peak" if row.pct_peak_bw is not None else ""
    mem = (
        f"peak+{row.peak_extra_bytes / 2**20:8.1f} MiB"
        if row.peak_extra_bytes is not None
        else ""
    )
    err = (
        f"  err loss {row.loss_rel_err:.0e} grad {row.grad_rel_err:.0e}"
        if row.loss_rel_err is not None and row.grad_rel_err is not None
        else ""
    )
    return f"{row.p50_ms:9.3f} ms  {bw}  {mem}  kernels={row.kernels_per_call}{err}"


class Job(BaseModel):
    """One (shape, binning, impl) group: all its modes x CUDA-graph rows."""

    config: BenchConfig
    env: EnvInfo
    shape: ShapeSpec
    binning: Binning
    impl: str


def run_group(job: Job) -> list[Row]:
    config = job.config
    loss_config = config.loss.model_copy(update={"binning": job.binning})
    graph_options = [False, True] if config.cuda_graph else [False]
    reference = (
        compute_reference(
            job.shape, loss_config, config.device, config.seed, config.dtype
        )
        if config.verify
        else None
    )
    release_memory(config.device)
    rows = []
    for mode, graphed in itertools.product(config.modes, graph_options):
        start = time.perf_counter()
        row = measure_case(
            job.impl,
            job.shape,
            loss_config,
            mode,
            job.env,
            cuda_graph=graphed,
            warmup=config.warmup,
            reps=config.reps,
            seed=config.seed,
            reference=reference,
            dtype=config.dtype,
        )
        release_memory(config.device)
        if config.device == "cuda":
            row.allocated_after_bytes = torch.cuda.memory_allocated()
        rows.append(row)
        label = f"{job.shape.name:10s} {job.binning:4s} {job.impl:22s} "
        label += f"{mode:7s}{' graph' if graphed else '':6s}"
        took = time.perf_counter() - start
        print(f"{label} {_fmt(row)}  ({took:.1f}s)", file=sys.stderr, flush=True)
    return rows


_ROWS_MARKER = "ACE_BENCH_ROWS:"


def run_isolated(job: Job) -> list[Row]:
    """Run ``job`` in a fresh Python process; a crash becomes error rows."""
    proc = subprocess.run(
        [sys.executable, "-m", "ace_bench.run", "--worker"],
        input=job.model_dump_json(),
        stdout=subprocess.PIPE,
        text=True,
        check=False,
    )
    for line in proc.stdout.splitlines():
        if line.startswith(_ROWS_MARKER):
            payload = json.loads(line.removeprefix(_ROWS_MARKER))
            return [Row.model_validate(r) for r in payload]
    error = f"worker exited with code {proc.returncode}, no results"
    print(f"{job.shape.name} {job.binning} {job.impl}: {error}", file=sys.stderr)
    template = run_group_template(job)
    return [r.model_copy(update={"status": "error", "error": error}) for r in template]


def run_group_template(job: Job) -> list[Row]:
    """Placeholder rows (no measurement) for a group whose worker crashed."""
    graph_options = [False, True] if job.config.cuda_graph else [False]
    s = job.shape
    return [
        Row(
            impl=job.impl,
            binning=job.binning,
            mode=mode,
            cuda_graph=graphed,
            shape=s.name,
            batch=s.batch,
            channels=s.channels,
            spatial="x".join(map(str, s.spatial)),
            voxels=s.voxels,
            status="error",
            input_bytes=0,
            target_bytes=0,
            ideal_bytes=0,
        )
        for mode, graphed in itertools.product(job.config.modes, graph_options)
    ]


def worker() -> None:
    """Entry for ``--worker``: read a Job from stdin, print rows to stdout."""
    job = Job.model_validate_json(sys.stdin.read())
    rows = run_group(job)
    payload = json.dumps([r.model_dump() for r in rows])
    print(f"{_ROWS_MARKER}{payload}", flush=True)


def run(config: BenchConfig, out_root: Path, use_wandb: bool) -> Path:
    env = collect_env(config.device)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = out_root / f"{stamp}-{config.name}"
    out.mkdir(parents=True)
    meta: dict[str, Any] = {
        "config": config.model_dump(),
        "env": env.model_dump(),
        "run_id": out.name,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"{env.device_name} | torch {env.torch_version} | git {env.git_sha}")

    rows: list[Row] = []
    for shape, binning, impl in itertools.product(
        config.shapes, config.binnings, config.impls
    ):
        job = Job(config=config, env=env, shape=shape, binning=binning, impl=impl)
        rows += run_isolated(job) if config.isolate else run_group(job)

    df = rows_to_frame(rows).with_columns(pl.lit(out.name).alias("run_id"))
    df.write_parquet(out / "rows.parquet")
    summary = write_summary(df, meta, out)
    print(f"\nwrote {out}\n{summary}")
    if use_wandb:
        print(f"W&B: {mirror_run(out)}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("config", type=Path, nargs="?")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--wandb", action="store_true", help="mirror to W&B")
    parser.add_argument("--device", choices=["cpu", "cuda"], help="override config")
    parser.add_argument(
        "--no-isolate", action="store_true", help="all groups in this process"
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker()
        return
    if args.config is None:
        parser.error("config is required")
    config = BenchConfig.from_yaml(args.config)
    if args.no_isolate:
        config = config.model_copy(update={"isolate": False})
    if args.device:
        config = config.model_copy(update={"device": args.device})
    run(config, args.out, args.wandb)


if __name__ == "__main__":
    main()
