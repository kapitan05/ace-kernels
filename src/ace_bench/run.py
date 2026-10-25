"""Run a benchmark experiment and write results.

    uv run python -m ace_bench.run bench/configs/quick_cpu.yaml
    uv run python -m ace_bench.run bench/configs/baseline.yaml --wandb

Writes ``results/<timestamp>-<name>/``: ``rows.parquet`` (one row per
impl x binning x shape x mode [x cuda_graph]), ``meta.json`` (config + env) and
``summary.md`` (tables).
"""

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from ace_bench.config import BenchConfig
from ace_bench.env import REPO_ROOT, collect_env
from ace_bench.measure import Row, measure_case, release_memory, rows_to_frame
from ace_bench.report import write_summary
from ace_bench.tracking import Tracker


def _fmt(row: Row) -> str:
    if row.status != "ok":
        return f"{row.status}: {row.error or ''}"[:100]
    bw = f"{row.pct_peak_bw:5.1f}% peak" if row.pct_peak_bw is not None else ""
    mem = (
        f"peak+{row.peak_extra_bytes / 2**20:8.1f} MiB"
        if row.peak_extra_bytes is not None
        else ""
    )
    return f"{row.p50_ms:9.3f} ms  {bw}  {mem}  kernels={row.kernels_per_call}"


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

    graph_options = [False, True] if config.cuda_graph else [False]
    rows: list[Row] = []
    tracker = Tracker(
        use_wandb,
        group=config.group,
        job_type="bench",
        name=out.name,
        config={**meta["config"], "env": meta["env"], "hypothesis": config.hypothesis},
        tags=config.impls,
    )
    with tracker:
        for shape in config.shapes:
            for binning in config.binnings:
                loss_config = config.loss.model_copy(update={"binning": binning})
                for impl in config.impls:
                    for mode in config.modes:
                        for graphed in graph_options:
                            start = time.perf_counter()
                            row = measure_case(
                                impl,
                                shape,
                                loss_config,
                                mode,
                                env,
                                cuda_graph=graphed,
                                warmup=config.warmup,
                                reps=config.reps,
                                seed=config.seed,
                            )
                            release_memory(config.device)
                            rows.append(row)
                            label = f"{shape.name:10s} {binning:4s} {impl:22s} "
                            label += f"{mode:7s}{' graph' if graphed else '':6s}"
                            took = time.perf_counter() - start
                            print(f"{label} {_fmt(row)}  ({took:.1f}s)", flush=True)
        df = rows_to_frame(rows)
        df = df.with_columns(pl.lit(out.name).alias("run_id"))
        df.write_parquet(out / "rows.parquet")
        summary = write_summary(df, meta, out)
        tracker.log_table("results", df)
        tracker.summary({"run_id": out.name, "rows": len(rows)})
        tracker.log_files(f"bench-{config.name}", "bench-results", [out])
    print(f"\nwrote {out}\n{summary}")
    if tracker.url:
        print(f"W&B: {tracker.url}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("config", type=Path)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--wandb", action="store_true", help="mirror to W&B")
    parser.add_argument("--device", choices=["cpu", "cuda"], help="override config")
    args = parser.parse_args()
    config = BenchConfig.from_yaml(args.config)
    if args.device:
        config = config.model_copy(update={"device": args.device})
    run(config, args.out, args.wandb)


if __name__ == "__main__":
    main()
