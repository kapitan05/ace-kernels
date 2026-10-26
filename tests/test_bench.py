"""The measurement pipeline itself: calculator, timing, rows, report, configs."""

import json
from pathlib import Path

import polars as pl
import pytest

from ace_bench import calculator
from ace_bench.config import BenchConfig, ShapeSpec
from ace_bench.env import REPO_ROOT, collect_env, peak_bandwidth_gbps
from ace_bench.measure import Row, measure_case, rows_to_frame
from ace_bench.report import write_summary
from ace_bench.run import Job, run_isolated
from ace_bench.timing import measure
from ace_kernels import ACEConfig, get_impl
from ace_kernels.kernels import HAS_TRITON, TRITON_IMPLS

CONFIG_DIR = REPO_ROOT / "bench" / "configs"
TINY = ShapeSpec(name="tiny", batch=1, channels=3, spatial=(8, 8, 8))


@pytest.mark.parametrize("path", sorted(CONFIG_DIR.glob("*.yaml")), ids=str)
def test_bench_configs_are_valid(path: Path) -> None:
    config = BenchConfig.from_yaml(path)
    for impl in config.impls:
        if impl in TRITON_IMPLS and not HAS_TRITON:
            continue  # Linux+CUDA-only impl, absent on this machine
        get_impl(impl)  # raises on typos


def test_ideal_bytes() -> None:
    assert calculator.ideal_bytes(100, 10, "fwd") == 110
    assert calculator.ideal_bytes(100, 10, "fwd_bwd") == 2 * 110 + 100


def test_bandwidth_and_launch_floors() -> None:
    assert calculator.bandwidth_floor_ms(320 * 10**9, 320.0) == pytest.approx(1e3)
    assert calculator.launch_floor_ms(200) == pytest.approx(1.0)
    assert calculator.effective_gbps(10**9, 1e3) == pytest.approx(1.0)


def test_peak_bandwidth_lookup() -> None:
    assert peak_bandwidth_gbps("Tesla T4") == 320.0
    assert peak_bandwidth_gbps("NVIDIA RTX PRO 6000 Blackwell") == 1792.0
    assert peak_bandwidth_gbps("NVIDIA H100 PCIe") == 2000.0
    assert peak_bandwidth_gbps("NVIDIA H100 80GB HBM3") == 3350.0
    assert peak_bandwidth_gbps("unknown") is None


def test_timing_cpu() -> None:
    calls = []
    stats = measure(lambda: calls.append(1), "cpu", warmup=2, reps=5)
    assert len(calls) == 7
    assert stats.reps == 5
    assert stats.p20_ms <= stats.p50_ms <= stats.p80_ms


@pytest.mark.parametrize("mode", ["fwd", "fwd_bwd"])
def test_measure_case_cpu(mode: str) -> None:
    env = collect_env("cpu")
    config = ACEConfig(binning="soft", to_onehot_y=True)
    row = measure_case("reference", TINY, config, mode, env, warmup=1, reps=2)  # type: ignore[arg-type]
    assert row.status == "ok", row.error
    assert row.p50_ms is not None and row.p50_ms > 0
    assert row.saved_extra_bytes is not None and row.saved_extra_bytes > 0
    assert row.input_bytes == 1 * 3 * 512 * 4


def test_checkpoint_saves_no_extra_bytes() -> None:
    env = collect_env("cpu")
    config = ACEConfig(binning="soft", to_onehot_y=True)
    row = measure_case("reference_checkpoint", TINY, config, "fwd", env, reps=1)
    assert row.status == "ok", row.error
    assert row.saved_extra_bytes == 0


def test_rows_frame_keeps_types_for_all_none_columns() -> None:
    row = Row(
        impl="x",
        binning="hard",
        mode="fwd",
        cuda_graph=False,
        shape="s",
        batch=1,
        channels=1,
        spatial="1",
        voxels=1,
        status="unsupported",
        input_bytes=4,
        target_bytes=4,
        ideal_bytes=8,
    )
    df = rows_to_frame([row])
    assert df.schema["p50_ms"] == pl.Float64
    assert df.schema["kernels_per_call"] == pl.Int64
    assert df.schema["status"] == pl.String


def test_report_writes_tables_and_plots(tmp_path: Path) -> None:
    env = collect_env("cpu")
    rows = [
        measure_case(
            impl,
            TINY,
            ACEConfig(binning=b, to_onehot_y=True),
            "fwd_bwd",
            env,
            warmup=0,
            reps=1,
        )
        for impl in ("reference", "reference_checkpoint")
        for b in ("hard", "soft")
    ]
    df = rows_to_frame(rows).with_columns(pl.lit("run").alias("run_id"))
    meta = {
        "run_id": "run",
        "env": env.model_dump(),
        "config": {
            "name": "t",
            "hypothesis": "h",
            "impls": ["reference", "reference_checkpoint"],
        },
    }
    text = write_summary(df, json.loads(json.dumps(meta)), tmp_path)
    assert "## soft binning, fwd_bwd" in text
    assert (tmp_path / "summary.tex").read_text().count(r"\begin{table}") == 2
    assert sorted(p.name for p in (tmp_path / "plots").iterdir()) == [
        "hard-fwd_bwd.png",
        "soft-fwd_bwd.png",
    ]


def _job(impl: str) -> Job:
    config = BenchConfig(
        name="t",
        group="g",
        hypothesis="h",
        device="cpu",
        impls=[impl],
        modes=["fwd"],
        shapes=[TINY],
        warmup=0,
        reps=1,
    )
    return Job(
        config=config, env=collect_env("cpu"), shape=TINY, binning="hard", impl=impl
    )


def test_isolated_group_returns_rows() -> None:
    rows = run_isolated(_job("reference"))
    assert [r.status for r in rows] == ["ok"]
    assert rows[0].p50_ms is not None


def test_crashed_worker_becomes_error_rows() -> None:
    rows = run_isolated(_job("does_not_exist"))  # KeyError in the worker
    assert [r.status for r in rows] == ["error"]
    assert "worker exited" in (rows[0].error or "")
