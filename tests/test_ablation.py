"""Cross-sprint ablation aggregation (ace_bench.ablation)."""

from pathlib import Path

import polars as pl

from ace_bench.ablation import build
from ace_bench.config import ShapeSpec
from ace_bench.env import collect_env
from ace_bench.measure import measure_case, rows_to_frame
from ace_kernels import ACEConfig

KITS = ShapeSpec(name="kits23", batch=1, channels=3, spatial=(8, 8, 8))


def _write_run(root: Path, commit: str, run: str, impls: list[str]) -> None:
    env = collect_env("cpu")
    rows = [
        measure_case(
            i,
            KITS,
            ACEConfig(binning=b, to_onehot_y=True),
            "fwd_bwd",
            env,
            warmup=0,
            reps=1,
        )
        for i in impls
        for b in ("hard", "soft")
    ]
    out = root / f"{commit}-run-x" / "results" / run
    out.mkdir(parents=True)
    rows_to_frame(rows).write_parquet(out / "rows.parquet")


def test_ablation_skips_excluded_and_accuracy_only_runs(tmp_path: Path) -> None:
    root = tmp_path / "kaggle"
    _write_run(
        root, "aaaaaaaaaaaa", "20260101T000000Z-paper", ["reference", "v1_clean"]
    )
    _write_run(root, "afa97a0e3643", "20260102T000000Z-paper", ["v2_recompute"])
    _write_run(root, "bbbbbbbbbbbb", "20260103T000000Z-verify-truth", ["v1_vectorized"])
    text = build(root, tmp_path / "out")
    assert "v1_clean" in text
    assert "v2_recompute" not in text  # excluded run (harness leak)
    assert "| S3 | v1_vectorized" not in text  # accuracy-only run: no timings
    best = pl.read_parquet(tmp_path / "out" / "ablation_rows.parquet")
    assert set(best["impl"]) == {"reference", "v1_clean"}
    assert (tmp_path / "out" / "ladder-hard.png").exists()
