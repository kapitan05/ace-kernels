"""Optional Weights & Biases mirror of benchmark results.

Parquet files in ``results/`` are the source of truth; W&B adds off-machine
persistence (Kaggle sessions are ephemeral), dashboards and shareable links.
The API key is read by wandb itself from ``WANDB_API_KEY``; it is never passed
or logged here. Without ``enabled`` every call is a no-op.
"""

import argparse
import json
from pathlib import Path
from types import TracebackType
from typing import Any

import polars as pl


class Tracker:
    def __init__(
        self,
        enabled: bool,
        *,
        project: str = "ace-kernels",
        group: str,
        job_type: str,
        name: str,
        config: dict[str, Any],
        tags: list[str],
    ) -> None:
        self._run: Any = None
        if not enabled:
            return
        import wandb

        self._wandb = wandb
        self._run = wandb.init(
            project=project,
            group=group,
            job_type=job_type,
            name=name,
            config=config,
            tags=tags,
        )

    @property
    def url(self) -> str | None:
        return None if self._run is None else str(self._run.url)

    def log_table(self, key: str, df: pl.DataFrame) -> None:
        if self._run is None:
            return
        columns: list[str | int] = list(df.columns)
        table = self._wandb.Table(columns=columns, data=df.rows())
        self._run.log({key: table})

    def summary(self, values: dict[str, float | int | str | None]) -> None:
        if self._run is not None:
            self._run.summary.update(values)

    def log_files(self, name: str, kind: str, paths: list[Path]) -> None:
        if self._run is None:
            return
        artifact = self._wandb.Artifact(name=name, type=kind)
        for p in paths:
            if p.is_dir():
                artifact.add_dir(str(p), name=p.name)
            elif p.exists():
                artifact.add_file(str(p))
        self._run.log_artifact(artifact)

    def __enter__(self) -> "Tracker":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._run is not None:
            self._run.finish(exit_code=0 if exc_type is None else 1)


def mirror_run(run_dir: Path, project: str = "ace-kernels") -> str | None:
    """Upload one finished benchmark dir (meta.json + rows.parquet) to W&B."""
    meta = json.loads((run_dir / "meta.json").read_text())
    df = pl.read_parquet(run_dir / "rows.parquet")
    config = meta["config"]
    with Tracker(
        True,
        project=project,
        group=config["group"],
        job_type="bench",
        name=meta["run_id"],
        config={**config, "env": meta["env"]},
        tags=[*config["impls"], meta["env"]["device_name"]],
    ) as tracker:
        tracker.log_table("results", df)
        tracker.summary({"run_id": meta["run_id"], "rows": df.height})
        tracker.log_files(f"bench-{config['name']}", "bench-results", [run_dir])
        return tracker.url


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mirror benchmark results to W&B (key from `wandb login`)"
    )
    parser.add_argument("path", type=Path, help="a run dir, or a dir containing runs")
    args = parser.parse_args()
    runs = sorted(p.parent for p in args.path.rglob("meta.json"))
    if not runs:
        raise SystemExit(f"no runs (meta.json) under {args.path}")
    for run_dir in runs:
        print(f"{run_dir.name}: {mirror_run(run_dir)}")


if __name__ == "__main__":
    main()
