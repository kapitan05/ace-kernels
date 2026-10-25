"""Optional Weights & Biases mirror of benchmark results.

Parquet files in ``results/`` are the source of truth; W&B adds off-machine
persistence (Kaggle sessions are ephemeral), dashboards and shareable links.
The API key is read by wandb itself from ``WANDB_API_KEY``; it is never passed
or logged here. Without ``enabled`` every call is a no-op.
"""

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
