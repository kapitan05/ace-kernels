"""Optional Weights & Biases mirror of benchmark results.

Parquet files in ``results/`` are the source of truth; W&B adds off-machine
persistence (Kaggle sessions are ephemeral), dashboards and shareable links.
The API key is read by wandb itself from ``WANDB_API_KEY``; it is never passed
or logged here. Without ``enabled`` every call is a no-op.
"""

import argparse
import json
from html import escape
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

    def log_images(self, paths: list[Path], prefix: str = "plots") -> None:
        if self._run is None or not paths:
            return
        self._run.log({f"{prefix}/{p.stem}": self._wandb.Image(str(p)) for p in paths})

    def log_text(self, key: str, path: Path) -> None:
        """Show a text file (op table, saved tensors, log tail) on the run page."""
        if self._run is not None and path.exists():
            text = path.read_text(errors="replace")[-100_000:]
            self._run.log({key: self._wandb.Html(f"<pre>{escape(text)}</pre>")})

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
        tracker.log_images(sorted((run_dir / "plots").glob("*.png")))
        tracker.summary({"run_id": meta["run_id"], "rows": df.height})
        tracker.log_files(f"bench-{config['name']}", "bench-results", [run_dir])
        return tracker.url


def mirror_profiles(
    root: Path, group: str, logs: list[Path], project: str = "ace-kernels"
) -> str | None:
    """One W&B run holding every profile dir under ``root`` (+ job logs)."""
    profile_dirs = sorted(p.parent for p in root.rglob("ops.txt"))
    nsys = sorted(root.rglob("nsys"))
    with Tracker(
        True,
        project=project,
        group=group,
        job_type="profile",
        name=f"profiles-{root.name}",
        config={"profiles": [d.name for d in profile_dirs]},
        tags=["profile"],
    ) as tracker:
        for d in profile_dirs:
            tracker.log_text(f"{d.name}/ops", d / "ops.txt")
            tracker.log_text(f"{d.name}/saved_tensors", d / "saved_tensors.txt")
            tracker.log_text(f"{d.name}/dynamo_explain", d / "dynamo_explain.txt")
        for log in logs:
            tracker.log_text(f"logs/{log.stem}", log)
        tracker.log_files("profiles", "profile", [*profile_dirs, *nsys, *logs])
        return tracker.url


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mirror benchmark results to W&B (key from `wandb login`)"
    )
    parser.add_argument("path", type=Path, help="a run dir, or a dir containing runs")
    args = parser.parse_args()
    runs = sorted(p.parent for p in args.path.rglob("meta.json"))
    has_profiles = any(args.path.rglob("ops.txt"))
    if not runs and not has_profiles:
        raise SystemExit(f"no runs (meta.json) or profiles (ops.txt) under {args.path}")
    groups = set()
    for run_dir in runs:
        groups.add(json.loads((run_dir / "meta.json").read_text())["config"]["group"])
        print(f"{run_dir.name}: {mirror_run(run_dir)}")
    if has_profiles:
        group = groups.pop() if len(groups) == 1 else "profiles"
        logs = sorted(args.path.glob("*.log"))
        print(f"profiles: {mirror_profiles(args.path, group, logs)}")


if __name__ == "__main__":
    main()
