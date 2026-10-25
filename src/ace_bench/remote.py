"""Run a job script on a Kaggle GPU from the command line and fetch its results.

    uv run python -m ace_bench.remote submit kaggle/run_sprint2.sh  # push+wait+fetch
    uv run python -m ace_bench.remote status kaggle/run_sprint2.sh
    uv run python -m ace_bench.remote fetch kaggle/run_sprint2.sh

How it works (no secrets on Kaggle):
1. ``git archive HEAD`` -> private Kaggle dataset ``<user>/ace-kernels-src``
   (new dataset version per submit). Only committed code runs remotely.
2. ``kaggle/entry.py`` rendered for the job -> script kernel
   ``<user>/ace-kernels-<job>`` with GPU (T4) + internet, dataset attached.
3. Poll the kernel status, then download ``results.tgz`` + log into
   ``results/kaggle/<sha>-<job>/`` and unpack it.

Kaggle credentials come from the Kaggle CLI's own config (``~/.kaggle``).
"""

import argparse
import json
import re
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

from ace_bench.env import REPO_ROOT

DATASET_SLUG = "ace-kernels-src"
ACCELERATOR = "NvidiaTeslaT4"
DONE = {"complete", "error", "cancel_requested", "cancel_acknowledged"}


def _kaggle(*args: str, check: bool = True) -> str:
    out = subprocess.run(
        ["kaggle", *args], capture_output=True, text=True, cwd=REPO_ROOT
    )
    if check and out.returncode != 0:
        raise RuntimeError(f"kaggle {' '.join(args)} failed:\n{out.stdout}{out.stderr}")
    return out.stdout + out.stderr


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True, cwd=REPO_ROOT
    ).stdout.strip()


def kaggle_user() -> str:
    match = re.search(r"username:\s*(\S+)", _kaggle("config", "view"))
    if not match:
        raise RuntimeError("Kaggle username not configured (kaggle config view)")
    return match.group(1)


def job_name(script: Path) -> str:
    return script.stem.replace("_", "-")


def kernel_id(user: str, script: Path) -> str:
    return f"{user}/ace-kernels-{job_name(script)}"


def upload_snapshot(user: str) -> str:
    """Upload the committed tree as a new version of the private source dataset."""
    if _git("status", "--porcelain", "--untracked-files=no"):
        sys.exit("uncommitted changes: commit first (only committed code runs)")
    sha = _git("rev-parse", "HEAD")
    dataset_id = f"{user}/{DATASET_SLUG}"
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        commit = folder / "COMMIT"
        commit.write_text(f"{sha}\n")
        tar_path = folder / "src.tar"
        _git("archive", "--format=tar", "-o", str(tar_path), "HEAD")
        with tarfile.open(tar_path, "a") as tar:
            tar.add(commit, arcname="COMMIT")
        commit.unlink()
        meta = {
            "title": DATASET_SLUG,
            "id": dataset_id,
            "licenses": [{"name": "other"}],
        }
        (folder / "dataset-metadata.json").write_text(json.dumps(meta))
        exists = "ready" in _kaggle("datasets", "status", dataset_id, check=False)
        if exists:
            _kaggle("datasets", "version", "-p", tmp, "-m", sha[:12], "-q")
        else:
            _kaggle("datasets", "create", "-p", tmp, "-q")  # private by default
    print(f"uploaded {sha[:12]} -> dataset {dataset_id}, waiting until ready")
    time.sleep(30)  # right after upload, status can still report the old version
    for _ in range(60):
        if "ready" in _kaggle("datasets", "status", dataset_id, check=False):
            return sha
        time.sleep(10)
    sys.exit("dataset not ready after 10 min")


def push_kernel(user: str, script: Path) -> str:
    kid = kernel_id(user, script)
    entry = (REPO_ROOT / "kaggle" / "entry.py").read_text()
    entry = entry.replace("__JOB_SCRIPT__", script.as_posix())
    meta = {
        "id": kid,
        "title": kid.split("/")[1],
        "code_file": "entry.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": ACCELERATOR,
        "dataset_sources": [f"{user}/{DATASET_SLUG}"],
    }
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "entry.py").write_text(entry)
        (Path(tmp) / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
        print(_kaggle("kernels", "push", "-p", tmp, "--accelerator", ACCELERATOR))
    return kid


def status(kid: str) -> str:
    # e.g. 'has status "KernelWorkerStatus.RUNNING"' (older CLIs: 'status "running"')
    out = _kaggle("kernels", "status", kid)
    match = re.search(r'status "?(?:KernelWorkerStatus\.)?([A-Za-z_]+)', out)
    return match.group(1).lower() if match else "unknown"


def wait(kid: str, poll_s: int = 60) -> str:
    start = time.time()
    while (state := status(kid)) not in DONE:
        print(f"[{(time.time() - start) / 60:5.1f} min] {kid}: {state}", flush=True)
        time.sleep(poll_s)
    print(f"{kid}: {state}")
    return state


def fetch(kid: str, script: Path) -> Path:
    tmp = REPO_ROOT / "results" / "kaggle" / "_download"
    tmp.mkdir(parents=True, exist_ok=True)
    _kaggle("kernels", "output", kid, "-p", str(tmp), "-o")
    commit = tmp / "COMMIT"
    sha = commit.read_text().strip()[:12] if commit.exists() else "unknown"
    out = REPO_ROOT / "results" / "kaggle" / f"{sha}-{job_name(script)}"
    if out.exists():
        out = out.with_name(f"{out.name}-{time.strftime('%Y%m%dT%H%M%S')}")
    tmp.rename(out)
    archive = out / "results.tgz"
    if archive.exists():
        with tarfile.open(archive) as tar:
            tar.extractall(out, filter="data")
    print(f"fetched -> {out}")
    for log in out.glob("*.log"):
        print(f"log: {log}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run job scripts on Kaggle GPUs")
    parser.add_argument("action", choices=["submit", "status", "fetch"])
    parser.add_argument(
        "script", type=Path, help="job script, e.g. kaggle/run_sprint2.sh"
    )
    parser.add_argument("--no-wait", action="store_true", help="submit and return")
    args = parser.parse_args()
    script = args.script
    if not (REPO_ROOT / script).is_file():
        sys.exit(f"no such job script: {script}")
    user = kaggle_user()
    kid = kernel_id(user, script)
    if args.action == "status":
        print(f"{kid}: {status(kid)}")
    elif args.action == "fetch":
        fetch(kid, script)
    else:
        sha = upload_snapshot(user)
        push_kernel(user, script)
        if not args.no_wait:
            wait(kid)
            out = fetch(kid, script)
            ran = (
                (out / "COMMIT").read_text().strip()
                if (out / "COMMIT").exists()
                else ""
            )
            if ran != sha:
                print(
                    f"WARNING: Kaggle ran commit {ran or '?'} but {sha} was submitted "
                    "(dataset version not ready yet?) - resubmit"
                )


if __name__ == "__main__":
    main()
