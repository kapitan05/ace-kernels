"""Environment metadata attached to every result, and device peak bandwidth."""

import platform
import subprocess
from pathlib import Path

import torch
from pydantic import BaseModel

# Peak DRAM bandwidth (GB/s, 1e9 bytes/s) from vendor spec sheets. Matched by
# substring of torch.cuda.get_device_name(); first match wins, so the more
# specific names come first.
PEAK_BANDWIDTH_GBPS: list[tuple[str, float]] = [
    ("RTX PRO 6000", 1792.0),
    ("RTX 5090", 1792.0),
    ("RTX 4090", 1008.0),
    ("H100 PCIe", 2000.0),
    ("H100", 3350.0),
    ("A100-SXM4-80GB", 2039.0),
    ("A100", 1555.0),
    ("L4", 300.0),
    ("T4", 320.0),
    ("P100", 732.0),
]

REPO_ROOT = Path(__file__).resolve().parents[2]


class EnvInfo(BaseModel):
    device: str
    device_name: str
    peak_bandwidth_gbps: float | None
    sm_count: int | None
    torch_version: str
    cuda_version: str | None
    triton_version: str | None
    driver_version: str | None
    python_version: str
    platform: str
    git_sha: str | None
    git_dirty: bool | None


def peak_bandwidth_gbps(device_name: str) -> float | None:
    for key, gbps in PEAK_BANDWIDTH_GBPS:
        if key in device_name:
            return gbps
    return None


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, check=True, cwd=REPO_ROOT, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def _triton_version() -> str | None:
    try:
        import triton
    except ImportError:
        return None
    return str(triton.__version__)


def collect_env(device: str) -> EnvInfo:
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        name, sm_count = props.name, props.multi_processor_count
        driver = _run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
        )
    else:
        name, sm_count, driver = platform.processor() or platform.machine(), None, None
    status = _run(["git", "status", "--porcelain", "--untracked-files=no"])
    sha = _run(["git", "rev-parse", "HEAD"])
    commit_file = REPO_ROOT / "COMMIT"  # written into remote (Kaggle) snapshots
    if sha is None and commit_file.exists():
        sha, status = commit_file.read_text().strip(), ""
    return EnvInfo(
        device=device,
        device_name=name,
        peak_bandwidth_gbps=peak_bandwidth_gbps(name) if device == "cuda" else None,
        sm_count=sm_count,
        torch_version=str(torch.__version__),
        cuda_version=torch.version.cuda,
        triton_version=_triton_version(),
        driver_version=driver,
        python_version=platform.python_version(),
        platform=platform.platform(),
        git_sha=sha,
        git_dirty=None if sha is None else bool(status),
    )
