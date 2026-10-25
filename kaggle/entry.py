"""Entry point executed on Kaggle (rendered and pushed by ``ace_bench.remote``).

Unpacks the code snapshot from the attached private dataset into /tmp (anything in
/kaggle/working becomes downloadable output, so the venv must not live there),
runs the job script, and leaves only results.tgz in /kaggle/working.
"""

import glob
import os
import shutil
import subprocess
import sys
import tarfile

JOB_SCRIPT = "__JOB_SCRIPT__"
WORK = "/tmp/ace-kernels"


def find_source() -> None:
    projects = glob.glob("/kaggle/input/**/pyproject.toml", recursive=True)
    if projects:  # Kaggle may have unpacked the archive
        shutil.copytree(os.path.dirname(projects[0]), WORK)
        return
    archives = glob.glob("/kaggle/input/**/src.tar", recursive=True)
    if not archives:
        sys.exit("code snapshot not found under /kaggle/input")
    with tarfile.open(archives[0]) as tar:
        tar.extractall(WORK, filter="data")


find_source()
os.chdir(WORK)
print(open("COMMIT").read().strip(), flush=True)
env = dict(
    os.environ,
    UV_CACHE_DIR="/tmp/uv-cache",
    UV_PROJECT_ENVIRONMENT="/tmp/ace-venv",
    UV_LINK_MODE="copy",
)
code = subprocess.run(["bash", JOB_SCRIPT], env=env).returncode
if os.path.exists("results.tgz"):
    shutil.copy("results.tgz", "/kaggle/working/results.tgz")
shutil.copy("COMMIT", "/kaggle/working/COMMIT")
sys.exit(code)
