#!/usr/bin/env bash
# Sprint 2 on a Kaggle GPU (T4): environment check, correctness gate, baseline
# benchmarks, profiles. Run from the repo root:  bash kaggle/run_sprint2.sh
# W&B mirroring is on when WANDB_API_KEY is set (never printed).
set -euo pipefail

echo "== GPU"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

echo "== Environment (uv, locked deps)"
command -v uv >/dev/null || pip install -q uv
uv sync --frozen
uv run python - <<'PY'
import torch
assert torch.cuda.is_available(), "no CUDA device visible"
major, minor = torch.cuda.get_device_capability()
arch = f"sm_{major}{minor}"
print(torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(), arch)
assert arch in torch.cuda.get_arch_list(), (
    f"locked torch build lacks {arch}: {torch.cuda.get_arch_list()}")
PY

echo "== Correctness gate (CPU + CUDA)"
uv run pytest tests/ -q

WANDB=""
if [[ -n "${WANDB_API_KEY:-}" ]]; then WANDB="--wandb"; fi

echo "== Benchmarks"
uv run python -m ace_bench.run bench/configs/baseline_paper.yaml $WANDB
uv run python -m ace_bench.run bench/configs/baseline_scaling.yaml $WANDB

echo "== torch.profiler traces, memory snapshots, saved tensors"
for binning in hard soft; do
  for impl in reference reference_compile; do
    uv run python -m ace_bench.profile --impl "$impl" --binning "$binning" --shape kits23
  done
done

if command -v nsys >/dev/null; then
  echo "== Nsight Systems"
  mkdir -p results/nsys
  for binning in hard soft; do
    out="results/nsys/reference-$binning-kits23"
    uv run nsys profile --capture-range=cudaProfilerApi --trace=cuda,nvtx \
      --force-overwrite=true -o "$out" \
      python -m ace_bench.profile --impl reference --binning "$binning" --shape kits23 --nsys
    uv run nsys stats --report cuda_gpu_kern_sum "$out.nsys-rep" > "$out-kernels.txt" || true
  done
else
  echo "nsys not installed: skipped"
fi

if command -v ncu >/dev/null; then
  echo "== Nsight Compute (often blocked in containers: ERR_NVGPUCTRPERM)"
  uv run ncu --set basic -c 20 -o results/ncu-reference-hard-small --force-overwrite \
    python -m ace_bench.profile --impl reference --binning hard --shape small \
    > results/ncu.log 2>&1 || echo "ncu failed, see results/ncu.log"
else
  echo "ncu not installed: skipped"
fi

tar czf results.tgz results
echo "== Done: results.tgz (download from the notebook output)"
