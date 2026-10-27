# Shared by kaggle/run_*.sh (sourced): GPU info, locked env, correctness gate.
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
# PYTEST_ARGS / PYTEST_K narrow the gate for quick dev jobs (default: whole suite);
# GATE_CONTINUE=1 (dev jobs only) keeps going after failures to also get timings
if ! uv run pytest ${PYTEST_ARGS:-tests/} ${PYTEST_K:+-k "$PYTEST_K"} -q -p no:cacheprovider; then
  if [[ -n "${GATE_CONTINUE:-}" ]]; then echo "== GATE FAILED (continuing: dev job)"; else exit 1; fi
fi

WANDB=""
if [[ -n "${WANDB_API_KEY:-}" ]]; then WANDB="--wandb"; fi
