#!/usr/bin/env bash
# Sprint 11 iteration: v6 + module tests on the GPU, then a small v5-vs-v6 timing.
# bash kaggle/run_sprint11_dev.sh
export PYTEST_ARGS="tests/test_triton.py tests/test_module.py tests/test_golden.py tests/test_equivalence.py tests/test_sync.py"
export PYTEST_K="triton or v6 or module"
export GATE_CONTINUE=1
source kaggle/_setup.sh
uv run python -m ace_bench.run bench/configs/sprint11_dev.yaml $WANDB
tar czf results.tgz results
echo "== Done: results.tgz"
