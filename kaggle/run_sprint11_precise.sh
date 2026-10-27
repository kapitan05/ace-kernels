#!/usr/bin/env bash
# Sprint 11.4: precise vs fast exp (accuracy vs float64 truth + timing).
# bash kaggle/run_sprint11_precise.sh
export PYTEST_ARGS="tests/test_triton.py tests/test_golden.py tests/test_sync.py"
export PYTEST_K="v6 or triton"
source kaggle/_setup.sh
uv run python -m ace_bench.run bench/configs/sprint11_precise.yaml $WANDB
tar czf results.tgz results
echo "== Done: results.tgz"
