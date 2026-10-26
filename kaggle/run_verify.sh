#!/usr/bin/env bash
# Accuracy verification against a float64 truth (no full gate: quick re-check).
# Run from the repo root: bash kaggle/run_verify.sh
export PYTEST_ARGS="tests/test_golden.py -k cuda"
source kaggle/_setup.sh
uv run python -m ace_bench.run bench/configs/verify_truth.yaml $WANDB
tar czf results.tgz results
echo "== Done: results.tgz"
