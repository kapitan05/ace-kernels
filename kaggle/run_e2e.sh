#!/usr/bin/env bash
# End-to-end training check only (paper network, AMP), with a reference control.
# Run from the repo root: bash kaggle/run_e2e.sh
export PYTEST_ARGS="tests/test_bench.py -k e2e"
source kaggle/_setup.sh
uv run python -m ace_bench.e2e bench/configs/sprint8_e2e.yaml $WANDB
tar czf results.tgz results
echo "== Done: results.tgz"
