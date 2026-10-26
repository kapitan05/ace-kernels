#!/usr/bin/env bash
# Quick Triton iteration on a Kaggle GPU (Triton has no macOS builds): env setup
# plus only the tests that exercise Triton code. bash kaggle/run_triton_dev.sh
export PYTEST_ARGS="tests/test_triton.py tests/test_golden.py tests/test_sync.py -k triton"
source kaggle/_setup.sh
tar czf results.tgz results 2>/dev/null || true
echo "== Done"
