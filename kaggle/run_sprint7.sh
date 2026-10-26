#!/usr/bin/env bash
# Sprint 7 on a Kaggle GPU: Triton soft binning (forward + backward).
# Run from the repo root: bash kaggle/run_sprint7.sh
source kaggle/_setup.sh

echo "== Benchmarks"
uv run python -m ace_bench.run bench/configs/sprint7_paper.yaml $WANDB
uv run python -m ace_bench.run bench/configs/sprint7_memory.yaml $WANDB

echo "== Profiles"
for impl in v3_triton v4_triton; do
  uv run python -m ace_bench.profile --impl "$impl" --binning soft --shape kits23
done

tar czf results.tgz results
echo "== Done: results.tgz"
