#!/usr/bin/env bash
# Sprint 5 on a Kaggle GPU: Triton private-histogram forward (hard binning).
# Run from the repo root: bash kaggle/run_sprint5.sh
source kaggle/_setup.sh

echo "== Benchmarks"
uv run python -m ace_bench.run bench/configs/sprint5_paper.yaml $WANDB
uv run python -m ace_bench.run bench/configs/sprint5_scaling.yaml $WANDB

echo "== Profiles"
for impl in v3_triton v2_recompute; do
  uv run python -m ace_bench.profile --impl "$impl" --binning hard --shape kits23
done

tar czf results.tgz results
echo "== Done: results.tgz"
