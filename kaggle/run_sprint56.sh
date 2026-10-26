#!/usr/bin/env bash
# Sprints 5+6 on a Kaggle GPU: Triton forward (v3) and forward + backward (v4).
# Run from the repo root: bash kaggle/run_sprint56.sh
source kaggle/_setup.sh

echo "== Benchmarks"
uv run python -m ace_bench.run bench/configs/sprint56_paper.yaml $WANDB
uv run python -m ace_bench.run bench/configs/sprint56_scaling.yaml $WANDB

echo "== Profiles"
for impl in v2_recompute v3_triton v4_triton; do
  uv run python -m ace_bench.profile --impl "$impl" --binning hard --shape kits23
done

tar czf results.tgz results
echo "== Done: results.tgz"
