#!/usr/bin/env bash
# Sprint 3 on a Kaggle GPU: v1 ablation (loop removal, then cleanup, then compile)
# against the reference baselines. Run from the repo root: bash kaggle/run_sprint3.sh
source kaggle/_setup.sh

echo "== Benchmarks"
uv run python -m ace_bench.run bench/configs/sprint3_paper.yaml $WANDB
uv run python -m ace_bench.run bench/configs/sprint3_scaling.yaml $WANDB

echo "== Profiles"
for binning in hard soft; do
  for impl in v1_vectorized v1_clean v1_clean_compile; do
    uv run python -m ace_bench.profile --impl "$impl" --binning "$binning" --shape kits23
  done
done

tar czf results.tgz results
echo "== Done: results.tgz"
