#!/usr/bin/env bash
# Sprint 4 on a Kaggle GPU: handwritten backward (v2) vs default autograd
# Run from the repo root: bash kaggle/run_sprint4.sh
source kaggle/_setup.sh

echo "== Benchmarks"
uv run python -m ace_bench.run bench/configs/sprint4_paper.yaml $WANDB
uv run python -m ace_bench.run bench/configs/sprint4_memory.yaml $WANDB

echo "== Profiles"
for binning in hard soft; do
  for impl in v2_recompute v2_grad_in_fwd; do
    uv run python -m ace_bench.profile --impl "$impl" --binning "$binning" --shape kits23
  done
done

tar czf results.tgz results
echo "== Done: results.tgz"
