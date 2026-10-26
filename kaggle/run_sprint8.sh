#!/usr/bin/env bash
# Sprint 8a on a Kaggle GPU: fused activation (v5) and half-precision logits.
# Run from the repo root: bash kaggle/run_sprint8.sh
source kaggle/_setup.sh

echo "== Benchmarks"
uv run python -m ace_bench.run bench/configs/sprint8_paper.yaml $WANDB
uv run python -m ace_bench.run bench/configs/sprint8_fp16.yaml $WANDB

echo "== Profiles"
for binning in hard soft; do
  uv run python -m ace_bench.profile --impl v5_triton --binning "$binning" --shape kits23
done

tar czf results.tgz results
echo "== Done: results.tgz"
