#!/usr/bin/env bash
# Sprint 8 on a Kaggle GPU: fused activation (v5), half-precision logits, and the
# end-to-end training check.
# Run from the repo root: bash kaggle/run_sprint8.sh
source kaggle/_setup.sh

echo "== Benchmarks"
uv run python -m ace_bench.run bench/configs/sprint8_paper.yaml $WANDB
uv run python -m ace_bench.run bench/configs/sprint8_fp16.yaml $WANDB

echo "== End-to-end training (paper network, AMP)"
uv run python -m ace_bench.e2e bench/configs/sprint8_e2e.yaml $WANDB

echo "== Profiles"
for binning in hard soft; do
  uv run python -m ace_bench.profile --impl v5_triton --binning "$binning" --shape kits23
done

tar czf results.tgz results
echo "== Done: results.tgz"
