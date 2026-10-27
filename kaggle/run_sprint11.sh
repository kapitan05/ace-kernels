#!/usr/bin/env bash
# Sprint 11 on a Kaggle GPU: v6 (O(1) index, tensor-core reduction, fused
# epilogue, torch.library ops), realistic inputs, deep-supervision sizes, accuracy,
# end-to-end with torch.compile. Run from the repo root: bash kaggle/run_sprint11.sh
source kaggle/_setup.sh

echo "== Benchmarks"
uv run python -m ace_bench.run bench/configs/sprint11_paper.yaml $WANDB
uv run python -m ace_bench.run bench/configs/sprint11_realistic.yaml $WANDB
uv run python -m ace_bench.run bench/configs/sprint11_deepsup.yaml $WANDB
uv run python -m ace_bench.run bench/configs/sprint11_verify.yaml $WANDB

echo "== End-to-end training (paper network, AMP, torch.compile)"
uv run python -m ace_bench.e2e bench/configs/sprint11_e2e.yaml $WANDB

echo "== Profiles"
for binning in hard soft; do
  uv run python -m ace_bench.profile --impl v6_triton --binning "$binning" --shape kits23
done

tar czf results.tgz results
echo "== Done: results.tgz"
