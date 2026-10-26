# ace-kernels

Fast, memory-lean PyTorch / Triton implementations of the mL1-ACE calibration loss
from [Average-Calibration-Losses](https://github.com/cai4cai/Average-Calibration-Losses)
([paper](https://arxiv.org/abs/2506.03942)).

## Layout

```
src/ace_reference/   frozen upstream losses (correctness oracle, never edited)
src/ace_kernels/
  config.py          ACEConfig: all loss options (pydantic)
  registry.py        ace_loss(input, target, config, impl=...) + implementation registry
  impls/             one module per implementation: reference, baselines
                     (compile, checkpoint), v1_vectorized, v1_clean(+compile),
                     v2_recompute / v2_grad_in_fwd (handwritten backward)
  common.py          shared pre/post-processing, bin boundaries, soft weights
  binning.py         per-bin sums W, P, G for hard/soft (shared by v1_clean, v2)
  cases.py           deterministic inputs + named edge cases (shared by tests/bench)
  instrumentation.py bytes saved for backward (saved_tensors_hooks)
src/ace_bench/       measurement: run.py (benchmarks), profile.py, report.py,
                     calculator.py (bandwidth/launch floors), timing.py, tracking.py (W&B)
bench/configs/       experiment YAMLs (each states its hypothesis)
kaggle/              job scripts + entry.py run on Kaggle GPUs via ace_bench.remote
tests/               correctness gate, parametrized over every registered impl
  golden/            stored reference loss + grad per case
scripts/make_golden.py
docs/report.md       sprint notes and results
docs/derivation.md   forward + handwritten backward math (v2)
```

## Usage

```python
from ace_kernels import ACEConfig, ace_loss

config = ACEConfig(binning="hard", to_onehot_y=True)  # paper settings otherwise
loss = ace_loss(logits, labels, config, impl="reference")
```

## Commands

```
uv run pytest tests/                       # correctness gate (CUDA tests skip without GPU)
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run python scripts/make_golden.py       # only on purpose: regenerates golden file

uv run python -m ace_bench.run bench/configs/quick_cpu.yaml        # pipeline smoke test
uv run python -m ace_bench.run bench/configs/baseline_paper.yaml --wandb   # on GPU
uv run python -m ace_bench.report results/<run_id>                 # rebuild tables/plots
uv run python -m ace_bench.profile --impl reference --binning soft --shape kits23

uv run python -m ace_bench.remote submit kaggle/run_sprint2.sh     # run on Kaggle T4, fetch results
uv run python -m ace_bench.tracking results/kaggle/<dir>           # mirror results to W&B
```

Benchmark output (`results/<timestamp>-<name>/`, git-ignored): `rows.parquet`
(source of truth), `meta.json` (config, hypothesis, env, git sha), `summary.md`,
`summary.tex`, `plots/`. With `--wandb` the same rows, config and files are
mirrored to W&B (`WANDB_API_KEY` from the environment, never logged).

Metrics per row: p50/p20/p80 ms, effective GB/s and % of peak DRAM bandwidth
(ideal bytes / time), bandwidth floor and launch floor (kernels x 5 us), CUDA
kernels per call, bytes saved for backward beyond the inputs, peak extra memory.

## Adding an implementation

Create `src/ace_kernels/impls/<name>.py` with a function decorated by
`@register("<name>", description=..., binnings=..., devices=...)` and import it in
`impls/__init__.py`. Golden, equivalence and property tests pick it up automatically.
