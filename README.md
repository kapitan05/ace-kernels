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
  impls/             one module per implementation (Sprint 1: reference only)
  cases.py           deterministic inputs + named edge cases (shared by tests/bench)
  instrumentation.py bytes saved for backward (saved_tensors_hooks)
tests/               correctness gate, parametrized over every registered impl
  golden/            stored reference loss + grad per case
scripts/make_golden.py
docs/report.md       sprint notes and results
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
```

## Adding an implementation

Create `src/ace_kernels/impls/<name>.py` with a function decorated by
`@register("<name>", description=..., binnings=..., devices=...)` and import it in
`impls/__init__.py`. Golden, equivalence and property tests pick it up automatically.
