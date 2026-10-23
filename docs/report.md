# Report

## Sprint 1: Project skeleton + correctness harness (2026-10-23)

**Done**
- Vendored upstream `hardl1ace.py` / `softl1ace.py` from commit `faa93c7` byte-identical
  (`src/ace_reference/`, SHA-256 checked by a test). Upstream has no LICENSE file.
- `ace_loss(input, target, config, impl)` + registry; `reference` registered.
- 43 named cases (`src/ace_kernels/cases.py`): paper setting 3D/2D/1D, 16 classes,
  one-hot vs label targets, saturated logits, empty class (ignored / kept), no
  background, class weights (vector / scalar), reductions sum / none, sigmoid binary,
  1 and 50 bins, raw / constant / exact-edge / 0-1 probabilities, `right=True`, soft
  `empty_weight`.
- Tests (101 pass on CPU, 44 CUDA variants skip without GPU):
  - `test_golden`: every impl reproduces stored reference loss + grad (fp32 tol:
    loss rtol 1e-5 / atol 1e-6, grad rtol 1e-4 / atol 1e-7).
  - `test_oracle`: reference equals an independent float64 NumPy restatement.
  - `test_equivalence`: hypothesis random shapes/options vs reference; voxel-order
    invariance; loss in [0, 1], finite grads.
  - `test_saved_tensors`: instrumentation checks; reference saves > input-size bytes.
  - `test_registry`, `test_reference_frozen`.
- Mutation check (throwaway script): deliberately wrong impls are caught.
  `loss * (1 + 1e-4)`: 43/43 cases fail; `num_bins + 1`: 40/43; doubled gradient:
  43/43; flipped `right`: 2/43 (only the exact-edge cases, which is why they exist).

**Findings**
1. Bin boundaries are part of the loss definition. The reference builds them with
   `torch.linspace(0, 1 + eps32, K + 1)` in float32, which differs from NumPy's
   linspace by ~1 ulp; probabilities exactly on an edge then land in a different bin.
   Optimized kernels must reuse that exact boundary tensor (not `floor(p * K)`), or
   the exact-edge tests fail.
2. The reference casts to fp32 internally, so fp64 `gradcheck` is impossible on it;
   gradcheck applies to our implementations only.
3. Paper configs: `num_bins=20`, softmax, one-hot targets, `include_background=True`,
   `ignore_empty_classes=True`. Training shapes: batch 2, patches up to
   224x224x144, 4 or 16 classes. `amos22.yaml` notes the patch was reduced "for
   softl1ace to prevent CUDA OOM": memory is a real bottleneck in the paper.
4. Environment: torch 2.14.1, monai 1.6.1 (upstream used torch 2.5, monai 1.4). The
   golden file pins reference behaviour in this environment.

**Verify**
```
uv sync
uv run pytest tests/ -v
uv run ruff check . && uv run ruff format --check . && uv run mypy
```
