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

## Sprint 2: Measurement infrastructure (2026-10-25)

**Done (infrastructure; GPU numbers pending the Kaggle run)**
- `ace_bench.run`: YAML experiment (hypothesis + grid) -> one row per
  impl x binning x shape x mode [x CUDA graph]: p50/p20/p80 ms (CUDA events, L2
  flushed per rep, warmup), effective GB/s and % peak BW, bandwidth floor, kernels
  per call + launch floor, saved-for-backward bytes beyond inputs, peak extra
  memory; OOM / errors / failed graph capture recorded as row status.
  Output: parquet + meta.json (env, git sha) + markdown/LaTeX tables + plots.
- `ace_bench.profile`: torch.profiler trace (Perfetto), op table, saved tensors,
  CUDA memory snapshot (memory_viz), `torch._dynamo.explain`, `--nsys` mode
  (NVTX ranges + cudaProfilerApi capture range).
- Baselines registered as impls (so they are correctness-tested):
  `reference_compile` (torch.compile), `reference_checkpoint`
  (torch.utils.checkpoint). CUDA graphs: `cuda_graph: true` benchmark option.
- W&B mirror (`--wandb`): results table, config + hypothesis + env, files as
  artifact. Parquet stays the source of truth.
- Kaggle from the command line: `python -m ace_bench.remote submit kaggle/run_sprint2.sh`
  uploads the committed tree as a private dataset, runs `kaggle/run_sprint2.sh` on a
  T4 (env check incl. GPU arch in torch build, pytest gate, both configs, profiles,
  nsys/ncu if present) and downloads `results.tgz` to `results/kaggle/<sha>-<job>/`.
  No secrets on Kaggle; W&B upload happens from the Mac
  (`python -m ace_bench.tracking results/kaggle/<dir>`).
- Configs: `baseline_paper.yaml` (paper shapes kits23 / acdc17 / brats21 / amos22,
  hard+soft, fwd+fwd_bwd, 3 impls, CUDA graph rows), `baseline_scaling.yaml`
  (cubes 32^3-192^3), `quick_cpu.yaml` (smoke).
- Tests: 162 pass on CPU (bench pipeline, configs valid, baselines vs golden).

**Findings so far (device-independent, measured on CPU)**
1. `torch.compile` traces the reference with 0 graph breaks (Dynamo unrolls the
   B x C loop), so compile is a real baseline, not a strawman.
2. Saved for backward (2x4x32^3, soft): 167 tensors, 11x the input size; hard: 3x.
   `reference_checkpoint` saves 0 extra bytes but costs ~1.6-2.4x time (CPU).
   This is the memory problem behind the paper's "reduce patch for softl1ace to
   prevent CUDA OOM".

**Incident**: `ruff --fix/format` ran over the untracked upstream clone in the repo
dir (`Average-Calibration-Losses/`, 44 files). Verified every change was
ruff-only (HEAD + same ruff == working copy), then restored with `git checkout`.
Ruff is now limited to `src/`, `tests/`, `scripts/`.

**Verify**
```
uv run pytest tests/ -q && uv run ruff check . && uv run mypy
uv run python -m ace_bench.run bench/configs/quick_cpu.yaml       # look at summary.md + plots
uv run python -m ace_bench.profile --impl reference --binning soft --shape small
```
GPU run: `uv run python -m ace_bench.remote submit kaggle/run_sprint2.sh`, then the
baseline section is written from `results/kaggle/<sha>-run-sprint2/`.
