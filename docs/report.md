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

### Sprint 2 baseline results: Tesla T4 (Kaggle), commit fb8ebdc, 2026-10-25

Source: `results/kaggle/fb8ebdca9b33-run-sprint2/` (W&B group `sprint2-baseline`).
Correctness gate on GPU: 255 passed (all 43 golden cases on CUDA).

Paper shapes, fwd+bwd, p50 ms (bandwidth floor = ideal bytes at 320 GB/s):

| shape | binning | reference | % peak BW | kernels/call | compile | checkpoint | BW floor | saved for bwd | peak extra |
|---|---|---|---|---|---|---|---|---|---|
| kits23 2x4x128^3 | hard | 99.4 | 0.84 | 428 | 65.3 (1.5x) | 169 (0.59x) | 0.84 | 192 MiB | 320 MiB |
| kits23 | soft | 180 | 0.47 | 981 | 38.1 (4.7x) | 279 (0.65x) | 0.84 | 704 MiB | 864 MiB |
| brats21 2x4x224^2x144 | hard | 353 | 0.82 | 428 | 241 (1.5x) | 587 | 2.89 | 662 MiB | 1106 MiB |
| brats21 | soft | 606 | 0.48 | 981 | 150 (4.0x) | 941 | 2.89 | 2.4 GiB | 3.0 GiB |
| amos22 2x16x176^2x64 | hard | 801 | 0.32 | 1604 | 528 (1.5x) | 1287 | 2.58 | 726 MiB | 1.3 GiB |
| amos22 | soft | 1491 | 0.17 | 3813 | **2894 (0.52x)** | 2118 | 2.58 | 2.6 GiB | 3.4 GiB |

Where the GPU time goes (torch.profiler traces, kits23 fwd+bwd):

| impl | GPU ms/step | scatter/gather kernels | other |
|---|---|---|---|
| reference hard | 97 | **84%** | 16% elementwise |
| reference soft | 167 | 51% | 48% elementwise (soft-weight temporaries) |
| compile hard | 65 | **85%** | 14% fused Triton |
| compile soft | 40 | 43% | 57% fused Triton |

**Findings**
1. The reference reaches 0.2-0.8% of peak bandwidth: 100-500x above the floor.
2. It is **not launch-bound**: launch floor (kernels x 5 us) is 2-19 ms vs 100-1500 ms
   measured, and CUDA-graph replay changes nothing for hard binning (99.4 -> 99.4 ms).
3. The bottleneck is **atomic contention in scatter_add/scatter_reduce**: millions of
   voxels add into 20 bin addresses per (b, c), so atomics serialize (84% of GPU time
   for hard). Matches the meeting note "replace scatter_add with tiling". Fix:
   per-block private histograms, one atomic per bin per block (Triton, Sprint 5).
4. torch.compile: 1.5x (hard) / 4-4.7x (soft), still ~2% of peak; it keeps the
   scatter atomics. With 16 classes (amos22) compile is **2x slower** than eager.
5. The reference soft loss cannot be captured in a CUDA graph: `torch.tensor` sentinels
   inside the loss are host->device copies (hard and compile capture fine).
6. Memory: soft keeps 11x the input for backward (2.4 GiB at BraTS size). Checkpoint
   keeps 0 extra but its **peak is unchanged** (the backward recomputes everything at
   once) and it costs 1.5-1.7x time: no free lunch from off-the-shelf recomputation.
7. Kaggle limits: `nsys` not installed; `ncu` blocked (`ERR_NVGPUCTRPERM`, no access
   to GPU performance counters). Kernel-level counters need another GPU host.

**Consequence for Sprint 3**: v1_vectorized / v1_clean still scatter into few bins per
slice, so contention remains; expect modest gains, not the floor. Their GPU run tells
how much is loop/sync overhead vs contention.

## Sprint 3: v1, vectorized PyTorch (2026-10-25)

**Done**
- `ace_kernels/common.py`: shared pre/post-processing reproducing the reference
  (activation, one-hot, background, empty classes, weights, reduction), the
  bit-identical bin boundaries, and `soft_assign` (reference soft weights).
- `v1_vectorized`: only the B x C loop removed (flat `[B*C*K]` bins, offset
  `r*K`); otherwise the reference's ops (one-hot, `scatter_reduce(mean)`, six
  scatters for soft, boolean-mask NaN fill).
- `v1_clean`: + plain sums (no `scatter_reduce`), no one-hot for index targets
  (gather the true-class bin, scatter on `[B, N]`), `torch.where` masking (no host
  sync; guarded division so empty bins give zero gradient, not NaN).
- `v1_clean_compile`: `torch.compile(v1_clean)`.
- Tests: all three pass the 43 golden cases (loss + grad) and the hypothesis
  equivalence vs reference; 255 CPU tests pass. New CUDA-only `test_sync.py`:
  v1_clean must run fwd+bwd under `set_sync_debug_mode("error")`; documents that
  the reference syncs.
- Experiments: `sprint3_paper.yaml` (ablation on paper shapes, with CUDA-graph
  rows) and `sprint3_scaling.yaml`; Kaggle job `kaggle/run_sprint3.sh`.

**Findings (device-independent, 2x4x32^3, saved for backward beyond inputs)**

| impl | hard: tensors / bytes | soft: tensors / bytes | CPU fwd+bwd hard / soft (sanity only) |
|---|---|---|---|
| reference | 38 / 3.00x | 167 / 11.00x | 9.9 / 16.6 ms |
| v1_vectorized | 10 / 3.00x | 27 / 11.00x | 7.7 / 9.5 ms |
| v1_clean | 8 / 3.00x | 31 / 11.75x | 3.1 / ~10 ms |

1. Vectorizing cuts the number of saved tensors 4-6x but not the bytes: default
   autograd must keep per-voxel tensors (hard: int64 bin index = 2x input + softmax
   output; soft: interpolation weights and products). The memory fix needs the
   handwritten backward (Sprint 4), not better PyTorch ops.
2. A single scatter over concatenated left/right contributions (fewer kernels)
   made soft memory worse (15.75x): autograd keeps the 2N-sized concatenations.
   Reverted to per-side scatters. Fewer kernels is not automatically better.
3. "Small (uint8) bin indices" is not possible in PyTorch: scatter ops require int64
   indices. Moves to the Triton kernel (Sprint 5).
4. The index-target path (no one-hot) saves writing a `[B, C, N]` float tensor
   but autograd keeps its gathered `[B, N]` int64 indices (+0.75x for soft at C=4).

### Sprint 3 GPU results: Tesla T4 (Kaggle), commit 33facc7, 2026-10-25

Source: `results/kaggle/33facc78d8f3-run-sprint3/` (W&B group `sprint3-v1`).
Correctness gate on GPU: 452 passed. 208 benchmark rows, 0 OOM, 8 expected
`capture_failed` (reference soft).

fwd+bwd p50 ms, speedup vs reference (T4, 320 GB/s; all rows: ~same time as CUDA graph):

| shape | binning | reference | compile | v1_vectorized | v1_clean | v1_clean_compile | best % peak BW |
|---|---|---|---|---|---|---|---|
| kits23 | hard | 99.4 | 66.1 (1.5x) | 84.3 (1.2x) | 37.4 (2.7x) | **20.1 (4.9x)** | 4.2 |
| brats21 | hard | 352 | 238 (1.5x) | 308 (1.1x) | 137 (2.6x) | **74 (4.7x)** | 3.9 |
| amos22 | hard | 805 | 501 (1.6x) | 599 (1.3x) | 218 (3.7x) | **112 (7.2x)** | 2.3 |
| kits23 | soft | 182 | **38.9 (4.7x)** | 132 (1.4x) | 118 (1.5x) | 57 (3.2x) | 2.2 |
| brats21 | soft | 610 | **139 (4.4x)** | 440 (1.4x) | 385 (1.6x) | 201 (3.0x) | 2.1 |
| amos22 | soft | 1494 | 2704 (0.55x) | 729 (2.1x) | 550 (2.7x) | **335 (4.5x)** | 0.8 |

GPU time in scatter/gather kernels (kits23 profiles): v1_vectorized 92% / 65%
(hard / soft), v1_clean 83% / 61%, v1_clean_compile 86% / 89%.

**Hypothesis check**
- Loop removal gives a large speedup: **rejected**. Kernels drop 428 -> 78, time only
  1.1-1.3x (2x for soft at 16 classes). Confirms Sprint 2: not launch-bound.
- v1_clean adds a real gain: **confirmed**, 2.6-3.7x hard. Most of it is dropping
  `scatter_reduce(mean)` for `scatter_add` (mean-reduce is far slower under
  contention), plus no one-hot at 16 classes.
- Becomes CUDA-graph capturable: **confirmed** (no host<->device copies; tested). But
  graphs bring ~0 speedup anywhere: launch overhead is not the bottleneck.
- Saved-for-backward unchanged: **confirmed** (192 MiB hard, 704-750 MiB soft at
  kits23). Compile lowers it (128 / 356 MiB) by fusing intermediates.
- Below the bandwidth floor: **confirmed**, best 4.2% of peak.

**New findings**
1. Scatter contention is still 83-92% of GPU time in every v1 variant: PyTorch ops
   cannot avoid it. Per-block private histograms (Triton, Sprint 5) are the lever.
2. Vectorizing raises **peak** memory (kits23 hard 320 -> 448-496 MiB, brats21 soft
   3.0 -> 3.8 GiB): all B x C slices' temporaries exist at once. Compile brings it
   back down (256 MiB / 2.2 GiB). Speed and peak memory trade off in PyTorch.
3. reference_compile wins for soft at 4 classes (4.7x) but is 1.8x *slower* than
   eager at 16 classes; v1_clean_compile is the only variant that is fast for both
   class counts.

**Harness issues found and fixed during the 5 Sprint 3 runs** (each a real bug):
- GPU-vs-CPU gradient tolerance too tight for soft binning with ~3 voxels per bin
  (cancelling terms amplify fp32 reordering ~1000x); hypothesis made deterministic.
- Exact-tie subgradient (|x| at 0) differs between eager and compiled kernels; tie
  cases check loss exactly and gradients for finiteness for compiled impls.
- Wrong assumption corrected: the reference's boolean NaN fill does not sync; the
  soft reference is not graph-capturable because of a host->device copy.
- 38 rows OOM'd from ~10 GiB leaked across rows. Groups now run in isolated
  processes, and each row records memory left allocated. That column located the leak
  in `measure_saved_tensors`: the pack hook returned the tensor itself, creating
  uncollectable tensor<->grad_fn cycles for saved outputs (fixed + regression test).

## Sprint 4: v2, handwritten backward (2026-10-26)

**Done (GPU numbers pending `kaggle/run_sprint4.sh`)**
- `docs/derivation.md`: forward, backward (`∂f/∂p_i = Σ_j s_kj/(K_ne W_kj)
  [w_ij + w'_ij (p_i - y_i - d_kj)]`), activation chain, edge cases, memory table.
- `binning.py`: per-bin sums shared by v1_clean and v2 (refactor, v1_clean unchanged).
- `v2_recompute`: one `autograd.Function` logits -> loss; forward without autograd;
  saves inputs + `coef`, `d` (`[B, C, K]`); backward recomputes softmax/bins.
- `v2_grad_in_fwd`: gradient computed in forward, saved (1x input), scaled in
  backward (mean/sum reductions; falls back to recompute for `none`).
- Mask/weights/reduction gradient via autograd on `[B, C]` (exact semantics).
- Tests (380 pass on CPU): both variants pass all 43 golden cases (loss + grad) and
  hypothesis equivalence; **fp64 gradcheck** (finite differences) for 6 configs x
  hard/soft; **fp64 match vs autograd** through v1_clean to 1e-9 on 30 random cases;
  saved bytes exactly `2 B C K` floats (recompute) / 1x input (grad_in_fwd); CUDA
  no-sync and graph-capture tests extended to v2.

**Findings (device-independent, 2x4x32^3)**

| impl | saved for backward, hard | soft |
|---|---|---|
| reference | 3.00x input | 11.00x |
| v1_clean | 3.00x | 11.75x |
| v2_recompute | **0.0012x** (1.3 KB) | **0.0012x** |
| v2_grad_in_fwd | 1.00x | 1.00x |
