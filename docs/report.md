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

**Done**
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

### Sprint 4 GPU results: Tesla T4 (Kaggle), commit a2696e8, 2026-10-26

Source: `results/kaggle/a2696e81824d-run-sprint4/` (W&B group `sprint4-v2`).
Gate: 675 passed. Paper grid: 152 ok + 8 expected capture_failed, no leaks.
Memory grid: 29 ok, 3 OOM (real, not leaks: left-allocated 0 everywhere).

fwd+bwd, kits23 / brats21 (ms; saved-for-backward MiB; peak MiB of the isolated call):

| impl | hard kits23 | soft kits23 | soft brats21 |
|---|---|---|---|
| reference | 100 ms; 192; 320 | 183; 704; 864 | 605; 2426; 3008 |
| v1_clean | 38; 192; 496 | 116; 752; 1120 | 383; 2591; 3866 |
| v1_clean_compile | **20**; 128; 256 | **55**; 356; 640 | **196**; 1227; 2206 |
| v2_recompute | 47; **0**; 496 | 126; **0**; 992 | 416; **0**; 3424 |
| v2_grad_in_fwd | 47; 64 (1x); 496 | 127; 64; 992 | 416; 221; 3424 |

Memory headroom (2 x 4 x V^3, soft, fwd+bwd):

| V | reference | v1_clean | v2_recompute |
|---|---|---|---|
| 256 | 1409 ms, peak 6.9 GiB | **OOM** | 954 ms, peak 7.9 GiB |
| 288 | **OOM** | **OOM** | 1364 ms, peak 11.3 GiB |

**Findings**
1. Saved-for-backward drops to 0 (exact: 2 B C K floats) as designed, at every size.
   v2 trains at 288^3 soft where the reference and v1 OOM: the paper's failure mode.
2. Peak of the isolated loss call does not drop: it is set by voxel-sized forward
   temporaries (int64 bin index, gathers) and backward recomputation, not by saved
   tensors. In training the distinction matters (saved tensors coexist with all
   network activations until backward; temporaries are freed at once), but a fully
   lean loss needs kernels without temporaries (Triton).
3. Time: v2 is 1.1-1.2x slower than v1_clean (recompute pass) and ~2x slower than
   v1_clean_compile; the forward still uses the contended scatter.
4. v2_grad_in_fwd has no advantage (same time, keeps 1x input): dropped from now on.

## Sprints 5+6: Triton kernels, hard binning (2026-10-26)

**Done**
- `kernels/hard_hist.py` (Sprint 5): per-block private histograms; each program bins
  a block of voxels for every channel against the reference's edges (bucketize
  semantics) and issues one atomic add per bin and statistic. `v3_triton` = v2
  backward + this forward.
- `kernels/hard_bwd.py` (Sprint 6): one pass over `[C, BLOCK]` tiles: re-bin, gather
  `coef`, softmax/sigmoid backward across channels in registers. `v4_triton` =
  Triton forward + Triton backward; saves input, `coef`, presence, `[B, C]` jacobian.
- Class presence from the bin statistics (`sum_k G_k > 0 == target.sum() > 0`).
- Tests (GPU): counts and sum_y exactly equal to PyTorch; sum_p vs a float64 truth
  (within 1e-5, never less accurate than scatter_add); v4 gradient == v2; golden,
  sync, graph capture; 256^3 grid regression. Gate on T4: 973 passed.

**Bugs found on the GPU** (each fixed + regression test)
1. Dev run 1: my test fed raw single-channel logits (softmax is ignored for one
   channel) -> out-of-range bins -> device assert poisoned the CUDA context.
2. Dev run 2: sum_p differed from scatter_add by up to 4.9e-5 relative at 1e5
   voxels per slice. Not a bug: fp32 summation order; vs float64 the kernel (tree
   sums per block) is at least as accurate as scatter_add (sequential atomics).
3. Run 1: kernels failed at >= 256^3 ("invalid argument"): voxel blocks on grid axis
   1, limited to 65535. Moved to axis 0.
4. Run 1 profile: 51% of v4 time was `nonempty_classes`' scatter of B*N labels into
   B*C counters (same contention, outside the kernels). Replaced by the presence
   from bin stats: v4 kits23 12.1 -> 4.9 ms.

### Results: Tesla T4, commit dc02925 (`results/kaggle/dc029258d7b6-run-sprint56/`)

fwd+bwd p50 ms (speedup vs reference), all 128 rows ok, no leaks:

| shape | reference | v1_clean_compile | v2_recompute | v3_triton | **v4_triton** |
|---|---|---|---|---|---|
| kits23 | 99.5 | 19.0 (5.2x) | 38.2 | 10.4 (9.6x) | **4.86 (20.5x)** |
| acdc17 | 50.0 | 9.6 | 18.9 | 5.2 | **2.40 (20.8x)** |
| brats21 | 342 | 66.8 | 140 | 36.0 | **17.2 (19.9x)** |
| amos22 | 794 | 108 | 232 | 41.9 | **20.4 (39x)** |

- % of peak bandwidth (fwd+bwd): reference 0.3-0.8, v1_clean_compile 2.4-4.4,
  **v4 12.7-17.5** (constant 16-17% from 128^3 to 288^3).
- Peak memory (fwd+bwd, isolated call): v4 is **2.5x below the reference** and
  ~equal to one input-sized tensor (kits23 320 -> 128 MiB; brats21 1106 -> 441;
  288^3: 1.46 GiB vs 2.9 for v1_clean_compile, 5.7 for v2). Saved for backward: 0.
- Forward alone: v3/v4 3.3 ms vs reference 69 ms (kits23): **21x**.
- v4 GPU time per step (kits23 profile, 4.70 ms): histogram kernel 2.55 ms (54%),
  softmax (forward + backward recompute) 1.47 ms (31%), backward kernel 0.56 ms
  (12%). The contended scatters are gone entirely.

**Hypothesis check**: v3 several x faster than v2 and dominated by the PyTorch
backward: **confirmed** (3.7x; backward was ~60% of v3). v4 within a few x of the
floor: **partly** (12-17% of peak; 6x above the floor). Peak memory ~ input size:
**confirmed**.

**Next levers** (measured): the histogram kernel runs at ~6% of peak bandwidth
(2.55 ms vs 0.16 ms of traffic): it is compute-bound on the `[BLOCK, K]`
compare-and-reduce. Softmax twice = 31%: fuse into both kernels (Sprint 8).

## Sprint 7: Triton soft binning (2026-10-26)

**Done**
- `kernels/soft_hist.py`: `soft_assign_tl` (reference half boundaries, sentinel
  table, IEEE `div_rn` weights) + per-block private histograms through a
  `[BLOCK, K]` weight matrix. `kernels/soft_bwd.py`: two-sided gradient
  (`w' = -/+1/S`) + activation backward in registers. v3/v4 now cover soft.
- Benchmark output check (`5423f09`): every row's loss and gradient compared with
  the reference on the same full-size input (`loss_rel_err`, `grad_rel_err`,
  WARNING above 1e-4), untimed, fallback v2_recompute when the reference OOMs.
- GPU gate: 1269 passed (dev run: 578; soft kernels passed on their first GPU run).

### Results: Tesla T4, commit 5423f09 (`results/kaggle/5423f093984d-run-sprint7/`)

fwd+bwd, soft binning, p50 ms (speedup vs reference); peak MiB:

| shape | reference | reference_compile | v1_clean_compile | v2_recompute | v3_triton | **v4_triton** |
|---|---|---|---|---|---|---|
| kits23 | 182; 864 | 39.8; 688 | 56.2; 640 | 120; 992 | 39.2; 896 | **7.4 (24.6x); 128** |
| acdc17 | 92.9; 432 | 19.2 | 27.3 | 58.2 | 19.8 | **3.3 (28x); 64** |
| brats21 | 607; 3008 | 136; 2388 | 199; 2206 | 393; 3424 | 135; 3092 | **26.4 (23x); 441** |
| amos22 | 1492; 3360 | 2690 (0.55x) | 333 | 571 | 151 | **29.8 (50x); 484** |

Memory sweep (2 x 4 x V^3 soft): reference OOM at 288^3; v4 at 288^3: 92.9 ms,
**peak 1.46 GiB** (v2_recompute 11.3 GiB, reference 6.9 GiB already at 256^3).

- v4 soft: 8.6-12.7% of peak bandwidth; 5.4x faster than the best PyTorch version
  at 4 classes (reference_compile), 11x at 16 classes (v1_clean_compile).
- Peak memory 6.8x below the reference (kits23 864 -> 128 MiB, brats21 3.0 GiB ->
  441 MiB) and ~1x the input size; saved for backward: per-bin tensors only.
- v4 soft GPU time (kits23, 6.49 ms): soft histogram kernel 4.14 ms (64%),
  torch.softmax x2 1.47 ms (23%), backward kernel 0.76 ms (12%).
- v3 soft is dominated by v2's PyTorch backward (elementwise kernels 60%).

**Output check findings** (new; full-size inputs)
1. Triton v3/v4: gradients within 0.8e-5..8.4e-5 of the reference, every row.
2. `torch.compile` (reference_compile, v1_clean_compile), soft only: loss matches but
   gradients deviate by 0.46-1.4% (relative to the max gradient) on every paper
   shape. **Correction (float64 verification run, `47c938d`):** this is not a
   compile defect. Max-norm gradient errors are dominated by ill-conditioned
   voxels: in bins where mean_p - mean_y is ~0, the |x| kink makes the gradient
   sign depend on rounding, so any change of summation order flips the gradient of
   the whole bin. Even the reference itself is 1-54% (max norm) from a float64
   truth. See "Accuracy against float64 truth" below.
3. v2_recompute soft at 256^3 / 288^3: loss 6.2e-5 / 1.3e-4, grad 1.4e-4 / 2.5e-4
   vs the reference (fp32 sums over ~16M voxels per bin; v4's tree sums deviate
   less). Comparing two fp32 results cannot say which is closer to the exact
   value: next step is a float64-truth comparison in the output check.

## Accuracy against float64 truth (2026-10-26)

Verification run `results/kaggle/47c938da465d-run-verify/` (T4): soft + hard, kits23 /
brats21 / amos22 / 256^3, six implementations, each compared with v2_recompute in
float64 (formula validated by gradcheck) on the same input.

**Loss error vs float64** (well-conditioned):

| | hard | soft |
|---|---|---|
| reference | 4.1e-7 .. 2.2e-6 | 3.3e-8 .. 1.3e-5 |
| v4_triton | 3.3e-8 .. 8.7e-7 | 2.2e-8 .. 8.1e-7 |
| v5_triton | 2.1e-8 .. 9.7e-7 | 2.2e-8 .. 8.1e-7 |
| v1_clean_compile / v2_recompute | ~reference | up to 7.5e-5 (256^3) |

The Triton kernels' loss is 10-100x closer to the exact value than the reference's
at large volumes (block-wise tree sums instead of one long sequence of fp32 atomic
adds); v2 / v1 (one scatter_add over all slices) are the least accurate at 256^3.

**Max-norm gradient error vs float64 is not a meaningful metric at scale**: up to
0.54 for every implementation including the reference. Two mechanisms: (1) the |x|
kink: the gradient of a bin is +-1/(K W) with the sign of mean_p - mean_y; bins with
that difference ~0 flip sign under any reordering; (2) hard binning: a probability
within ~1e-7 of an edge is binned differently in fp32 vs fp64. This is a property
of the loss (L1 of per-bin differences), not of an implementation. The output
check now reports the relative L2 gradient error and flags on loss + L2 (`next`
commit); max-norm errors stay as diagnostics.

### L2 gradient verification (`results/kaggle/55b7d1a680c0-run-verify/`, T4)

Relative L2 gradient error vs float64 truth (all 48 rows): **6e-5 .. 5.5e-4 for every
implementation, the reference itself 6e-5 .. 4.3e-4**: all implementations compute the
gradient equally accurately as a whole; the max-norm outliers are isolated voxels.

Loss error vs float64 (worst over kits23, brats21, amos22, 256^3): v4_triton 3e-8 ..
1e-6 and v5_triton 2e-8 .. 6e-7 vs reference up to 1.3e-5.

Flagged (worse than 2x the reference's own error vs truth), all explained:
- v1_clean_compile / v2_recompute, soft, brats21 and 256^3: loss 1.7e-5 / 7.5e-5 vs
  reference 4.7e-6 / 1.3e-5. One scatter_add over all slices accumulates more fp32
  rounding than the reference's per-slice sums. Intermediate steps only.
- v5_triton, hard, brats21: L2 gradient 2.9e-4 vs reference 6.2e-5 (loss 30x *more*
  accurate). In-kernel softmax uses Triton's fast exp (~1 ulp from torch), so more
  near-edge voxels change bin than between fp32 PyTorch implementations. Candidate
  fix: precise libdevice exp in `kernels/act.py` (cost to be measured).

## Sprint 8: fused softmax (v5), half precision, end-to-end training (2026-10-27)

**Done**
- 8a `v5_triton`: both kernels read logits; softmax/sigmoid in registers
  (`kernels/act.py`, same functions forward and backward); fp16/bf16 logits read
  natively, fp32 math; gradient in the logits' dtype. Existing kernels generalized
  (v3/v4 call them with in-kernel activation off).
- 8b `ace_bench.e2e`: the paper's training step (SegResNetDS 32 filters, deep
  supervision, Dice + CE + ACE, AdamW, AMP) on synthetic learnable volumes,
  identical batches per implementation; loss curves, step time, peak memory,
  largest cubic patch.
- Gate (T4): 1373 passed. Two earlier gate failures were test calibration (bound on
  near-boundary voxels; noise-level accuracy order), fixed.

### Results: Tesla T4, commit 1ab2f01 (`results/kaggle/1ab2f019d85e-run-sprint8/`)

fwd+bwd p50 ms (speedup vs reference); peak MiB, fp32 logits:

| shape | binning | reference | best PyTorch | v4_triton | **v5_triton** |
|---|---|---|---|---|---|
| kits23 | hard | 99.5; 320 | 19.3; 256 | 5.1; 128 | **3.6 (27.7x); 64** |
| brats21 | hard | 352; 1106 | 69.8; 884 | 18.7; 441 | **13.5 (26x); 221** |
| amos22 | hard | 804; 1301 | 109; 998 | 21.2; 484 | **14.3 (56x); 242** |
| kits23 | soft | 184; 864 | 39.2; 688 | 7.8; 128 | **6.3 (29x); 64** |
| brats21 | soft | 608; 3008 | 136; 2388 | 26.0; 441 | **20.8 (29x); 221** |
| amos22 | soft | 1492; 3360 | 334; 2208 | 29.7; 484 | **22.5 (66x); 242** |

- v5: 13-23% of peak bandwidth; peak memory **half of v4** (no probability tensor)
  and 5x (hard) / 13.6x (soft) below the reference; equal to one input tensor.
- fp16 logits: same time (the histogram kernel is compute-bound, not traffic-bound),
  peak memory halves again (kits23: 32 MiB vs 896 MiB for the reference, soft).
- v5 GPU time (kits23 hard, 3.57 ms): histogram kernel 2.92 ms (82%), backward
  kernel 0.55 ms (15%), rest ~3%. **Next lever: the histogram kernel.**

**End-to-end** (patch 128^3, batch 1, 4 classes, AMP, 60 steps):

| binning | reference step | v4 | v5 | peak GiB (all) | largest patch: reference / v4 / v5 |
|---|---|---|---|---|---|
| hard | 552 ms | 501 ms (1.10x) | 499 ms (1.11x) | 3.6 | 192^3 / **224^3** / **224^3** |
| soft | 631 ms | 512 ms (1.23x) | 510 ms (1.24x) | 3.6 | 224^3 / 224^3 / 224^3 |

- Inside the full training step the network dominates: the loss was ~10% (hard) /
  ~19% (soft) of the reference's step, so the step gets 1.1x / 1.24x faster. The
  gain grows with the loss's share (more classes, larger volumes, cheaper networks).
- Peak memory of the step is set by the network's activations (3.6 GiB at 128^3);
  near the limit the loss's memory matters: hard, the reference OOMs at 224^3 where
  v4/v5 fit (soft: all fit 224^3, all OOM 256^3).
- Loss curves overlap visually; max step-wise difference vs reference 1.5% (hard) /
  6.4% (soft); resolved by the control run below.

**Control run** (`results/kaggle/8f631653e11c-run-e2e/`, reference trained twice):

| binning | reference vs itself (max / mean) | v5_triton | v4_triton |
|---|---|---|---|
| hard | 2e-7 / 4e-8 (deterministic here) | 1.2% / 0.3% | 1.5% / 0.3% |
| soft | 5.7% / 2.1% (noise floor) | 4.4% / 1.4% | 9.8% / 5.6% |

- Step 0 (same weights, same batch) is identical for all runs (loss 1.46010, ACE
  term 0.853118, differences 8e-8): the forward pass is exact. Step 1 differs by
  ~1e-5 (fp-level gradient differences: summation order, ill-conditioned bins), and
  training amplifies them chaotically; each run then keeps its own trajectory.
- Fraction of steps a run's loss lies above the first reference run: v4 0.95,
  **reference rerun 0.95**, v5 0.87. The apparent "consistent offset" of v4 is shown
  by the reference's own rerun too: the first reference run took a slightly lower
  path; no bias. Quantifying the spread needs several seeds per implementation
  (planned for the full replication on the RTX PRO 6000).
- Step speedup 1.12x (hard) / 1.27x (soft); largest patch, hard: reference 208^3,
  v4 / v5 224^3; soft: all 224^3.
