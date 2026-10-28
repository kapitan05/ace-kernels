# Evidence: every claim and what proves it

Use this page to defend the work: each claim of the report's summary, the **tests**
that would fail if it were false, and the **measurement** that shows it. Run folders
are under `results/kaggle/` (commit hash first); W&B project `ace-kernels` mirrors them.

## Correctness and accuracy

| claim | proven by (tests) | shown by (measurement) | learn on |
|---|---|---|---|
| same loss and gradient as the original | `test_golden` (43 cases x all impls x CPU/CUDA), `test_equivalence` (random), `test_oracle` (float64, independent) | output-check columns in every `rows.parquet` | [2](02-correctness.html) |
| the handwritten backward is the true derivative | `test_v2::test_gradcheck_float64`, `test_matches_autograd_in_float64` | page 3 worked example | [3](03-pytorch.html) |
| Triton statistics = PyTorch statistics, exact bin counts | `test_triton::test_hard_stats_match_torch`, `test_soft_stats_match_torch` | | [4](04-triton.html) |
| O(1) index = `torch.bucketize` at every edge | `test_triton::test_direct_index_exact` (8 ulp around every edge, K = 1..100) | page 5 worked example | [5](05-sprint11.html) |
| drop-in: same API and results as the upstream classes | `test_module` (signature, 43 cases, other_act, weights) | | [5](05-sprint11.html) |
| as accurate as the original (gradient), more accurate (loss) | `test_triton::_assert_accurate` | verify runs `55b7d1a680c0-run-verify`, `60a3c6df2c3c-run-sprint11/*verify`; accuracy figure | [2](02-correctness.html) |
| original is 2e-3 off on realistic inputs | | `6ad66a182fc4-run-sprint11-precise` (kits23-trained rows) | [2](02-correctness.html) |
| precise exp removes the near-edge binning differences | `test_triton::test_v6_precise_exp_matches_torch_softmax` | precise run: L2 2.9e-4 -> 6.4e-5 | [5](05-sprint11.html) |

## Speed

| claim | proven by | shown by | learn on |
|---|---|---|---|
| 29-75x faster (fwd + bwd, paper shapes) | gate passed in the same job | `docs/ablation/ablation.md`, ladder plots, `60a3c6df2c3c-run-sprint11/*paper` | [index](index.html) |
| the original's cost is atomic contention, not the loop | v1 rows: loop removal only 1.2x | time-breakdown figure, Sprint 2 profiles | [1](01-loss.html) |
| 13-28% of peak bandwidth; 3.5-8x from the floor | | distance-to-limit figure (`pct_peak_bw`, `bw_floor_ms` columns) | [1](01-loss.html) |
| 7 launches per call; small calls 17-26x faster | `test_triton::test_v6_launches` | deep-supervision table and figure, `kernels_per_call` | [5](05-sprint11.html) |
| insensitive to the input distribution | | `60a3c6df2c3c-run-sprint11/*realistic` | [5](05-sprint11.html) |

## Memory

| claim | proven by | shown by | learn on |
|---|---|---|---|
| nothing voxel-sized saved for backward | `test_v2::test_recompute_saves_only_per_bin_stats`, `test_triton::test_v5_saves_only_per_bin_tensors` | `saved_extra_bytes` column; memory figure | [3](03-pytorch.html) |
| 5-14x less peak memory | | `peak_extra_bytes` column; memory figure | [3](03-pytorch.html) |
| fits where the original runs out of memory | | Sprint 4/7 memory runs (288^3), e2e probes (224^3 vs 208^3) | [4](04-triton.html), [6](06-training.html) |

## Training and integration

| claim | proven by | shown by | learn on |
|---|---|---|---|
| training curves unchanged (within run-to-run noise) | `test_bench` e2e CPU pipeline test | e2e curves with control rerun (`8f631653e11c-run-e2e`, `60a3c6df2c3c-run-sprint11/*e2e`) | [6](06-training.html) |
| step 1.10-1.23x eager, 1.24-1.38x with compile | | training-step figure | [6](06-training.html) |
| no host syncs; CUDA-graph capturable | `test_sync::test_has_no_host_sync`, `test_captures_in_cuda_graph` | `+graph` rows in benchmarks | [5](05-sprint11.html) |
| compiles without graph breaks | `test_triton::test_v6_compiles_fullgraph` | e2e `compile.parquet` | [6](06-training.html) |

## Negative results (also claims)

| claim | shown by |
|---|---|
| tensor-core reduction is 4-8x slower on a T4 | `v6_dot` rows in the ablation; Sprint 11 dev runs |
| `torch.compile` alone: <= 5x, and 2x slower for soft at 16 classes | Sprint 2 baseline rows (`reference_compile`) |
| `torch.utils.checkpoint` saves stored memory but not peak, costs 1.6x | Sprint 2 rows (`reference_checkpoint`) |
| the original ACE loss does not break `torch.compile` (MONAI DiceLoss does) | e2e `compile.parquet`; page 6 |

## Where the evidence is not yet complete

- Hardware counters (`ncu`: DRAM bytes, occupancy) are blocked on Kaggle: the
  bandwidth numbers are computed from ideal bytes, not measured bytes.
- Calibration metrics after full training (ECE/ACE/MCE, reliability diagrams) are the
  planned paper replication on the RTX PRO 6000 (Sprint 10).
- The ~15 ms gap between isolated and in-step timing of the compiled original soft
  loss (page 6) is unexplained.
