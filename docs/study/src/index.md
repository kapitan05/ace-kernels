# Study guide: making the mL1-ACE loss fast on the GPU

This guide teaches the whole project: the loss, why the original implementation is
slow, every optimization step (sprints 1-11), how each step is **proven** correct and
measured, and the tooling around it. Every code block on these pages is cut from the
repository at build time, and every table is read from a saved benchmark run, so
what you read is what the code does.

> **Result in one line.** Our implementation computes the same loss and gradients as
> the original, 29-75x faster and with 5-14x less peak memory on a Tesla T4 (paper
> training shapes), and is a drop-in replacement:
> `from ace_kernels.module import HardL1ACELoss, SoftL1ACELoss`.

## How to use this guide

Read the pages in order; each one follows the same pattern:

1. **Concept**: the idea in plain words.
2. **Code**: the real implementation, with the lines that matter pointed out.
3. **Worked example**: small numbers computed by actually running the code.
4. **Try it**: commands for your Mac (tests, CPU runs). Copy button on the right.
5. **Evidence**: the plot or table from a GPU run that proves the claim.
6. **Self-check**: questions with hidden answers. Answer before you open them.

GPU numbers come from Kaggle Tesla T4 runs stored in `results/kaggle/<commit>-<job>/`;
you do not need a GPU to study them. Rebuild this guide any time:

::run uv run python -m ace_bench.figures && uv run python -m ace_bench.study | writes docs/study/*.html; then open docs/study/index.html

## Map of the repository

::svg architecture

Five parts, each with one job:

| part | job | read first |
|---|---|---|
| `src/ace_reference/` | the upstream loss, frozen: the ground truth | `hardl1ace.py` |
| `src/ace_kernels/` core | options, shared semantics, the math in PyTorch | `config.py`, `binning.py` |
| `src/ace_kernels/impls/` | the optimization ladder, one change per entry | page 3-5 |
| `src/ace_kernels/kernels/` | Triton GPU kernels | page 4-5 |
| `src/ace_bench/` + `tests/` | measuring and proving | page 2, 7 |

Every implementation is registered under a name and called the same way, which is
what makes the ladder comparable step by step:

::code src/ace_kernels/registry.py::ace_loss | One entry point for all implementations: same inputs, same config, only `impl` changes. Tests and benchmarks loop over the registry.

## The deliverable and its data flow

::svg dataflow

The user-facing class only translates the upstream constructor into an `ACEConfig`
and picks the implementation for the device:

::code src/ace_kernels/module.py::_ACELoss.resolve_impl | "auto": Triton on CUDA, the handwritten-backward PyTorch version elsewhere (CPU, Mac).

## The ladder at a glance

Each row is one registry entry = one change, measured on the same GPU and input
(kits23 shape: 2 volumes x 4 classes x 128^3, forward + backward):

::table docs/ablation/ablation_rows.parquet | df.filter((pl.col("shape") == "kits23") & (pl.col("mode") == "fwd_bwd") & (pl.col("binning") == "hard")).select("impl", pl.col("p50_ms").round(2).alias("ms"), pl.col("pct_peak_bw").round(1).alias("% peak BW"), (pl.col("peak_extra_bytes") / 2**20).round(0).alias("peak MiB"), pl.col("kernels_per_call").alias("kernels"), pl.col("run").alias("run")).sort(ladder) | Hard binning, kits23, Tesla T4, in sprint order (latest valid run per implementation; all shapes: `docs/ablation/ablation.md`). Caveat: `v6_index` and `v6_dot` come from the Sprint 11 paper run, where the first shape (kits23) ran ~1.3x slow for every Triton version; on the other shapes `v6_index` is 1.05-1.15x faster than `v5_triton`.

::plot docs/ablation/ladder-hard.png | The ladder: time (log), % of peak bandwidth, peak memory per step | each step's gain is attributable to one change (one registry entry each).

## Sessions

| page | sprints | you will understand |
|---|---|---|
| [1. The loss](01-loss.html) | S1-S2 | the formula, hard vs soft binning, memory-bound, atomic contention |
| [2. Correctness](02-correctness.html) | S1, S9 | how "identical to the original" is proven, tolerances, float64 truth |
| [3. PyTorch](03-pytorch.html) | S3-S4 | vectorizing, deriving the backward by hand, recompute vs store |
| [4. Triton](04-triton.html) | S5-S8 | GPU kernels: privatized histograms, fused backward, fused softmax |
| [5. Sprint 11](05-sprint11.html) | S11 | launch overhead, O(1) bin index, packaging, failed experiments |
| [6. Training](06-training.html) | S8, S11 | the loss inside training, Amdahl, `torch.compile` vs our kernels |
| [7. Tooling](07-tooling.html) | all | benchmark harness, Kaggle CLI, W&B, profiler, git tags |
| [Evidence](evidence.html) | all | every claim and the test / plot that proves it |
| [Glossary](glossary.html) | all | the terms |

::check Why is each optimization a separate registry entry instead of one improved implementation? | So every speedup (or slowdown) is attributable to exactly one change: the ablation table compares neighbours that differ in one idea. It also keeps the earlier versions runnable as baselines and as test oracles for the later ones.
