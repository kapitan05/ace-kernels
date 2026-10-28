# 1. The loss, and why the original is slow

## Concept: what mL1-ACE measures

A segmentation network outputs, for every voxel and class, a probability `p`. A
model is **calibrated** when its confidence matches reality: of all voxels predicted
with probability ~0.8 for "kidney", ~80% really are kidney. The **average calibration
error (ACE)** measures the gap:

1. Split the probability range [0, 1] into K bins (K = 20 in the paper).
2. Put every voxel into the bin of its probability.
3. Per bin: compare the mean predicted probability with the fraction of voxels that
   really belong to the class (mean of the 0/1 target).
4. Average the absolute gaps over the non-empty bins: that is the loss for one
   (volume, class) pair. **mL1-ACE** then averages over classes (and the batch).

For one (b, c) pair with bins k = 1..K, voxel sets B_k, n_k = |B_k|:

    f_bc = mean over non-empty k of | (1/n_k) sum_{i in B_k} p_i  -  (1/n_k) sum_{i in B_k} y_i |

**Hard binning** puts each voxel in exactly one bin, as above. **Soft binning**
spreads each voxel over its two nearest bin *centres* with linear weights, so the loss
is smooth in `p` (gradients do not jump when a voxel crosses a bin edge). Everything
below comes in both variants.

## Worked example

Eight voxels, one class, K = 4 bins of width 0.25. The code below runs the original
loss on them (at build time) and recomputes it by hand:

::exec
import torch
from ace_kernels import ACEConfig, ace_loss

p = torch.tensor([[[0.05, 0.10, 0.30, 0.45, 0.60, 0.80, 0.90, 0.95]]])  # [B, C, N]
y = torch.tensor([[[0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]]])
config = ACEConfig(num_bins=4, activation="none")  # inputs are probabilities

edges = torch.linspace(0, 1 + torch.finfo(torch.float32).eps, 5)[1:]
bins = torch.bucketize(p, edges)
print("bin of each voxel:", bins.flatten().tolist())
gaps = []
for k in range(4):
    m = bins == k
    mp, my = p[m].mean().item(), y[m].mean().item()
    gaps.append(abs(mp - my))
    print(f"bin {k}: n={int(m.sum())}  mean p={mp:.3f}  mean y={my:.3f}  gap={gaps[-1]:.3f}")
print(f"by hand:          {sum(gaps) / len(gaps):.4f}")
print(f"reference (hard): {ace_loss(p, y, config, impl='reference').item():.4f}")
soft = config.model_copy(update={'binning': 'soft'})
print(f"reference (soft): {ace_loss(p, y, soft, impl='reference').item():.4f}")
::end

Bin 3 (p in [0.75, 1]) predicts 0.88 on average but only 2 of 3 voxels are positive:
the model is over-confident there, and the loss pushes those probabilities down.

## The original code

The upstream loss (`src/ace_reference/hardl1ace.py`, frozen and SHA-checked) computes
the per-bin statistics with a Python loop over batch x class, and three scatter
operations per slice:

::walk src/ace_reference/hardl1ace.py:70-105
@ batch_size, num_channels = input.shape[:2] | **Bin edges.** `boundaries` holds K+1 equally spaced values from 0 to `1 + eps32` (the tiny extra `eps32 = 1.19e-7` makes a probability of exactly 1.0 fall *inside* the last bin instead of past it). Only `boundaries[1:]` (the K upper edges) is used below. Note it is a **float32** tensor: its values are not exactly `k/K`, which is why every later implementation must use this same table to get bit-identical bins.
@ mean_p_per_bin = torch.zeros( | **Output buffers**, one row of K bins per (batch, class) pair: mean prediction, mean target and voxel count per bin. Shape `[B, C, K]`, tiny (640 values for B=2, C=4, K=20).
@ input = input.flatten(start_dim=2).float() | Spatial dimensions are flattened: a 3D volume `[B, C, X, Y, Z]` becomes `[B, C, N]` with N = X·Y·Z voxels. `.float()` copies the target to float32 (a full extra copy of the target tensor).
@ for b in range(batch_size): | **The Python loop** over every (b, c) pair: 2 x 4 = 8 iterations for kits23, 2 x 16 = 32 for amos22. Each iteration launches several GPU kernels, so the number of launches grows with B·C (428 per call for kits23 hard).
@ bin_idx = torch.bucketize( | `bucketize` returns, for each of the N probabilities of this slice, the index of its bin: the number of edges smaller than p (`right=False`). This allocates an **int64 tensor of N elements** (8 bytes per voxel), which autograd keeps for the backward.
@ bin_counts[b, c, :] = torch.zeros_like(boundaries[1:]).scatter_add( | **Counting.** `scatter_add` adds 1 at position `bin_idx[i]` for every voxel i. On the GPU, every voxel's thread does an *atomic* add into one of only K = 20 addresses: millions of colliding atomics. This is the contention that costs 84% of the original's GPU time.
@ mean_p_per_bin[b, c, :] = torch.empty_like(boundaries[1:]).scatter_reduce( | **Means per bin**, again by scattering all N voxels: `scatter_reduce(..., reduce="mean")` for the probabilities and once more for the targets. Two more full passes over the data, two more rounds of atomics. `include_self=False` ignores the uninitialized `empty_like` values.
@ # Remove nonsense bins: | Bins with no voxels get NaN means, so that the later `nanmean` skips them. This boolean-mask assignment is another pass over `[B, C, K]` (cheap) but uses a mask that autograd also has to track.
::end

Then the loss itself, from the per-bin means:

::walk src/ace_reference/hardl1ace.py:233-278
@ mean_p_per_bin, mean_gt_per_bin, bin_counts = hard_binned_calibration( | Calls the binning function above; the activation (softmax) and the one-hot conversion of the target already happened earlier in `forward`.
@ f = torch.nanmean(torch.abs(mean_p_per_bin - mean_gt_per_bin), dim=-1) | **The loss per (b, c)**: absolute gap per bin, averaged over the K bins while skipping NaN (= empty) bins. Result `[B, C]`. The gradient flows back through `abs` (the sign of the gap) and the means.
@ if self.ignore_empty_classes: | Classes that do not occur in this volume's target contribute 0 (otherwise an absent class would still be penalized). `target.sum(...)` reads the **whole** target tensor again just to find which classes are present; v4 later derives presence from the per-bin target sums instead, at no cost.
@ num_of_classes = target.shape[1] | **Class weights** (optional): validated, then multiplied into `f`. `torch.as_tensor` on a Python list creates a CPU tensor and `.to(f)` copies it to the GPU on every call: a host-to-device copy (harmless for speed, but it prevents CUDA-graph capture; our versions use Python scalars).
@ if self.reduction == LossReduction.MEAN.value: | **Reduction** over the `[B, C]` values: mean (the paper's setting), sum, or none (keeps a broadcastable `[B, C, 1, 1, 1]` shape).
::end

Autograd differentiates all of this automatically, which means it **stores**
intermediate per-voxel tensors (bin indices, one-hot targets, gathered values) for the
backward pass: 3x the input for hard binning, 11x for soft.

## Why it is slow: memory-bound work done the expensive way

The loss does very little arithmetic per voxel, so the ideal implementation is
**memory-bound**: its time is set by how fast the GPU can read the inputs. On a T4
(320 GB/s), reading the kits23 batch (2 x 4 x 128^3 float32 logits + labels) once for
the forward and twice for the backward takes ~0.8 ms. That is the **bandwidth floor**.
The original takes ~100 ms (hard) / ~180 ms (soft): 118x / 217x above the floor.

::plot docs/study/assets/figures/distance-to-limit.png | Measured time divided by the bandwidth floor, per implementation (1 = hardware limit) | the original is two orders of magnitude from the hardware limit; each sprint moves closer; the final kernels are 4-7x from it.

Where do 100 ms go? The profiler (Sprint 2) answered it, and the answer was not the
obvious one:

::plot docs/study/assets/figures/time-breakdown.png | GPU time by kernel type, kits23 hard, per implementation | in the original, 84% of the GPU time is scatter (atomic) kernels; removing the Python loop alone (v1) leaves the atomics and is only 1.2-2.5x faster.

**Atomic contention.** An atomic add makes a read-modify-write on one memory address
indivisible. When millions of GPU threads atomically add into the same 20 addresses,
they serialize: the hardware handles colliding updates one after another. The fix
(page 4) is to let each block of threads build a **private** histogram in its own
registers and publish it with one atomic per bin.

Other waste found in Sprint 2 (each fixed by a later sprint):

| inefficiency | cost | fixed in |
|---|---|---|
| B x C Python loop (428-3813 kernel launches per call) | launch overhead, small kernels | S3 (v1) |
| three scatter passes, each re-reading the data | 3x memory traffic | S3 (v1), S5 (v3) |
| atomic contention on K addresses | 84% of GPU time | S5 (v3) |
| per-voxel tensors saved for backward | 3x / 11x input memory | S4 (v2) |
| separate softmax pass (and again in backward) | 31% of v4's time | S8 (v5) |
| host-to-device copy in soft binning (`torch.tensor` in forward) | breaks CUDA graphs | S3 |

## How speed is measured (preview of page 7)

GPU work is asynchronous: Python returns before the GPU finishes, so wall-clock timing
is wrong. The harness records CUDA events around each call, synchronizes, flushes the
L2 cache before each repetition (so data is read from DRAM as in training) and reports
the median of 10+ repetitions:

::walk src/ace_bench/timing.py::measure
@ for _ in range(warmup): | **Warm-up calls are not timed.** The first calls compile Triton kernels, run the autotuner (it tries several block sizes), pick cuDNN algorithms and grow the memory allocator; timing them would measure setup, not the loss.
@ if device != "cuda": | On the CPU, a simple wall clock is correct: CPU work is synchronous.
@ flush = torch.empty(_L2_FLUSH_BYTES | **Why events and a flush.** A 256 MB buffer (larger than the T4's 4 MB L2 cache) and one pair of CUDA events per repetition are created *before* timing, so allocation is not measured.
@ torch.cuda.synchronize() | Wait until all previous GPU work has finished, so the first measured repetition starts from an idle GPU.
@ for start, end in zip(starts, ends, strict=True): | Each repetition: `flush.zero_()` writes 256 MB, evicting the loss's data from the L2 cache, so the call reads its inputs from DRAM as it would in training (where the network ran in between). `start.record()` / `end.record()` put timestamps **into the GPU's stream**: they measure GPU execution time, not the Python call (which returns before the GPU finishes).
@ torch.cuda.synchronize() | One synchronization at the end, then the elapsed time of each event pair. `_summarize` reports median (p50), p20, p80, mean and standard deviation; the tables in this guide use the median, which is robust to outliers.
::end

## Try it

::run uv run pytest tests/test_reference_frozen.py -v | the vendored upstream files are byte-identical to commit faa93c7 (SHA-256)
::run uv run python -m ace_bench.run bench/configs/quick_cpu.yaml | a small CPU benchmark of every implementation; prints one line per row and writes results/<timestamp>-quick-cpu/

## Evidence

::table results/kaggle/fb8ebdca9b33-run-sprint2/results/*baseline-paper/rows.parquet | df.filter((pl.col("mode") == "fwd_bwd") & ~pl.col("cuda_graph") & (pl.col("status") == "ok")).select("binning", "shape", "impl", pl.col("p50_ms").round(1).alias("ms"), pl.col("pct_peak_bw").round(2).alias("% peak BW"), pl.col("kernels_per_call").alias("kernels")).sort("binning", "shape", "impl") | Sprint 2 baseline (Tesla T4): the original and the off-the-shelf fixes (`torch.compile`, checkpointing). Compile helps soft binning a lot (4.7x) but stays far from the limit.

::check Why does removing the Python loop over B x C barely help (1.2x)? | The loop only adds kernel launches (~5 us each); the time is spent inside the scatter kernels, where millions of atomics collide on 20 addresses. Fewer, bigger scatters still collide.
::check Why does a CUDA graph not speed up the original hard loss? | A CUDA graph removes CPU launch overhead. The original is not launch-bound but GPU-bound (atomic contention), so there is nothing for the graph to remove.
::check What is the bandwidth floor of kits23 forward + backward on a T4? | Ideal bytes = 2 x (input + target) + input (read twice, write the gradient once) ~ 268 MB; 268 MB / 320 GB/s ~ 0.84 ms.
