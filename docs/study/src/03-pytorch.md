# 3. PyTorch: vectorize, then write the backward by hand

Before writing GPU kernels, two questions are answered in plain PyTorch:
how much does removing the Python loop give (Sprint 3, v1), and how much memory does
a handwritten backward save (Sprint 4, v2)? Both versions also become **references for
the kernels**: v2's math is exactly what the Triton kernels compute.

## v1: one scatter for all slices instead of B x C loops

**Concept.** The loop over (batch, class) launches 3 scatters per slice. Instead, give
every (b, c) slice its own block of K bins in one flat buffer of size B*C*K: voxel
`i` of slice `r` goes to flat bin `r*K + bin(p_i)`. One scatter then handles all
slices at once.

::walk src/ace_kernels/impls/v1_vectorized.py:17-46
@ def _row_offsets(prep: Prepared, num_bins: int) -> torch.Tensor: | **The trick that removes the loop.** Number the B·C slices 0..B·C-1 (`[B, C, 1]` tensor) and multiply by K: slice r owns flat bins `r*K .. r*K+K-1` of one buffer of size B·C·K.
@ def _hard(prep: Prepared, config: ACEConfig) -> torch.Tensor: | `prep` (from `common.prepare`) holds the activated probabilities `[B, C, N]` and the one-hot target.
@ idx = torch.bucketize(p, hard_edges(k, p.device), right=config.right) | One `bucketize` for **all** slices at once (same edge table as the reference), then the slice offset is added: `flat[i]` is now a unique bin address across the whole batch.
@ counts = torch.zeros(size, device=p.device).scatter_add( | The reference's three scatters, but each **once** over the flattened batch instead of B·C times: far fewer launches. The atomic collisions are the same (still millions of voxels into K addresses per slice), so the speedup is only 1.2x: the evidence that launches were not the problem.
@ # copy before masking, like the reference's slice assignment: scatter_reduce | Empty bins -> NaN, as in the reference. The `.clone()` matters for autograd: `scatter_reduce` needs its output unchanged for its backward, and an in-place mask would corrupt it.
@ return (mean_p - mean_gt).abs().view(b, c, k) | Per-bin absolute gaps `[B, C, K]`; the shared `finalize` (nanmean, class mask, weights, reduction) turns them into the loss.
::end

`v1_clean` then removes the remaining waste: sums instead of `scatter_reduce(mean)`
(means are computed per bin afterwards), no one-hot target copy (class indices are
used directly), no host-device synchronization. The shared statistics function:

::walk src/ace_kernels/binning.py::hard_stats
@ idx = hard_bin_index(prep, config) | Bin of every voxel and channel, `[B, C, N]` int64, with the reference's edges and `right` convention.
@ flat = (idx + rows(prep) * k).flatten() | Same flat addressing as v1 (`rows` = slice number b·C + c).
@ zeros = p.new_zeros(b * c * k) | **Sums instead of means.** Three `scatter_add` into fresh zero buffers: count (adds a broadcast scalar 1, no ones-tensor allocated), sum of p. Means are computed later per bin (`sum / count`), which is cheaper than `scatter_reduce(mean)` and gives the per-bin sums the handwritten backward needs anyway.
@ if prep.onehot is not None: | Target sum. With a one-hot target: scatter it like p (C values per voxel, mostly zeros).
@ else: | With **class indices** (the paper's setting): each voxel has exactly one true class, so only that class's bin gets +1. `true_class` gives the voxel's class channel; `idx.gather` looks up the bin of the voxel *in that channel*. One value per voxel instead of C: no one-hot tensor is ever built.
@ shape = (b, c, k) | `BinStats(count, sum_p, sum_gt, empty)`: the whole forward state in four `[B, C, K]` tensors. Every later implementation (including the Triton kernels) produces exactly this.
::end

**Result.** Removing the loop alone: 1.2x. The cleanup: 2.6x. With `torch.compile`
on top: ~5x. Still under 5% of the hardware limit, because the scatters (atomics)
remain. This was the evidence that the next step had to be a custom kernel.

## v2: deriving the backward

**Concept.** Autograd stores whatever the forward computed per voxel (bin indices,
one-hot targets, gathered means) to replay it backwards: 3x the input (hard), 11x
(soft). But the loss only depends on per-bin sums, so the gradient can be written in
terms of tiny per-bin tensors. From `docs/derivation.md`:

    d_k = P_k / W_k - G_k / W_k                   (mean p minus mean y in bin k)
    df/dp_i = sum over bins j of voxel i:  s_kj / (K_ne W_kj) * [ w_ij + w'_ij (p_i - y_i - d_kj) ]

with `s_k = sign(d_k)`, `K_ne` = number of non-empty bins, `w` the bin weight of
the voxel and `w'` its derivative. For **hard** binning `w = 1, w' = 0`, so every
voxel in bin k gets the same gradient:

    df/dp_i = sign(d_k) / (K_ne * W_k)        =: coef_k

The backward therefore needs only `coef` and `d`: two `[B, C, K]` tensors (640 bytes
for B=2, C=4, K=20), plus the inputs that the caller holds anyway. It recomputes each
voxel's bin from `p` instead of storing it.

::walk src/ace_kernels/impls/v2_autograd.py::coefficients
@ d = bin_difference(stats) | `d_k = sum_p/W - sum_y/W` per bin (two divisions then subtract, exactly like the reference's two means, so the sign at exact ties matches), 0 for empty bins.
@ f = torch.nanmean(per_bin_error(stats), dim=-1) | Per-class loss `f = mean over non-empty bins of |d_k|` (empty bins are NaN and skipped), `[B, C]`.
@ nonempty = ~stats.empty | **K_ne**, the number of non-empty bins per (b, c), `[B, C, 1]` so it broadcasts over bins. `clamp_min(1)` avoids dividing by zero when every bin is empty (then all coef are 0 anyway).
@ safe_w = torch.where(nonempty, stats.weight, torch.ones_like(stats.weight)) | `coef_k = sign(d_k) / (K_ne · W_k)`: the derivative of `f` with respect to each voxel's probability in bin k (hard binning). Empty bins get 0 (no voxel can receive gradient from them). `safe_w` avoids 0/0 = NaN in the discarded branch of `where` (NaN would poison the gradient even when not selected).
@ return f, class_present(stats), coef, d | Class presence from the stats (`sum_k G_k > 0`, equal to `target.sum() > 0`): saves reading the target again.
::end

::walk src/ace_kernels/impls/v2_autograd.py::_df_dprobs
@ if config.binning == "hard": | **Hard**: every voxel gets the coefficient of its own bin. `hard_bin_index` recomputes the bins from p (nothing was stored), `gather` reads `coef[b, c, bin]` for each voxel: `[B, C, N]`.
@ p = prep.probs | **Soft**: recompute each voxel's two neighbouring bins, weights and span S (`soft_assign`, the same function as the forward).
@ resid = p - _true_target(prep)  # p_i - y_i | The derivation's formula per side j ∈ {left, right}: `coef_j · (w_j + w'_j (p - y - d_j))` with `w'_left = -1/S`, `w'_right = +1/S`. The `w'` term exists because moving p also moves the voxel's *weights* between the two bins.
::end

The `autograd.Function` ties it together; `mode` selects what is saved:

::walk src/ace_kernels/impls/v2_autograd.py::_ACELoss
@ class _ACELoss(torch.autograd.Function): | A custom `torch.autograd.Function`: we write `forward` and `backward` ourselves; autograd does not record the operations inside `forward`, so it stores nothing on its own.
@ prep = prepare(input, target, config, keep_labels=True) | Forward: activation, statistics, coefficients, loss. All voxel-sized intermediates are temporaries, freed when `forward` returns.
@ if mode == "triton_bwd":  # tiny saves only: coef and the reduction's jacobian | **What to keep for backward** (`ctx.save_for_backward`), by mode: `triton_bwd` (v4) and `recompute` (v2) save the inputs (which the caller holds anyway: no extra memory) plus the `[B, C, K]` coef / d and a `[B, C]` tensor. `grad_in_fwd` instead computes the full input gradient now (for an upstream gradient of 1) and saves only that.
@ def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[Any, ...]: | Backward receives `dL/dloss` (`grad_output`) and must return one gradient per forward input (None for target, config, mode, backend). `grad_in_fwd`: scale the precomputed gradient. `recompute`: rebuild everything from inputs + coef (`_input_grad`: `_df_dprobs`, upstream factor, activation backward).
::end

## Worked example: the formula vs autograd

The 8-voxel example from page 1, now with gradients. The handwritten formula, the
v2 implementation and autograd through the original code must agree:

::exec
import torch
from ace_kernels import ACEConfig, ace_loss

p = torch.tensor([[[0.05, 0.10, 0.30, 0.45, 0.60, 0.80, 0.90, 0.95]]])
y = torch.tensor([[[0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]]])
config = ACEConfig(num_bins=4, activation="none")

grads = {}
for impl in ("reference", "v2_recompute"):
    leaf = p.clone().requires_grad_(True)
    ace_loss(leaf, y, config, impl=impl).backward()
    grads[impl] = leaf.grad.flatten()

edges = torch.linspace(0, 1 + torch.finfo(torch.float32).eps, 5)[1:]
bins = torch.bucketize(p, edges).flatten()
k_ne = len(bins.unique())
by_hand = torch.empty(8)
for i, k in enumerate(bins.tolist()):
    m = bins == k
    d = p.flatten()[m].mean() - y.flatten()[m].mean()
    by_hand[i] = torch.sign(d) / (k_ne * m.sum())  # coef_k = sign(d_k) / (K_ne W_k)

print("bin        ", bins.tolist())
print("by hand    ", [round(v, 4) for v in by_hand.tolist()])
print("v2_recompute", [round(v, 4) for v in grads["v2_recompute"].tolist()])
print("reference  ", [round(v, 4) for v in grads["reference"].tolist()])
::end

Bin 3 is over-confident (gap +0.217), so its voxels get a positive gradient: gradient
descent lowers their probabilities. Bins 1 and 2 are under-confident and get pushed up.

## Measuring what autograd stores

The saved-tensor hooks of PyTorch see every tensor autograd keeps for backward. The
harness wraps a call and sums their sizes (excluding the inputs):

::walk src/ace_kernels/instrumentation.py::measure_saved_tensors
@ def measure_saved_tensors( | Runs `fn` (a loss call) and reports every tensor autograd saved for backward. `exclude`: tensors that do not count (the inputs, which exist anyway).
@ def pack(t: torch.Tensor) -> torch.Tensor: | `saved_tensors_hooks(pack, unpack)` calls `pack` for every tensor an operation saves. We record shape, dtype and storage size; storages are de-duplicated by data pointer (several saved views of one tensor count once).
@ # Return a detached alias (same storage), not ``t`` itself: for tensors an | **The leak bug.** Returning `t` itself kept a reference cycle through C++ autograd nodes that Python's garbage collector cannot see: each call leaked ~4x the input of GPU memory, and later benchmark groups ran out of memory (38 false OOM rows in Sprint 3). Returning `t.detach()` (same memory, no autograd link) fixed it.
@ with torch.autograd.graph.saved_tensors_hooks(pack, unpack): | The hooks are active only while `fn` runs.
::end

::exec
import torch
from ace_kernels import ACEConfig, ace_loss
from ace_kernels.cases import InputSpec, make_inputs
from ace_kernels.instrumentation import measure_saved_tensors

for binning in ("hard", "soft"):
    config = ACEConfig(binning=binning, to_onehot_y=True)
    x, y = make_inputs(InputSpec(2, 4, (32, 32, 32)), config)
    x.requires_grad_(True)
    for impl in ("reference", "v2_recompute"):
        _, stats = measure_saved_tensors(lambda: ace_loss(x, y, config, impl=impl), exclude=[x, y])
        print(f"{binning} {impl:13s} saved for backward: {stats.extra_bytes / x.nbytes:5.2f} x input")
::end

## Evidence

::plot docs/study/assets/figures/memory-ladder.png | Peak extra memory and bytes saved for backward per implementation, kits23 | from v2 on, nothing voxel-sized is saved (saved bytes ~0); peak memory drops further when the kernels stop materializing probabilities (v5).

::table results/kaggle/a2696e81824d-run-sprint4/results/*sprint4-memory*/rows.parquet | df.filter((pl.col("mode") == "fwd_bwd") & (pl.col("status") == "ok")).select("binning", "shape", "impl", pl.col("p50_ms").round(1).alias("ms"), (pl.col("saved_extra_bytes") / 2**20).round(1).alias("saved MiB"), (pl.col("peak_extra_bytes") / 2**20).round(0).alias("peak MiB")).sort("binning", "shape", ladder) | Sprint 4 memory run (T4): v2 saves ~0 bytes; at 288^3 soft the reference and v1 run out of memory, v2 does not.

## Try it

::run uv run pytest tests/test_v2.py -v | gradcheck in float64 against finite differences, equality with autograd, saved bytes = 2*B*C*K*4 + B*C
::run uv run pytest tests/test_saved_tensors.py -v | the saved-tensor measurement itself is tested

::check Why can the backward recompute bins instead of storing them, and what does it cost? | Bins depend only on p (and the fixed edge table), which the backward has anyway. The cost is reading p again and re-binning: cheap compared with storing and reading back int64 indices (8 bytes per voxel per class).
::check What is the gradient of a voxel in an empty bin, and in a bin with d_k = 0? | Empty bin: the voxel cannot be in it (empty means no voxels for hard binning); for soft, weights below empty_weight -> coef 0. d_k = 0: sign(0) = 0, so gradient 0, the same subgradient eager PyTorch's abs backward returns.
::check grad_in_fwd saves 1x the input. When is it better than recompute? | When backward speed matters more than memory: the backward becomes a multiply. It was not used later because the Triton backward (one pass, no stored gradient) gives both.
