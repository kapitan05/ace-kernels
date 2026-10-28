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

::code src/ace_kernels/impls/v1_vectorized.py::_hard | `v1_vectorized`: the reference's three scatters, but over all slices at once via `_row_offsets` (row * K). Same operations otherwise, so it isolates the effect of the loop.

`v1_clean` then removes the remaining waste: sums instead of `scatter_reduce(mean)`
(means are computed per bin afterwards), no one-hot target copy (class indices are
used directly), no host-device synchronization. The shared statistics function:

::code src/ace_kernels/binning.py::hard_stats | Three `scatter_add` into `[B*C*K]`: counts, sum of p, sum of y. For index targets, the target sum only scatters the *true* class of each voxel (1 value per voxel instead of C).

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

::code src/ace_kernels/impls/v2_autograd.py::coefficients | The whole per-bin "state" of the backward: `d`, the per-class error `f`, presence, and `coef = sign(d) / (K_ne W)` (0 for empty bins).

::code src/ace_kernels/impls/v2_autograd.py::_df_dprobs | Backward, per voxel: hard = gather `coef` at the voxel's bin; soft = the two-sided formula with `w' = -1/S, +1/S`.

The `autograd.Function` ties it together; `mode` selects what is saved:

::code src/ace_kernels/impls/v2_autograd.py::_ACELoss | `recompute` saves inputs + `coef`, `d`, presence; `grad_in_fwd` computes the full gradient already in the forward and only scales it in backward (the Liger-kernel trick: 1x input memory, no second pass).

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

::code src/ace_kernels/instrumentation.py::measure_saved_tensors | The pack hook returns `t.detach()`: returning `t` itself created a reference cycle that leaked GPU memory (a Sprint 3 harness bug that caused 38 out-of-memory rows; see page 7).

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
