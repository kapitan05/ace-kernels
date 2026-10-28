# 2. Proving "same as the original"

A faster loss is worthless if it computes something else. This page shows the layers
of evidence that every implementation returns the **same loss and the same gradient**
as the upstream code, and how "same" is defined for floating point.

## Concept: five independent layers

| layer | question it answers | where |
|---|---|---|
| frozen reference | is our ground truth really the upstream code? | `tests/test_reference_frozen.py` |
| independent oracle | does the upstream code compute the paper's formula? | `tests/oracle.py`, `test_oracle.py` |
| golden files + 43 cases | does every implementation match the reference on curated edge cases? | `tests/golden/`, `test_golden.py` |
| random equivalence | ... and on random shapes, classes, bins, options? | `test_equivalence.py` (hypothesis) |
| full-size output check | ... at real training sizes on the GPU, vs a float64 truth? | `ace_bench/measure.py` |

Plus targeted tests per technique: `gradcheck` of the handwritten backward
(`test_v2.py`), exact bin counts and kernel-vs-PyTorch statistics (`test_triton.py`),
the drop-in module against the upstream classes (`test_module.py`), no host syncs and
CUDA-graph capture (`test_sync.py`), memory saved for backward (`test_saved_tensors.py`).
**No benchmark of an implementation runs unless all of these pass** (the Kaggle job's
first step is the full test suite).

## Golden files: one test, every implementation

The reference's outputs (loss and input gradient) for the 43 named cases are stored
once in `tests/golden/`. One parametrized test runs every registered implementation on
every case and device:

::walk tests/test_golden.py::test_matches_golden
@ @pytest.mark.parametrize("device", DEVICES) | **Three parametrizations multiply**: every device (CPU, and CUDA when present) x every golden case (43) x every registered implementation (18). pytest generates one test per combination: ~1500 tests from 15 lines. `available_impls()` reads the registry, so a new implementation is covered automatically.
@ record = GOLDEN[case_name] | A golden record holds the exact input tensor, target, config, and the reference's loss and input gradient, computed once by `scripts/make_golden.py` and stored in `tests/golden/reference_cases.pt` (so the test does not depend on the reference running correctly today).
@ skip_unless_supported(impl, config.binning, device, case_name) | Skips combinations an implementation does not support (e.g. Triton versions on the CPU), and limits `slow` implementations (`torch.compile` recompiles for every new shape) to a few smoke cases.
@ result = run(impl, x, y, config) | `run` (in `tests/helpers.py`) makes a fresh leaf tensor, computes the loss, backpropagates `loss.sum()` and returns loss + input gradient.
@ if get_impl(impl).slow and case_name in TIE_CASES: | Tie cases (exact `mean p == mean y` in a bin) have a legitimately different subgradient under compiled code: only the loss is checked exactly there, the gradient for finiteness.
@ assert_same(result, expected) | Loss and gradient compared with the documented tolerances (loss 1e-5 relative; gradient 1e-4 CPU / 1e-3 CUDA).
::end

The cases are the edge conditions where an implementation is most likely to differ:

::code src/ace_kernels/cases.py::_cases | Empty classes (ignored or kept), no background, class weights, all reductions, sigmoid, one bin, fifty bins, probabilities exactly on bin edges (both `right` modes), probabilities exactly 0/1, saturated logits, one-hot vs index targets.

## The independent oracle

Golden files only prove "same as the reference". The oracle re-implements the paper's
formula independently, in float64 NumPy, voxel by voxel, so a bug shared by the
reference and our code would still be caught:

::walk tests/oracle.py:31-96
@ def _hard_weights(p: F64, num_bins: int, right: bool) -> F64: | **Hard weights** as a dense `[N, K]` matrix in float64 NumPy: 1 in the column of each voxel's bin. `np.searchsorted` plays the role of `torch.bucketize`; the edge table is the reference's float32 linspace converted to float64, so edge voxels fall into the same bin.
@ def _soft_weights(p: F64, num_bins: int) -> F64: | **Soft weights** written straight from the definition, one voxel at a time: below the first centre or above the last, all weight on the end bin; otherwise linear interpolation `1 - t` / `t` between the two neighbouring centres. Slow and obviously correct, which is the point of an oracle.
@ def _slice_error(p: F64, y: F64, config: ACEConfig) -> float: | **The formula itself** for one (b, c) slice: per non-empty bin k, `|w_k · p - w_k · y| / W_k` (weighted means), then the mean over non-empty bins. Matrix-vector products in float64: no summation-order issues at test sizes.
@ def oracle_ace(input: F64, target: F64, config: ACEConfig) -> F64: | **Activation** in float64 (numerically stable softmax: subtract the max), optional background removal, then the slice error for every (b, c).
@ if config.ignore_empty_classes: | Empty-class mask, class weights and reduction, mirroring the options of the upstream class. `test_oracle.py` asserts the *reference* matches this on all 43 cases: the ground truth is itself verified.
::end

## Tolerances: what "same" means in floating point

Two correct implementations that sum in a different order differ in the last bits.
The tolerances are chosen from measurement, not guessed, and documented in place:

::code tests/helpers.py:12-21 | Loss: relative 1e-5. Gradient: 1e-4 on CPU, 1e-3 on CUDA (few voxels per bin amplify rounding; mutants were off by >= 1%).

One subtle case: the loss contains `|mean p - mean y|`. Where a bin's difference is
exactly 0, the derivative is undefined (a kink) and any value in [-1, 1] is a valid
subgradient. Eager PyTorch returns 0; a compiled kernel that reorders float ops gets
+-1e-9 and returns +-1/N. Both are right; the tests treat these **tie cases** specially:

::code tests/helpers.py:35-52 | Cases with exact ties: the loss is checked exactly, the gradient only for finiteness (for compiled implementations).

## Full-size check against a float64 truth

Unit tests use small tensors. In every benchmark row, the harness also checks the
implementation at the real size against the reference **and** against a float64
computation (the handwritten-backward PyTorch version in float64):

Before measuring an implementation, the harness computes the reference's result **and**
a float64 truth on the same benchmark input:

::walk src/ace_bench/measure.py:151-194
@ def compute_reference( | Runs once per (shape, binning) group, on the exact input the timed rows will use (same seed, same distribution parameters, same storage dtype).
@ for impl in ("reference", FALLBACK_REFERENCE): | The upstream reference first; if it runs **out of memory** (soft binning at 288^3), fall back to `v2_recompute` (itself verified against the reference on all tests). The row records which one was used (`checked_against`).
@ try:  # float64 truth on the same (already rounded) input values | **The float64 truth**: the same input values (after fp32/fp16 rounding, so input rounding is not counted as error) through `v2_recompute` in float64. Then the reference's *own* error against this truth is stored: the yardstick for "as accurate as the original".
::end

Then, for every implementation and row:

::walk src/ace_bench/measure.py::_check_output
@ def _check_output( | Called for every benchmark row (unless `verify: false` in the config), on the same input as the timing.
@ row.checked_against = ref.impl | Copies the reference's own errors vs truth into the row, so every row can be judged on its own.
@ if impl == ref.impl: | Loss and gradient of the implementation (reused if it *is* the reference).
@ row.loss_rel_err, row.grad_rel_err = _rel(loss, ref.loss), _rel(grad, ref.grad) | **vs the reference**: `_rel` = max absolute difference / max |reference| (max-norm, diagnostic); `_rel_l2` = ‖a - b‖ / ‖b‖ over the whole gradient (the robust measure).
@ if ref.truth_loss is not None and ref.truth_grad is not None: | **vs float64 truth**: the same three errors. The report flags a row when its loss or L2 gradient error is more than 2x the reference's own error vs truth.
::end

## Worked example: why max-norm gradient errors mislead

At full size, the max-norm gradient error (largest relative difference of any single
element) of correct implementations can reach 0.5. The reason is the |x| kink: in a
bin whose mean p and mean y almost cancel, the *sign* of the difference decides the
gradient of all its voxels, and the sign depends on summation order:

::exec
import torch
g = torch.Generator().manual_seed(5)
p = torch.rand(200_000, generator=g, dtype=torch.float64)  # one bin's probabilities
y = (torch.rand(200_000, generator=g, dtype=torch.float64) < p).double()  # 0/1 targets
p = p - (p.mean() - y.mean()) + 1e-8  # true gap: +1e-8, below fp32 rounding
pf, yf = p.float(), y.float()
d64 = p.mean() - y.mean()
tree = pf.sum() / pf.numel() - yf.sum() / yf.numel()  # GPU-style tree sum
seq = pf.cumsum(0)[-1] / pf.numel() - yf.cumsum(0)[-1] / yf.numel()  # sequential
for name, d in (("float64", d64), ("fp32 tree sum", tree), ("fp32 sequential", seq)):
    print(f"{name:16s} gap {d.item():+.2e} -> d|gap|/dp sign {torch.sign(d).item():+.0f}")
::end

Same data, two correct fp32 summation orders: one says the gap is positive (every
voxel in the bin gets gradient +1/n), the other says exactly 0 (subgradient 0). The
gradient of every voxel in such a bin differs completely between two correct
implementations. The **relative L2 error** of the whole gradient (one number over all
voxels) is the meaningful accuracy measure; the max-norm stays as a diagnostic.

## Evidence

::table results/kaggle/55b7d1a680c0-run-verify/results/*verify-truth/rows.parquet | df.filter(pl.col("shape").is_in(["kits23", "cube256"])).select("binning", "shape", "impl", pl.col("loss_err_truth").alias("loss vs fp64"), pl.col("grad_l2_err_truth").alias("L2 grad vs fp64"), pl.col("ref_grad_l2_err_truth").alias("reference L2 vs fp64"), pl.col("grad_rel_err").alias("max-norm vs ref")).sort("binning", "shape", ladder) | Accuracy at full size against float64 (Tesla T4). All implementations have the same L2 gradient error as the original; the Triton versions have a 10-100x smaller loss error; the max-norm column is large for everyone, including the original's own float64 comparison.

::plot docs/study/assets/figures/accuracy.png | Loss error vs float64, noise inputs and trained-network-like inputs | on realistic peaked inputs the original's fp32 accumulation into one dominant bin is 2e-3 off; block-wise sums (ours) stay at 1e-7..6e-6.

## Try it: the mutation drill

A test suite is only as good as the bugs it catches. Break the math on purpose:

1. In `src/ace_kernels/binning.py`, function `bin_difference`, change
   `stats.sum_p / safe - stats.sum_gt / safe` to `stats.sum_p / safe + stats.sum_gt / safe`.
2. Run the suite and watch which tests fail (golden, oracle, equivalence, v2...).
3. Restore the file.

::run uv run pytest tests/ -q -x | before the mutation: all pass (CUDA tests skip on a Mac)
::run git checkout src/ace_kernels/binning.py | restore after the drill

::check Golden files prove "same as the reference". What does the oracle add? | Independence: the reference could itself be wrong (or both could share a bug, e.g. if we had copied its code). The oracle is written from the paper's formula in float64, so it validates the ground truth.
::check Why is a gradient tolerance of 1e-3 on CUDA acceptable and not hiding bugs? | It was set from measurement (worst legitimate case 3.5e-4: few voxels per bin amplify rounding) and checked against mutants, which were off by >= 1%. A real bug moves gradients by far more than 1e-3.
::check An implementation has max-norm gradient error 0.5 vs the reference. Is it wrong? | Not necessarily: look at the L2 error vs float64 and compare with the reference's own error vs float64. If both are equal (~1e-4), the 0.5 comes from sign flips in near-tie bins, which both fp32 implementations suffer.
