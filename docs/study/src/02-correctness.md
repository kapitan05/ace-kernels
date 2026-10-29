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

## Tolerances: why these numbers

::code tests/helpers.py:12-21 | The four constants every comparison uses.

**Why not demand exact equality?** Floating-point addition is not associative:
`(a + b) + c` and `a + (b + c)` can differ in the last bit. Every implementation sums
the same values in a different order (a Python loop, one big scatter, tree sums inside
a Triton block, atomics in a random order), so two *correct* implementations differ
by rounding. fp32 has 24 significant bits: one rounding is ~6e-8 relative
(`eps32 / 2`), and a loss built from sums of thousands of values accumulates a few of
those. A tolerance must be **above** this legitimate noise and **far below** what
any real bug produces. Each constant was set by measuring both sides:

| constant | value | legitimate noise (measured) | why this value |
|---|---|---|---|
| `LOSS_RTOL` | 1e-5 relative | the reference vs the float64 oracle: ~1e-7..1e-6 on test sizes | ~10x headroom over fp32 noise; a 1e-4 change in one bin edge already exceeds it (below) |
| `LOSS_ATOL` | 1e-6 absolute | | for losses near 0 (a relative tolerance of a tiny number is meaningless) |
| `GRAD_RTOL` | 1e-4 relative (CPU) | ~2e-7 on test cases | gradients involve divisions by bin weights, so a little more room than the loss |
| `GRAD_ATOL` | 1e-7 absolute | | gradient entries that should be ~0 |
| `CUDA_GRAD_RTOL` | 1e-3 relative (GPU vs CPU golden) | **3.5e-4** worst case, measured on a T4 | soft binning with only ~3 voxels in a bin: its gradient is the difference of two nearly equal terms, which amplifies ~1e-7 rounding ~1000x; 1e-3 is the next round number above |
| `CUDA_GRAD_ATOL` | 1e-6 absolute | | as above, for near-zero entries |

**Too loose would hide bugs; too tight would fail correct code.** The check that the
constants are not too loose: deliberately broken versions (**mutants**) of the math
must fail them. The worked example below compares three correct implementations with
the float64 truth, then three subtle bugs (mutants of the formula):

::exec
import torch
from ace_kernels import ACEConfig, ace_loss
from ace_kernels.cases import InputSpec, make_inputs

config = ACEConfig(to_onehot_y=True)
x, y = make_inputs(InputSpec(2, 4, (4, 4, 4), seed=3), config)  # small: some bins empty

def loss_grad(impl, inp):
    leaf = inp.clone().requires_grad_(True)
    loss = ace_loss(leaf, y, config, impl=impl)
    loss.backward()
    return loss.detach().double(), leaf.grad.double()

L64, G64 = loss_grad("v2_recompute", x.double())  # float64 truth
print("correct implementations (fp32) vs float64 truth:")
for impl in ("reference", "v1_clean", "v2_recompute"):
    L, G = loss_grad(impl, x)
    print(f"  {impl:13s} loss {abs(L - L64).item() / L64.item():.1e}   "
          f"grad {((G - G64).abs().max() / G64.abs().max()).item():.1e}")

def ace(p, y1h, k=20, bug=None):  # the formula, written out, with optional bugs
    edges = torch.linspace(0, 1 + torch.finfo(torch.float32).eps, k + 1)[1:]
    if bug == "edges shifted by 1e-4":
        edges = edges + 1e-4
    idx = torch.bucketize(p, edges, right=(bug == "right=True instead of False"))
    f = []
    for b in range(p.shape[0]):
        for c in range(p.shape[1]):
            gaps = []
            for kk in range(k):
                m = idx[b, c] == kk
                if m.any():
                    gaps.append((p[b, c][m].mean() - y1h[b, c][m].mean()).abs())
                elif bug == "empty bins counted in K_ne":
                    gaps.append(torch.tensor(0.0))
            f.append(torch.stack(gaps).mean())
    return torch.stack(f).mean()

p = torch.softmax(x, 1).flatten(2)
y1h = torch.zeros_like(p).scatter_(1, y.flatten(2), 1.0)
ok = ace(p, y1h)
print("mutants (relative change of the loss; tolerance 1e-5):")
for bug in ("empty bins counted in K_ne", "edges shifted by 1e-4", "right=True instead of False"):
    print(f"  {bug:30s} {abs(ace(p, y1h, bug=bug) - ok).item() / ok.item():.1e}")
::end

The correct implementations differ from the truth by ~1e-8..1e-7; the first two
mutants change the loss by 5e-3..1e-1, 500-10000x the tolerance: it sits safely
between legitimate noise and real bugs. The third mutant changes
nothing on random inputs, because no random probability lands *exactly* on a bin
edge, where `right` matters. That is why the golden cases include inputs placed
exactly on the edges (`bin-edges-left`, `bin-edges-right`): **some bugs are invisible
to random testing and need constructed edge cases.**

Ties are the other special case. The loss contains `|mean p - mean y|`; where a bin's
difference is exactly 0, the derivative is undefined (a kink) and any value in
[-1, 1] is a valid subgradient. Eager PyTorch returns 0; a compiled kernel that
reorders float ops gets +-1e-9 and returns +-1/N. Both are right; the tests treat
these **tie cases** specially:

::code tests/helpers.py:35-52 | Cases with exact ties: the loss is checked exactly, the gradient only for finiteness (for compiled implementations).

## The measurement harness: `ace_bench/measure.py`

Unit tests use tensors of a few thousand voxels. The real question is whether an
implementation is right, fast and lean **at training size**, on the GPU. `measure.py`
answers that for one combination at a time: it turns *(implementation, binning,
shape, mode, CUDA graph or not)* into **one result row** (`Row`, page 7) with time,
bandwidth, launches, memory and accuracy. Everything in the benchmark tables of this
guide is a set of such rows.

How it fits together (one benchmark config -> many rows):

```
run.py      for each (shape, binning, implementation): a fresh worker process
              compute_reference(...)          once per group: reference + float64 truth
              for mode in (fwd, fwd_bwd), graph in (no, yes):
                  measure_case(...)  -> Row   build input, check, count, measure
              release_memory()               between rows
            -> rows.parquet, summary.md, plots
```

### Step 1: the trusted result for this input

::walk src/ace_bench/measure.py:151-194
@ def compute_reference( | Runs once per (shape, binning) group, on the exact input the timed rows will use (same seed, same distribution parameters, same storage dtype).
@ for impl in ("reference", FALLBACK_REFERENCE): | The upstream reference first; if it runs **out of memory** (soft binning at 288^3), fall back to `v2_recompute` (itself verified against the reference on all tests). The row records which one was used (`checked_against`).
@ try:  # float64 truth on the same (already rounded) input values | **The float64 truth**: the same input values (after fp32/fp16 rounding, so input rounding is not counted as error) through `v2_recompute` in float64 (~15 more significant digits than fp32). Then the reference's *own* error against this truth is stored: the yardstick for "as accurate as the original".
::end

### Step 2: one row

::walk src/ace_bench/measure.py::measure_case
@ def measure_case( | One row = one implementation x binning x shape x mode (`fwd` or `fwd_bwd`) x graph flag, measured on `env.device`.
@ spec = InputSpec( | **The input**: deterministic (seeded) logits and labels for this shape and distribution, stored in the configured dtype (fp32, or fp16/bf16 to test half-precision logits). `requires_grad_(True)` as in training: even a forward-only call then pays for saving tensors for backward.
@ input_bytes = x.numel() * x.element_size() | **Sizes and the ideal**: bytes of logits and target, and the *ideal* bytes a perfect implementation would move (page 7), from which the bandwidth floor follows.
@ row = Row( | The row starts with its identity; measurement fields are filled below and stay `None` if not measured (e.g. CPU rows have no GPU memory numbers).
@ if not get_impl(impl).supports(loss_config.binning, device): | Unsupported combinations (Triton on CPU) become rows with `status="unsupported"` instead of errors.
@ fn = _step_fn(impl, x, y, loss_config, mode) | **What is timed**: a closure calling the loss (`fwd`) or the loss plus `autograd.grad` to the input (`fwd_bwd`, what training does).
@ fn()  # first call: compilation / allocator warmup, untimed | One untimed call: Triton compilation and autotuning, `torch.compile`, cuDNN and allocator warm-up happen here.
@ if cuda_graph: | **Graph variant**: capture the call in a CUDA graph (page 1, Test 2). If the implementation does something a graph cannot record (a host-to-device copy, a sync), the row is `capture_failed`: that itself is a finding.
@ else: | **Saved-for-backward bytes** (eager rows): run once under the saved-tensor hooks (page 3) and record everything autograd keeps, minus the inputs.
@ if device == "cuda": | **Launches**: count GPU kernels and ATen ops per call with the profiler; `launch_floor_ms` = kernels x 5 us.
@ if not cuda_graph:  # replaying a graph allocates nothing: peak meaningless | **Peak memory**: the maximum extra GPU memory during one call, above what was allocated before it (graph replays reuse a pre-allocated pool, so the number is meaningless there).
@ _apply_timing(row, measure(fn, device, warmup=warmup, reps=reps), env) | **Timing** (page 1: cold L2, CUDA events, median of `reps`), then bandwidth and % of peak.
@ if reference is not None and not cuda_graph:  # untimed output check | **Accuracy**: after timing (untimed, so the extra work does not distort the numbers), compare loss and gradient with the reference and the float64 truth.
@ except torch.OutOfMemoryError as exc: | **Failures become data**: an out-of-memory is a result (`oom`: "the original does not fit at 288^3"); any other exception is an `error` row with its message, and the benchmark continues.
::end

### Step 3: the accuracy columns

::walk src/ace_bench/measure.py::_check_output
@ def _check_output( | Called for every eager row (unless `verify: false` in the config), on the same input as the timing.
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

## Why the original is less accurate than ours

### How it was found

1. **Sprint 7** (`5423f09`): every benchmark row started comparing its output with
   the reference at full size. Our kernels differed from the reference by up to ~1e-5
   in the loss and "0.5" in the max-norm gradient error at large volumes. With two
   fp32 results, you cannot tell *which one* is wrong.
2. **Between Sprints 7 and 8** (`8cecf92`, verify run `results/kaggle/47c938da465d-run-verify/`): a
   third, much more precise opinion was added: the float64 truth (Step 1 above). Result:
   **the reference was the one further from the truth** in the loss; our Triton
   versions were 10-100x closer. The max-norm gradient errors turned out to be large for
   *every* implementation, the reference included (the |x| kink, worked example above).
3. **Sprint 9** (`55b7d1a`, verify run `results/kaggle/55b7d1a680c0-run-verify/`): the
   L2 gradient error was added: all implementations have the same L2 error as the
   reference (~1e-4).
4. **Sprint 11** (realistic inputs, `results/kaggle/6ad66a182fc4-run-sprint11-precise/`):
   on trained-network-like inputs (peaked, mostly correct softmax) the reference's
   loss is **2e-3 off** the float64 value; ours 1e-7..6e-6.

### Why: how each implementation adds up a bin

The loss needs, per bin, the sum of the probabilities of its voxels. The original
uses `scatter_add` / `scatter_reduce`: on the GPU, every voxel's value is atomically
added to **one running total** per bin. Each addition rounds to fp32 against a total
that keeps growing, so the error grows with the number of voxels in the bin (like
adding pennies to a huge balance: eventually each penny is partly rounded away).

Our kernels first add BLOCK (128-512) values **inside a program as a tree** (pairs,
then pairs of pairs: only ~log2(BLOCK) = 8-9 roundings deep), and only the block's
partial sum is added to the running total: 256x fewer additions to the big number.

On peaked, realistic inputs ~70% of all voxels fall into bin 0 (probabilities near 0),
so that one bin holds millions of values: the worst case for the running total. The
same effect, reproduced on the CPU with 2M values of a typical bin 0:

::exec
import numpy as np
rng = np.random.default_rng(0)
p = rng.uniform(0, 0.05, 2_097_152).astype(np.float32)  # bin 0: many small probabilities
truth = p.astype(np.float64).sum()
running = np.add.accumulate(p, dtype=np.float32)[-1]          # one running total (atomics)
blocks = p.reshape(-1, 256).sum(axis=1, dtype=np.float32)    # tree sum per block of 256
ours = np.add.accumulate(blocks, dtype=np.float32)[-1]        # then one add per block
for name, v in (("one running total (original)", running), ("blocks of 256, then total (ours)", ours)):
    print(f"{name:34s} relative error {abs(float(v) - truth) / truth:.1e}")
::end

The relative error of a bin's mean probability then propagates into the loss: the loss
is the *difference* `mean p - mean y` per bin, often much smaller than either mean, so
a small relative error in `mean p` becomes a larger relative error in the loss (that
is how 1.4e-4 in a sum becomes 2e-3 in the loss on realistic inputs).

### Was it "fixed"?

The original is frozen (it is the ground truth for *what* is computed, not for *how
precisely*), so it was not changed. Ours is more accurate as a side effect of the
privatized histograms (page 4), and that is now **guarded**: a test requires the
Triton statistics to be within 1e-5 of the float64 truth and never meaningfully worse
than PyTorch's own fp32 result:

::walk tests/test_triton.py:69-91
@ def _assert_accurate( | Three numbers per check: the Triton result, PyTorch's fp32 result on the same bins, and the float64 truth.
@ scale = truth.abs().clamp_min(1.0) | Errors relative to the truth (at least 1, so near-zero bins compare absolutely).
@ assert err_triton <= 1e-5, (err_triton, err_torch) | **Absolute bar**: within 1e-5 of the truth. **Relative bar**: at most 2x PyTorch's fp32 error (at the noise level of ~1e-6 the two swap places between runs because of random atomic order: measured 1.56e-6 vs 1.53e-6, hence the factor 2, not 1).
@ def _assert_sum_p_accurate( | Applied to the per-bin probability sums of every Triton kernel test (hard and soft, all shapes, channels and bin counts).
::end

The benchmark harness re-checks the same at full size in every row (Step 3 above), and
the report flags any implementation whose error exceeds 2x the reference's own. Two
intermediate PyTorch steps were flagged that way (`v1_clean_compile`, `v2_recompute`,
soft, up to 7.5e-5 at 256^3: one `scatter_add` over all slices accumulates even more
into each running total); the final Triton versions are not.

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
::check Why can the original be 2e-3 off on realistic inputs while ours is 1e-6? | The original adds every voxel to one running fp32 total per bin (atomics); with millions of voxels in bin 0 the rounding accumulates. Ours adds blocks of 256 as trees first and then only the block totals: ~256x fewer additions to the big total.
::check What does measure_case do after timing, and why after? | The output check (loss and gradient vs the reference and float64 truth). After timing, so the extra work and memory do not distort the time and peak-memory numbers.
::check An implementation has max-norm gradient error 0.5 vs the reference. Is it wrong? | Not necessarily: look at the L2 error vs float64 and compare with the reference's own error vs float64. If both are equal (~1e-4), the 0.5 comes from sign flips in near-tie bins, which both fp32 implementations suffer.
