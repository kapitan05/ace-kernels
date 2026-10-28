# 5. Sprint 11: launches, O(1) index, packaging, and what failed

Sprint 11 closed the gaps found in the Sprint 9 review. It is also the best example of
the method: every change was a **hypothesis** written down before the run, and three
of six were refuted. Refuted hypotheses are results too.

| hypothesis | result | verdict |
|---|---|---|
| O(1) bin index makes the histogram kernel 1.3-2.5x faster | loss 1.05-1.2x faster | partly: binning was not the expensive part |
| tensor-core reduction: 1.5-2.5x | 4-8x **slower** on the T4 | refuted (no Triton tensor-core path on sm_75) |
| fused epilogue: ~55 -> <= 8 launches, small volumes 2-5x | 7 launches, 32^3 17x, 16^3 26x | confirmed, stronger |
| precise exp fixes the brats21 gradient error for <= 5% | error = reference's, +0-6% | confirmed: default |
| reference slower on realistic inputs, ours insensitive | reference +17-60%, ours within 3% | confirmed |
| the reference breaks `torch.compile` of the step | the 8 breaks are MONAI DiceLoss, for every loss | refuted for the reference; v6 adds 0 |

## Launch overhead and the fused epilogue

**Concept.** Launching a GPU kernel costs ~5 us of CPU time. v5 did its real work in 2
kernels but needed ~50 more tiny PyTorch ops on `[B, C, K]` tensors (means, signs,
nanmean, class mask, weights, reduction, and an autograd call for the reduction's
jacobian). On a 128^3 volume that hides behind GPU work; on the small deep-supervision
outputs (32^3, 16^3) the call is **launch-bound**: its time is the launches.

::plot docs/study/assets/figures/scaling.png | Time vs volume, log-log, with each implementation's launch floor (kernels x 5 us) | v5 is flat at ~0.9 ms from 16^3 to 64^3 (launch-bound); v6 follows the data size down to 0.03 ms.

The fix: one Triton program computes the whole epilogue from the histogram:

::code src/ace_kernels/kernels/epilogue.py::_epilogue_kernel | All (b, c) rows x K bins in one tile: d, empty bins, K_ne, f (nanmean), coef, presence, class weights, mean/sum reduction and the jacobian. Same formulas and operation order as `coefficients` + `reduce_classes` (IEEE division).

::table results/kaggle/60a3c6df2c3c-run-sprint11/results/*sprint11-deepsup/rows.parquet | df.filter((pl.col("binning") == "hard") & (pl.col("status") == "ok")).with_columns(pl.when(pl.col("cuda_graph")).then(pl.col("impl") + " + graph").otherwise(pl.col("impl")).alias("impl")).pivot(on="impl", index="shape", values="p50_ms") | Deep-supervision levels, hard, ms per fwd+bwd (T4). v6 needs no CUDA graph; v5 needed one at small sizes.

::plot docs/study/assets/figures/deep-supervision.png | Per-level time for reference / v5 / v5 + CUDA graph / v6 | summed over the 4 levels of one training step: hard 62 -> 1.9 ms, soft 118 -> 2.8 ms.

## O(1) bin index, exactly

**Concept.** The kernels found a bin by comparing p against all K edges (20 compares,
40 for soft). The edges are almost `j / K`, so `floor(p * K)` is within one of the right
bin. Two correction steps against the **same float32 edge table** make it exact:

::code src/ace_kernels/kernels/bins.py::sorted_count | Guess `floor(p * n)`, then move up if the next edge is still counted, down if the current one is not. NaN -> 0 and +inf -> n, like the compare loop.

The same logic on the CPU, checked against `torch.bucketize` on every float32 within
8 ulp of every edge (the test on the GPU does the same for K = 1..100):

::exec
import torch

def sorted_count(p, table, right=False):
    n = table.numel()
    g = torch.clamp(p * n, 0, n).nan_to_num(0).to(torch.int64)
    hit = (lambda t: p >= t) if right else (lambda t: p > t)
    for _ in range(2):
        g = g + ((g < n) & hit(table[torch.clamp(g, max=n - 1)])).long()
        g = g - ((g > 0) & ~hit(table[torch.clamp(g - 1, min=0)])).long()
    return g

bad = 0
for k in (1, 3, 7, 20, 33, 100):
    table = torch.linspace(0, 1 + torch.finfo(torch.float32).eps, k + 1)[1:]
    vals = [table]
    up, down = table.clone(), table.clone()
    for _ in range(8):
        up, down = torch.nextafter(up, torch.tensor(2.0)), torch.nextafter(down, torch.tensor(-1.0))
        vals += [up, down]
    p = torch.cat(vals + [torch.rand(100_000)])
    for right in (False, True):
        bad += int((sorted_count(p, table, right) != torch.bucketize(p, table, right=right)).sum())
print("mismatches vs torch.bucketize:", bad)
::end

Result on the GPU: the loss got only 1.05-1.2x faster. The profile explains it: the
histogram kernel's cost is the `[BLOCK, K]` reduction (K compares + 3 multiply-sums per
voxel), not the index. That reduction is the remaining gap to the bandwidth limit.

## The failed experiment: tensor cores

**Idea.** The reduction is a matrix product: `onehot[K, BLOCK] @ values[BLOCK, 3]`.
Tensor cores do matrix products in fp16 very fast. To stay exact, every fp32 value is
split into three fp16 parts (`a + b/2^11 + c/2^22`); a 0/1 one-hot matrix is exact in fp16.

::code src/ace_kernels/kernels/bins.py::split3 | Exact fp16 split of an fp32 value (error ~2^-33 |v|); the product with a 0/1 matrix in fp32 accumulation reproduces the fp32 sums.

**Result.** Correct (tests pass) but 4-8x slower on the T4. Triton uses tensor cores
for `tl.dot` from compute capability 8.0; on the T4 (7.5) the matmul runs on ordinary
FMA units and does more work than the plain sums. Kept as `v6_dot` to re-test on the
RTX PRO 6000.

## Packaging: custom ops and the drop-in module

`torch.compile` cannot see inside an `autograd.Function` that calls Triton (v5 caused 6
graph breaks: an autograd call and data-dependent ops). Registering forward and backward
as `torch.library` custom ops makes the loss two opaque nodes in the compiled graph:

::code src/ace_kernels/impls/v6_triton.py::v6_forward | The op's arguments are tensors, a config string and a flag; its outputs are the loss plus the per-bin tensors the backward needs.

::code src/ace_kernels/impls/v6_triton.py:152-165 | `register_fake`: output shapes without running the kernel (what `torch.compile` traces with).

Precise vs fast exp: Triton's `tl.exp` is a fast approximation; probabilities can differ
from `torch.softmax` by an ulp and move a near-edge voxel to the neighbouring bin:

::code src/ace_kernels/kernels/act.py::exp_tl | `PRECISE` (constexpr) selects libdevice `expf`, the function CUDA PyTorch uses. Cost on the T4: 0-6%.

The deliverable wraps all this in the upstream constructor:

::code src/ace_kernels/module.py::HardL1ACELoss | Same arguments, order and defaults as upstream (a test compares the signatures); `impl` is the only addition, keyword-only.

## Try it

::run uv run pytest tests/test_module.py -v | signature identical to upstream; 43 cases vs the upstream classes (CPU; CUDA on Kaggle)
::run uv run python -c "from ace_kernels.module import SoftL1ACELoss; import torch; l = SoftL1ACELoss(to_onehot_y=True, softmax=True); x = torch.randn(1, 4, 16, 16, 16, requires_grad=True); y = torch.randint(0, 4, (1, 1, 16, 16, 16)); print(l(x, y)); print(l.resolve_impl(x.device))" | on a Mac: the PyTorch v2 path (v2_recompute)

::check Why did the O(1) index help less than expected? | It removed K compares per voxel, but the reduction still builds a [BLOCK, K] one-hot and does 3 multiply-sums over it: that is most of the kernel's arithmetic.
::check Why is the fused epilogue 17-26x faster at 32^3/16^3 but almost no gain at 128^3? | Small calls are launch-bound (~55 launches x 5+ us); large calls are GPU-bound and the epilogue's work is tiny either way.
::check What does `register_fake` do and why is it needed? | It tells torch.compile the output shapes/dtypes of the op without running it, so the compiler can trace the graph around the opaque op.
