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

::walk src/ace_kernels/kernels/epilogue.py::_epilogue_kernel
@ def _epilogue_kernel( | Launched with a grid of **one** program: the whole epilogue is a few hundred numbers. Inputs: the three per-bin statistics (`R = B·C'` rows of K bins), the class weights; outputs: `coef` and `d` for the backward, `jac = dloss/df`, the per-class values `f` (for reduction "none") and the scalar loss.
@ rows = tl.arange(0, RP) | **One tile holds everything**: `[RP, KP]` = all (b, c) rows x all bins (both rounded up to powers of two, masked by `inside`).
@ if HARD:  # noqa: SIM108 | **Empty bins**: count 0 (hard) or total weight below `empty_weight` (soft), as in the reference.
@ nonempty = filled & inside | **Per bin**: `d = mean p - mean y` with two IEEE divisions (same order as the reference, so the sign at exact ties matches), 0 for empty bins.
@ k_ne = tl.sum(nonempty.to(tl.float32), axis=1)  # [RP] | **Per row**: K_ne (non-empty bins), the per-class loss `f` = mean of `|d|` over non-empty bins (NaN if none, like `nanmean`), and `coef = sign(d) / (K_ne W)` for the backward.
@ cls_w = tl.load(cls_w_ptr + rows % CL, mask=row_ok, other=0.0) | **Class mask and weights**: row r is class `r % C'`. Absent classes (`sum_k sum_y = 0`) are zeroed when `ignore_empty_classes`. `scale` = derivative of the final loss with respect to each `f` (the jacobian the backward needs): presence x weight (/ R for the mean).
@ total = tl.sum(tl.where(row_ok, f, 0.0), axis=0) | **Reduction** (mean or sum) and all stores. Replaces ~50 PyTorch launches, including an autograd call that differentiated the reduction.
::end

::table results/kaggle/60a3c6df2c3c-run-sprint11/results/*sprint11-deepsup/rows.parquet | df.filter((pl.col("binning") == "hard") & (pl.col("status") == "ok")).with_columns(pl.when(pl.col("cuda_graph")).then(pl.col("impl") + " + graph").otherwise(pl.col("impl")).alias("impl")).pivot(on="impl", index="shape", values="p50_ms") | Deep-supervision levels, hard, ms per fwd+bwd (T4). v6 needs no CUDA graph; v5 needed one at small sizes.

::plot docs/study/assets/figures/deep-supervision.png | Per-level time for reference / v5 / v5 + CUDA graph / v6 | summed over the 4 levels of one training step: hard 62 -> 1.9 ms, soft 118 -> 2.8 ms.

## O(1) bin index, exactly

**Concept.** The kernels found a bin by comparing p against all K edges (20 compares,
40 for soft). The edges are almost `j / K`, so `floor(p * K)` is within one of the right
bin. Two correction steps against the **same float32 edge table** make it exact:

::walk src/ace_kernels/kernels/bins.py::sorted_count
@ def sorted_count( | Returns the same number as `torch.bucketize(p, table, right)`: how many table entries are below p (`<=` p if `right`). `table` = the reference's float32 edges, entry j ≈ (j+1)/n.
@ g = tl.minimum(tl.maximum(p * n, 0.0), n * 1.0) | **The guess**: `floor(p · n)`, clamped to `[0, n]` (handles p < 0, p > 1, +inf). NaN fails every comparison, so it is mapped to 0 explicitly, the compare loop's answer. Converting a non-negative float to int truncates, which is floor.
@ for _ in tl.static_range(2): | **Two correction rounds** against the real table. Up: if the entry at position g is still counted for p, g was too small. Down: if the entry just below g is *not* counted, g was too large. The guess is off by at most one (the float32 edges differ from j/n by a few ulp), so one round suffices; the second is insurance. Each round is two gathers from a tiny cached table and two compares: O(1) instead of n compares.
::end

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

::walk src/ace_kernels/kernels/bins.py::split3
@ def split3(v): | Splits an fp32 value into three fp16 pieces whose sum reproduces it (to ~2^-33 relative), so fp16 tensor-core products lose nothing.
@ a = v.to(tl.float16).to(tl.float32) | `a`: v rounded to fp16 (11 significant bits). `r = v - a` is exact in fp32 (the two are close).
@ b = (r * 2048.0).to(tl.float16).to(tl.float32) | `b`: the next 11 bits. The remainder is scaled by 2^11 first so it is not tiny in fp16 (fp16 has a narrow exponent range); the caller divides the matrix product's column by 2^11 again.
@ c = (r * 4194304.0).to(tl.float16).to(tl.float32) | `c`: the last bits, scaled by 2^22. Since a 0/1 one-hot times an fp16 value is exact and the tensor core accumulates in fp32, `a + b/2^11 + c/2^22` summed over a block equals the fp32 sum of v up to rounding order.
::end

**Result.** Correct (tests pass) but 4-8x slower on the T4. Triton uses tensor cores
for `tl.dot` from compute capability 8.0; on the T4 (7.5) the matmul runs on ordinary
FMA units and does more work than the plain sums. Kept as `v6_dot` to re-test on the
RTX PRO 6000.

## Packaging: custom ops and the drop-in module

`torch.compile` cannot see inside an `autograd.Function` that calls Triton (v5 caused 6
graph breaks: an autograd call and data-dependent ops). Registering forward and backward
as `torch.library` custom ops makes the loss two opaque nodes in the compiled graph:

::walk src/ace_kernels/impls/v6_triton.py:141-235
@ @torch.library.custom_op("ace_kernels::v6_forward", mutates_args=()) | **Registering an operator.** `custom_op` makes the function a PyTorch operator named `ace_kernels::v6_forward`, like a built-in. Its signature may only use tensors and simple types, hence the config is passed as a JSON string. `mutates_args=()` promises it modifies no input.
@ @v6_forward.register_fake | **The fake ("meta") implementation**: given input shapes, return empty tensors of the output shapes and dtypes, without running a kernel. `torch.compile` traces the program with fake tensors, so it needs this to know what the op returns.
@ @torch.library.custom_op("ace_kernels::v6_backward", mutates_args=()) | The backward is an operator too, so the compiled backward graph can contain it.
@ def _setup_context(ctx: Any, inputs: tuple[Any, ...], output: Outputs) -> None: | **Autograd for a custom op**: `setup_context` runs after the forward and saves what the backward needs: the inputs (held by the caller anyway) and the per-bin outputs. `set_materialize_grads(False)`: the helper outputs coef / d / jac never receive gradients, so autograd must not allocate zero tensors for them (3 launches saved).
@ def _op_backward( | Called by autograd with the gradient of each output; only the loss's is used. Returns one gradient per input (none for target, config, flag).
@ def _entry(opt: Options | bool) -> Any: | **The registry function**: float64 inputs (gradcheck in tests) go to the PyTorch version; `v6_triton` / `v6_precise` call the op (bool = precise flag); `v6_index` / `v6_dot` use a plain `autograd.Function` (no compile support needed for intermediate ladder steps). `config_arg` is marked as a compile-time constant (pydantic objects cannot be traced).
::end

Precise vs fast exp: Triton's `tl.exp` is a fast approximation; probabilities can differ
from `torch.softmax` by an ulp and move a near-edge voxel to the neighbouring bin:

::walk src/ace_kernels/kernels/act.py::exp_tl
@ def exp_tl(x, PRECISE: tl.constexpr): | `tl.exp` compiles to a fast hardware approximation (`ex2.approx`); `libdevice.exp` is NVIDIA's accurate `expf`, the one CUDA PyTorch uses for `torch.softmax`. With `PRECISE` our probabilities match `torch.softmax` more often to the last bit, so near-edge voxels land in the same bins as in the reference.
::end

The deliverable wraps all this in the upstream constructor:

::walk src/ace_kernels/module.py:44-106
@ def __init__( | The shared constructor of both modules (the public classes forward their upstream-shaped arguments here as keywords).
@ super().__init__(reduction=LossReduction(reduction).value) | **Same validation as upstream**: unknown reductions, a non-callable `other_act`, or more than one activation raise the same exception types (tested).
@ activation: Activation = ( | Upstream booleans -> one `ACEConfig` (pydantic, frozen: validated once, hashable). `_weight` accepts the same weight types as upstream (sequence, number, tensor) and stores Python floats (no device tensor: CUDA-graph safe).
@ if impl != "auto": | An explicitly requested implementation is checked now, not at the first call.
@ def resolve_impl(self, device: torch.device) -> str: | **Per call, by device**: Triton `v6_precise` on CUDA, the handwritten-backward PyTorch `v2_recompute` elsewhere. A model moved between devices keeps working.
@ def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor: | A custom `other_act` is applied first (the kernels then see probabilities, activation "none"); then the chosen implementation.
::end

## Try it

::run uv run pytest tests/test_module.py -v | signature identical to upstream; 43 cases vs the upstream classes (CPU; CUDA on Kaggle)
::run uv run python -c "from ace_kernels.module import SoftL1ACELoss; import torch; l = SoftL1ACELoss(to_onehot_y=True, softmax=True); x = torch.randn(1, 4, 16, 16, 16, requires_grad=True); y = torch.randint(0, 4, (1, 1, 16, 16, 16)); print(l(x, y)); print(l.resolve_impl(x.device))" | on a Mac: the PyTorch v2 path (v2_recompute)

::check Why did the O(1) index help less than expected? | It removed K compares per voxel, but the reduction still builds a [BLOCK, K] one-hot and does 3 multiply-sums over it: that is most of the kernel's arithmetic.
::check Why is the fused epilogue 17-26x faster at 32^3/16^3 but almost no gain at 128^3? | Small calls are launch-bound (~55 launches x 5+ us); large calls are GPU-bound and the epilogue's work is tiny either way.
::check What does `register_fake` do and why is it needed? | It tells torch.compile the output shapes/dtypes of the op without running it, so the compiler can trace the graph around the opaque op.
