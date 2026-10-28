# 4. Triton kernels: privatized histograms, fused backward, fused softmax

Sprints 5-8 replace the PyTorch internals with GPU kernels written in
[Triton](https://triton-lang.org): Python-syntax kernels compiled to GPU code, where
you program a **block** of data at a time instead of single threads.

## Concept: the Triton programming model in 6 lines

- A kernel runs as many independent **programs** (a grid). `tl.program_id(axis)` tells
  a program which piece of the data it owns.
- Each program works on whole **blocks**: `offs = pid * BLOCK + tl.arange(0, BLOCK)`
  is a vector of BLOCK indices, and `tl.load(ptr + offs, mask=offs < N)` loads them.
- Values live in **registers**; `tl.sum(x, axis=0)` reduces a block inside the program.
- `tl.constexpr` arguments are compile-time constants: `if ACT == 1:` is resolved when
  compiling, so one source gives specialized kernels without runtime branches.
- `@triton.autotune` tries several BLOCK sizes / warp counts on first call and keeps
  the fastest per input-size key.
- `tl.atomic_add` is the only way programs combine results in global memory.

Our grid: axis 0 = blocks of voxels, axis 1 = batch element. (It was the other way
round first; CUDA limits grid axis 1 to 65535 programs, which a 256^3 volume exceeds.
The fix and a regression test are in Sprint 6.)

## The forward kernel: a private histogram per program

::svg histogram-tile

Each program loads BLOCK voxels, computes their bin, builds the one-hot matrix
`[BLOCK, K]` in registers, sums it over voxels (`tl.sum`) into a **private** K-bin
histogram, and publishes it with **one** atomic add per bin and statistic. Collisions
drop from millions to a few thousand:

::walk src/ace_kernels/kernels/hard_hist.py::_hard_hist_kernel
@ @triton.autotune( | **Autotuning.** On the first call for a new `key` (voxels N, channels C, bins K, DOT), Triton compiles and times every config in `_CONFIGS` (BLOCK 128/256/512 voxels per program x 4/8 warps of 32 threads) and caches the fastest. `reset_to_zero=["out_ptr"]`: the output is accumulated with atomics, so it must be re-zeroed between the autotuner's trial runs. `prune_configs_by` drops tiles that do not fit in shared memory for the DOT variant.
@ def _hard_hist_kernel( | **Arguments.** Pointers to the logits (any float dtype), the class-index labels or a one-hot target, the float32 edge table, and the output `[3, B, C', K]` (count, sum p, sum y). Then sizes, and the `tl.constexpr` flags: each combination of flags compiles to a separate specialized kernel, so `if ACT == 1:` or `if DIRECT:` cost nothing at run time. `KP`/`CP`: K and C rounded up to a power of two, because `tl.arange` only makes power-of-two vectors.
@ # voxel blocks on grid axis 0 | **Which data this program owns.** The launch grid is `(ceil(N / BLOCK), B)`: `program_id(1)` is the batch element, `program_id(0)` the block of voxels. `offs` is the vector of this block's BLOCK voxel indices; `valid` masks the tail of the last block (N is rarely a multiple of BLOCK). Voxel blocks are on axis 0 because axis 1 allows only 65535 programs, fewer than a 256^3 volume needs (a real bug found at 256^3).
@ bins = tl.arange(0, KP) | The vector `0..KP-1` of bin numbers, and `bin_ok` masks the padding bins K..KP-1 (K = 20 -> KP = 32: 12 wasted lanes, part of why the reduction is the remaining cost).
@ if USE_LABELS: | With class-index targets, load each voxel's true class once (int64, `-1` for padding). All channels below reuse it, so no one-hot target is ever read or built.
@ m = tl.zeros([BLOCK], dtype=tl.float32) | **Fused softmax, part 1.** For `ACT == 1` (softmax), compute each voxel's maximum logit `m` and `s = sum_c exp(x_c - m)` over all channels once (`softmax_stats`, below). For sigmoid or none, neutral values are used.
@ for c in range(OFFSET, C): | **Loop over the loss channels** (skipping channel 0 when the background is excluded). Load this channel's logits for the block (converted to fp32 in registers, so fp16/bf16 inputs work and are read at half the bytes) and turn them into probabilities: `exp(x - m) / s`. The probability tensor never exists in GPU memory.
@ if DIRECT: | **The bin of every voxel**, identical to `torch.bucketize` on the reference's edge table. Default (v3-v5): count the edges below p with K vectorized compares (`tl.static_range` unrolls the loop at compile time; each `tl.load(edges_ptr + k)` is one scalar, cached). `DIRECT` (v6): O(1) floor + correction (page 5).
@ if USE_LABELS: | Target of this channel: `y = 1` where the voxel's class is `c`. Padding voxels give 0.
@ if DOT: | (Sprint 11 side branch, slower on the T4: the same reduction as a tensor-core matrix product. Page 5.)
@ else: | **The private histogram.** `hit` is the `[BLOCK, KP]` one-hot matrix (row = voxel, column = bin), built in registers by comparing every voxel's bin with every bin number. Summing over voxels (`axis=0`) gives this block's count per bin; multiplying by p or y before summing gives the per-bin sums. This is the "privatization": all accumulation happens inside the program, without touching global memory.
@ out = out_ptr + (b * (C - OFFSET) + c - OFFSET) * K + bins | **Publish once.** The address of row (b, c) of each statistic; `BCK` is the distance between the three statistics. Three `atomic_add` of K values each: per block, 3K atomics instead of 3·BLOCK. Atomic float adds arrive in a different order every run, which is why sums can differ in the last bit between runs (but counts are exact).
::end

**How Python launches it.** A kernel is called like a function indexed by its grid:

::walk src/ace_kernels/kernels/hard_hist.py::hard_hist_fused
@ def hard_hist_fused( | The Python wrapper: takes the logits `[B, C, N]` (a free view of `[B, C, X, Y, Z]`), the target and the config; returns the `[3, B, C', K]` statistics.
@ b, c, n = x.shape | `offset = 1` drops the background channel from the loss (`include_background=False`). The output is **zero-initialized**: programs only ever *add* into it.
@ use_labels = target.dim() == 2 | Class indices (`[B, N]`) or a one-hot target (`[B, C, N]`) select the `USE_LABELS` variant. `act_code` maps the config's activation to the constexpr (softmax over one channel is the identity, as in the reference). The tensor-core path is only allowed where it is exact.
@ def grid(meta: dict[str, int]) -> tuple[int, int]: | **The grid is a function** of the chosen config: the autotuner calls it with each candidate `BLOCK`, so the number of programs is always `ceil(N / BLOCK)` x B.
@ _hard_hist_kernel[grid]( | **The launch.** Tensors are passed as pointers; plain ints as runtime scalars; keyword `UPPER_CASE` values fill the `tl.constexpr` parameters (changing one compiles a new kernel variant, cached afterwards). Unused pointers (one-hot when labels are used) get any valid tensor. `edge_table` returns the reference's float32 edges, built once per device and cached (no `linspace` launch per call).
::end

**Exactness.** The bin of a probability must match the reference bit-for-bit, or the
loss would differ for voxels on edges. The kernel compares against the reference's own
float32 edge tensor (`torch.linspace(0, 1 + eps32, K + 1)`) with the same comparison as
`torch.bucketize` (`right=False`: count of edges < p). Edge-exact inputs are golden
cases (`bin-edges-left/right`), and `test_hard_stats_match_torch` requires **exactly
equal** bin counts.

## Worked example: the private-histogram reduction on the CPU

The same arithmetic as one Triton program, written with torch on the CPU, against the
reference's scatter:

::exec
import torch
torch.manual_seed(0)
K, BLOCK, N = 20, 256, 4096
p = torch.rand(N)
y = (torch.rand(N) < p).float()
edges = torch.linspace(0, 1 + torch.finfo(torch.float32).eps, K + 1)[1:]
idx = torch.bucketize(p, edges)

# reference style: every voxel adds into one of K addresses (atomics on a GPU)
counts_ref = torch.zeros(K).scatter_add(0, idx, torch.ones(N))
sum_p_ref = torch.zeros(K).scatter_add(0, idx, p)

# kernel style: per block, one-hot [BLOCK, K] -> private histogram -> one add per bin
counts, sum_p, atomics = torch.zeros(K), torch.zeros(K), 0
for start in range(0, N, BLOCK):
    b_idx, b_p = idx[start:start + BLOCK], p[start:start + BLOCK]
    hit = (b_idx[:, None] == torch.arange(K)[None, :]).float()  # [BLOCK, K] in registers
    counts += hit.sum(0)                                        # tl.sum(hit, axis=0)
    sum_p += (hit * b_p[:, None]).sum(0)
    atomics += 2 * K                                            # one atomic per bin
print("counts identical:", torch.equal(counts, counts_ref))
print("sum p rel. diff :", ((sum_p - sum_p_ref).abs() / sum_p_ref).max().item(), "(fp32 summation order)")
print(f"atomic adds     : reference {2 * N}, kernel {atomics} ({2 * N // atomics}x fewer)")
::end

## The backward kernel: one pass, nothing stored

The backward (Sprint 6) reads each voxel's logits once, recomputes probability and
bin, gathers `coef[bin]` from the tiny per-bin table (page 3), applies the softmax
backward across channels in registers, and writes the gradient once:

::walk src/ace_kernels/kernels/hard_bwd.py::_hard_bwd_kernel
@ @triton.autotune(configs=_CONFIGS, key=["N", "C", "K"]) | Autotuned like the forward (no `reset_to_zero`: each output element is written exactly once, no atomics).
@ def _hard_bwd_kernel( | Inputs: logits, the per-bin `coef` table from the forward (`[B, C', K]`, a few hundred floats), `g = dL/df` per (b, c), the edge table. Output: the gradient with respect to the logits, `[B, C, N]`.
@ # voxel blocks on grid axis 0 | **A 2D tile `[CP, BLOCK]`**: all channels of BLOCK voxels in one program. `ptrs` is a matrix of addresses (row = channel, column = voxel). The softmax backward needs all channels of a voxel together, so they are kept in one tile.
@ if FROM_LOGITS: | **Recompute the probabilities** from the logits with exactly the same functions as the forward (`softmax_stats`, `activate_tl`): bit-identical p, so every voxel falls into the same bin as in the forward. (v4 instead loaded probabilities from `torch.softmax`.)
@ if DIRECT: | **Recompute the bins** (same method as the forward). Nothing voxel-sized was stored between forward and backward: re-binning costs a few compares, storing int64 bins would cost 8 bytes per voxel and channel.
@ c_loss = C - OFFSET  # channels that enter the loss | **The derivative of the loss with respect to p**: gather `coef[b, c, bin]` for every voxel and channel (a gather from a tiny table that stays in cache), times `g[b, c]`. The background channel, if excluded from the loss, gets 0 here, but still receives gradient through the softmax below.
@ if ACT == 1:  # softmax over channels | **Chain rule through the activation, in registers.** Softmax: `dx_c = p_c (gp_c - sum_c' p_c' gp_c')`, the sum over channels is a `tl.sum(axis=0)` within the tile. Sigmoid: `gp · p (1 - p)`. Then one store in the input's dtype. Total traffic: read logits once, write the gradient once.
::end

## Fusing the softmax (Sprint 8)

v4 still called `torch.softmax` before both kernels: one extra read + write of a
probability tensor in the forward and again in the backward (31% of v4's time). v5
computes the softmax inside both kernels from the logits:

::walk src/ace_kernels/kernels/act.py::softmax_stats
@ def softmax_stats( | Called by every kernel that starts from logits. Arguments: the logits pointer, batch index, the block's voxel offsets and mask, and `CP` (channels rounded to a power of two).
@ chan = tl.arange(0, CP) | Load a `[CP, BLOCK]` tile of logits: row c, column voxel. Padding rows (c >= C) and padding voxels load `-inf`, so they vanish in the max and give `exp(-inf) = 0` in the sum.
@ m = tl.where(valid, tl.max(x, axis=0), 0.0) | **Numerically stable softmax**: subtracting the per-voxel maximum `m` before `exp` keeps every exponent <= 0, so nothing overflows even for large logits. `s` is the softmax denominator. Padding voxels get `m = 0, s = 1` (safe values, never used).
::end

::walk src/ace_kernels/kernels/act.py::activate_tl
@ def activate_tl( | One function for every activation; `ACT` is a compile-time constant, so only one branch is compiled into each kernel.
@ if ACT == 1: | Softmax probability of a channel: `exp(x - m) / s`. `tl.div_rn` is IEEE round-to-nearest division, like PyTorch's `/` (Triton's default division may be approximate). `exp_tl` is Triton's fast exp, or libdevice `expf` with `PRECISE` (page 5).
@ elif ACT == 2: | Sigmoid `1 / (1 + exp(-x))`, used for binary (single-channel) segmentation.
@ else: | No activation: the input already holds probabilities.
::end

Side effect: logits in fp16/bf16 are read directly (half the bytes), all math in fp32.
And no probability tensor exists at all: peak memory halves again.

## Soft binning

Soft binning assigns each probability to its two neighbouring bin centres with linear
weights. Bit-exactness again comes from using the reference's own tables, including
its +-10000 sentinels at the ends:

::walk src/ace_kernels/kernels/soft_hist.py::soft_assign_tl
@ def soft_assign_tl( | Soft binning of a vector (or tile) of probabilities: for each, the two bins it contributes to and the two weights. Shared by the soft forward and backward kernels.
@ if DIRECT:  # half boundaries half[1:], 2K values | **Half-bin index h**: the reference interleaves bin centres and bin edges into one table of 2K+1 "half boundaries" (`linspace(0, 1+eps, 2K+1)`); h counts those below p. From h, the neighbouring centres follow directly.
@ left = tl.maximum(h - 1, 0) // 2 | **The two bins**: the centres just below and above p. At the ends (below the first centre / above the last) both are the end bin; the weights then add up on that one bin.
@ left_dist = p - tl.load(repl_ptr + h) | **Distances to the two centres**, read from the reference's replicated-centres table (`-10000, c0, c0, c1, c1, ..., 10000`). The +-10000 sentinels reproduce the reference's behaviour at the ends exactly (the span S becomes ~10000 there). Table lookups with per-voxel indices are gathers from a tiny cached table.
@ span = left_dist + right_dist | **Linear interpolation weights**: the closer centre gets the larger weight; `lw + rw = 1`. IEEE division again, to match PyTorch bit for bit.
::end

## Evidence

::plot docs/ablation/ladder-soft.png | Ablation ladder, soft binning, kits23 | v3 (Triton forward) removes the atomic bottleneck; v4 (Triton backward) the PyTorch backward; v5 the softmax passes; soft gains are larger than hard (the reference's soft path is 11x memory and 981 kernels).

::plot docs/study/assets/figures/time-breakdown.png | GPU time by kernel type | v4: the Triton kernels plus a visible softmax share; v5/v6: only the Triton kernels remain.

::table results/kaggle/5423f093984d-run-sprint7/results/*sprint7-memory*/rows.parquet | df.filter((pl.col("mode") == "fwd_bwd") & ~pl.col("cuda_graph")).select("shape", "impl", "status", pl.col("p50_ms").round(1).alias("ms"), (pl.col("peak_extra_bytes") / 2**20).round(0).alias("peak MiB")).sort("shape", ladder) | Sprint 7 (soft, T4): at 288^3 v4 needs 1.46 GiB peak where the reference does not fit.

**Seeing it yourself: profiler traces.** Each profile folder contains `trace.json`.
Open [ui.perfetto.dev](https://ui.perfetto.dev), drag the file in, and look at the GPU
stream row: for v5 you see two long kernels per step (histogram, backward) and almost
nothing else; for the reference, hundreds of short kernels.

::run open results/kaggle/1ab2f019d85e-run-sprint8/results/profile/v5_triton-hard-kits23-fwd_bwd | the folder with trace.json (Perfetto), summary tables, saved-tensor list

## Try it

Triton has no macOS build, so the kernel tests run on the Kaggle GPU. Read their
intent in `tests/test_triton.py`; to run them remotely (page 7):

::run uv run python -m ace_bench.remote submit kaggle/run_triton_dev.sh | Triton tests on a Kaggle T4 (~15-30 min); then: uv run python -m ace_bench.remote fetch kaggle/run_triton_dev.sh

::check Why are atomics per slice reduced from N to 3 K N / BLOCK? | Each program handles BLOCK voxels and issues one atomic per bin (K) for each of the 3 statistics: N / BLOCK programs x 3 K atomics.
::check Why must forward and backward use the same exp/softmax function? | The bin of a voxel is decided by its probability. If the backward recomputed a probability that differs by 1 ulp and crosses a bin edge, the voxel would use another bin's coef: a wrong gradient for that voxel.
::check Why does the backward kernel tile all channels of a voxel together? | The softmax backward needs sum over channels of p_c g_c for each voxel; having all channels in one program keeps that sum in registers instead of another pass.
