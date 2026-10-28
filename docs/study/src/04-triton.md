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

::code src/ace_kernels/kernels/hard_hist.py::_hard_hist_kernel | Read top to bottom: block offsets and labels; optional fused softmax (`softmax_stats`, page 5 of the ladder); per channel: activation, bin index (`DIRECT`: O(1), Sprint 11; else K compares), target y, reduction (`DOT`: tensor-core variant; else the one-hot sums), three `atomic_add`s.

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

::code src/ace_kernels/kernels/hard_bwd.py::_hard_bwd_kernel | The tile is `[channels, BLOCK]`: all channels of a voxel in one program, because the softmax backward couples channels: `dx_c = p_c (g_c - sum_c' p_c' g_c')`.

## Fusing the softmax (Sprint 8)

v4 still called `torch.softmax` before both kernels: one extra read + write of a
probability tensor in the forward and again in the backward (31% of v4's time). v5
computes the softmax inside both kernels from the logits:

::code src/ace_kernels/kernels/act.py::softmax_stats | Per voxel: max over channels and the sum of exp (numerically stable softmax), on a `[CP, BLOCK]` tile.

::code src/ace_kernels/kernels/act.py::activate_tl | Probability of one channel from its logit and the voxel's (max, sum). Forward and backward call the **same** function, so a voxel lands in the same bin in both passes.

Side effect: logits in fp16/bf16 are read directly (half the bytes), all math in fp32.
And no probability tensor exists at all: peak memory halves again.

## Soft binning

Soft binning assigns each probability to its two neighbouring bin centres with linear
weights. Bit-exactness again comes from using the reference's own tables, including
its +-10000 sentinels at the ends:

::code src/ace_kernels/kernels/soft_hist.py::soft_assign_tl | Half-bin index h (count of half boundaries below p), left/right bins, distances to the replicated centre table, IEEE division (`tl.div_rn`) like PyTorch's `/`.

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
