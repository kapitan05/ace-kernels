# Glossary

| term | meaning | page |
|---|---|---|
| ACE, mL1-ACE | average calibration error: mean over non-empty bins of \|mean p - mean y\|; mL1 = averaged over classes, used as a loss | [1](01-loss.html) |
| hard / soft binning | each voxel in one bin / spread over the two nearest bin centres with linear weights (smooth in p) | [1](01-loss.html) |
| calibration | predicted probabilities match observed frequencies | [1](01-loss.html) |
| memory-bound | time set by bytes moved, not arithmetic; the limit is bandwidth (T4: 320 GB/s) | [1](01-loss.html) |
| bandwidth floor | ideal bytes / peak bandwidth: the fastest possible time | [7](07-tooling.html) |
| roofline | comparing measured performance with the hardware's compute and bandwidth limits | [1](01-loss.html) |
| % of peak bandwidth | ideal bytes / measured time / peak bandwidth | [7](07-tooling.html) |
| launch overhead, launch-bound | ~5 us CPU cost per GPU kernel; a call is launch-bound when that dominates | [5](05-sprint11.html) |
| kernel | one GPU program launch | [4](04-triton.html) |
| atomic add, contention | indivisible read-modify-write; threads hitting the same address serialize | [1](01-loss.html) |
| privatization | each block accumulates in private registers and publishes once | [4](04-triton.html) |
| Triton program, block, grid | one instance of a kernel; the vector of elements it handles; all instances | [4](04-triton.html) |
| constexpr | compile-time kernel argument; branches on it cost nothing at run time | [4](04-triton.html) |
| autotune | trying several block sizes on first call, keeping the fastest | [4](04-triton.html) |
| fusion | doing several steps in one kernel so data is read once | [4](04-triton.html) |
| recompute vs store | redo cheap work in backward instead of keeping tensors in memory | [3](03-pytorch.html) |
| saved tensors | what autograd keeps between forward and backward | [3](03-pytorch.html) |
| subgradient, kink | derivative of \|x\| at 0 is undefined; any value in [-1, 1] is valid (PyTorch: 0) | [2](02-correctness.html) |
| ulp | unit in the last place: the gap between adjacent floats | [2](02-correctness.html) |
| relative L2 error | \|\|a - b\|\| / \|\|b\|\| over the whole tensor; robust accuracy measure | [2](02-correctness.html) |
| golden files | stored reference outputs for fixed inputs | [2](02-correctness.html) |
| oracle | independent implementation used as ground truth | [2](02-correctness.html) |
| gradcheck | compares an analytic gradient with finite differences (float64) | [3](03-pytorch.html) |
| CUDA graph | a recorded sequence of kernels replayed with one launch | [5](05-sprint11.html) |
| host sync | the CPU waits for the GPU (e.g. `.item()`): stalls and breaks graphs | [1](01-loss.html) |
| torch.compile, graph break | PyTorch's compiler; a break splits the graph where it cannot trace | [6](06-training.html) |
| custom op (`torch.library`) | registering a function so the compiler treats it as one opaque node | [5](05-sprint11.html) |
| deep supervision | loss on several output resolutions of the network (4 for SegResNetDS) | [6](06-training.html) |
| Amdahl's law | speedup of the whole = 1 / ((1 - s) + s / x) for a part s made x times faster | [6](06-training.html) |
| AMP | automatic mixed precision: fp16 compute where safe | [6](06-training.html) |
| tensor cores | matrix-multiply units (fp16/bf16); used by Triton `tl.dot` from compute capability 8.0 | [5](05-sprint11.html) |
| libdevice exp | NVIDIA's accurate `expf` (vs Triton's fast approximation) | [5](05-sprint11.html) |
| ablation | removing / adding one change at a time to attribute effects | [index](index.html) |
| control run | the baseline repeated, to measure its own noise | [6](06-training.html) |
