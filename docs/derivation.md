# mL1-ACE: forward and handwritten backward

Notation for one (b, c) slice after activation and background removal: voxels
`i = 1..N`, probabilities `p_i`, targets `y_i ∈ {0, 1}`, bins `k = 1..K`.

## Forward

Each probability is assigned to bins with weights `w_ik`:

- **hard**: `w_ik = 1` for the bin `k(i)` that contains `p_i`, else 0. Bin edges are
  the reference's float32 `linspace(0, 1 + eps32, K + 1)`; `right` picks the
  bucketize convention.
- **soft**: linear interpolation between the two neighbouring bin centres
  `c_l < p_i <= c_r` (centres = odd points of `linspace(0, 1 + eps32, 2K + 1)`):

      w_il = (c_r - p_i) / S,   w_ir = (p_i - c_l) / S,   S = c_r - c_l

  Below the first / above the last centre both neighbours are the end bin; the
  reference uses sentinels ±10000 there, so `S ≈ 10000` and the two weights still
  sum to 1 on that bin.

Per bin:

    W_k = Σ_i w_ik,   P_k = Σ_i w_ik p_i,   G_k = Σ_i w_ik y_i
    d_k = P_k / W_k - G_k / W_k            (mean prob minus mean target)
    e_k = |d_k|

A bin is **empty** when `W_k = 0` (hard) or `W_k < empty_weight` (soft); empty bins
are excluded. With `K_ne` non-empty bins:

    f = (1 / K_ne) Σ_{k non-empty} e_k     (the reference's nanmean)

Then per slice: `f_bc ← f_bc · [class present]` (if `ignore_empty_classes`)
`· weight_c`, and the reduction over (b, c): mean, sum or none.

## Backward

Let `g_bc = ∂L/∂f_bc`. It comes from the mask, weights and reduction only, and is
obtained by differentiating that tiny `[B, C] → loss` step (`reduce_classes`) with
autograd, so it matches the reference by construction.

`f` depends on `p_i` only through the bins that contain `i`. With
`s_k = sign(d_k)` (sign(0) = 0, as PyTorch's `abs` backward):

    ∂f/∂p_i = (1 / K_ne) Σ_{k non-empty} s_k ∂d_k/∂p_i

`d_k = (P_k - G_k) / W_k` (same value, different rounding), so

    ∂d_k/∂p_i = (∂P_k/∂p_i - ∂G_k/∂p_i) / W_k - d_k (∂W_k/∂p_i) / W_k

with `w'_ik = ∂w_ik/∂p_i`:

    ∂W_k/∂p_i = w'_ik
    ∂P_k/∂p_i = w_ik + w'_ik p_i
    ∂G_k/∂p_i = w'_ik y_i

Collecting terms, **each bin j that voxel i contributes to** gives

    ∂f/∂p_i = Σ_j  s_kj / (K_ne W_kj) · [ w_ij + w'_ij (p_i - y_i - d_kj) ]

- **hard**: one bin, `w = 1`, `w' = 0`:

      ∂f/∂p_i = s_k(i) / (K_ne W_k(i))

- **soft**: two sides j ∈ {l, r}, `w'_il = -1/S`, `w'_ir = +1/S`. At the ends both
  sides hit the same bin and the `w'` terms cancel (`∂W/∂p = 0` there).

So the backward needs, per bin, only

    coef_k = s_k / (K_ne W_k)   (0 for empty bins)   and   d_k

— two `[B, C, K]` tensors — plus a recomputation of each voxel's bins and weights
from `p`. Nothing voxel-sized has to be stored between forward and backward.

## Through the activation

`∂L/∂p_bci = g_bc ∂f_bc/∂p_i` (zero for a dropped background channel), then:

- softmax over channels: `∂L/∂x_c = p_c (∂L/∂p_c - Σ_c' p_c' ∂L/∂p_c')`
- sigmoid: `∂L/∂x = ∂L/∂p · p (1 - p)`
- none: `∂L/∂x = ∂L/∂p`

The softmax couples channels, so the dropped background channel still receives a
gradient through the other channels' probabilities.

## Edge cases

| case | handling |
|---|---|
| empty bin | excluded from `K_ne`; `coef_k = 0`, no gradient (reference: NaN, masked by nanmean) |
| all bins empty | `f = NaN` like the reference; gradient 0 (reference: NaN) |
| exact tie `d_k = 0` | `s_k = 0`, same as eager PyTorch (compiled kernels may differ, see tests) |
| `p` exactly on a hard edge | same bucketize and edges as the reference |
| background dropped, index targets | voxels of class 0 add to no channel's `G` |
| class absent, `ignore_empty_classes` | `g_bc = 0` via the mask |

## Memory

| | saved for backward (beyond inputs) |
|---|---|
| reference, hard | 3× input (int64 bin index 2×, softmax output 1×) |
| reference, soft | 11× input |
| v2_recompute | `coef`, `d`: 2 · B·C·K floats + class presence B·C bools (648 B at B=2, C=4, K=20) |
| v2_grad_in_fwd | the input gradient (1× input), backward only scales it |
