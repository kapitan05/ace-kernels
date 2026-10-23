"""Independent NumPy (float64) restatement of mL1-ACE, written from the math.

Per (b, c) slice with probabilities p_i and targets y_i, bin weights w_ik:
    W_k = sum_i w_ik,  P_k = sum_i w_ik p_i,  G_k = sum_i w_ik y_i
    f   = mean over non-empty k of |P_k - G_k| / W_k
Hard binning: w_ik = 1 if p_i falls in bin k.  Soft binning: linear interpolation
between the two nearest bin centres (all weight to the end bin outside the
centres).

The bin boundaries come from ``torch.linspace`` in float32, exactly like the
reference: NumPy's linspace differs by ~1 ulp, which moves probabilities that sit
exactly on an edge into the neighbouring bin. The boundary tensor is therefore
part of the loss definition; optimized kernels must reproduce it bit-exactly.
"""

import numpy as np
import numpy.typing as npt
import torch

from ace_kernels.config import ACEConfig

F64 = npt.NDArray[np.float64]
EPS32 = torch.finfo(torch.float32).eps


def _linspace32(steps: int) -> F64:
    """Reference boundaries: float32 linspace over [0, 1 + eps32]."""
    return torch.linspace(0.0, 1.0 + EPS32, steps).double().numpy()


def _hard_weights(p: F64, num_bins: int, right: bool) -> F64:
    edges = _linspace32(num_bins + 1)[1:]
    idx = np.searchsorted(edges, p, side="right" if right else "left")
    w = np.zeros((p.size, num_bins))
    w[np.arange(p.size), idx] = 1.0
    return w


def _soft_weights(p: F64, num_bins: int) -> F64:
    centres = _linspace32(2 * num_bins + 1)[1::2]
    w = np.zeros((p.size, num_bins))
    for i, v in enumerate(p):
        if v <= centres[0]:
            w[i, 0] = 1.0
        elif v > centres[-1]:
            w[i, -1] = 1.0
        else:
            j = int(np.searchsorted(centres, v, side="left")) - 1  # c_j < v <= c_j+1
            t = (v - centres[j]) / (centres[j + 1] - centres[j])
            w[i, j] += 1.0 - t
            w[i, j + 1] += t
    return w


def _slice_error(p: F64, y: F64, config: ACEConfig) -> float:
    if config.binning == "hard":
        w = _hard_weights(p, config.num_bins, config.right)
        empty = w.sum(0) == 0
    else:
        w = _soft_weights(p, config.num_bins)
        empty = w.sum(0) < config.empty_weight
    weight_sum = w.sum(0)
    errors = [
        abs(w[:, k] @ p - w[:, k] @ y) / weight_sum[k]
        for k in range(config.num_bins)
        if not empty[k]
    ]
    return float(np.mean(errors))


def oracle_ace(input: F64, target: F64, config: ACEConfig) -> F64:
    """mL1-ACE for ``input`` [B, C, ...] logits/probs and one-hot ``target``."""
    x = input.astype(np.float64)
    if config.activation == "softmax" and x.shape[1] > 1:
        e = np.exp(x - x.max(axis=1, keepdims=True))
        x = e / e.sum(axis=1, keepdims=True)
    elif config.activation == "sigmoid":
        x = 1.0 / (1.0 + np.exp(-x))
    y = target.astype(np.float64)
    if not config.include_background and x.shape[1] > 1:
        x, y = x[:, 1:], y[:, 1:]

    b, c = x.shape[:2]
    xf, yf = x.reshape(b, c, -1), y.reshape(b, c, -1)
    f = np.array(
        [[_slice_error(xf[i, j], yf[i, j], config) for j in range(c)] for i in range(b)]
    )
    if config.ignore_empty_classes:
        f = f * (yf.sum(-1) > 0)
    if config.weight is not None and c != 1:
        f = f * np.broadcast_to(np.asarray(config.weight, dtype=np.float64), (c,))
    if config.reduction == "mean":
        return np.asarray(f.mean())
    if config.reduction == "sum":
        return np.asarray(f.sum())
    return f.reshape(b, c, *([1] * (x.ndim - 2)))
