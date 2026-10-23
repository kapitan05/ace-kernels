# ace_reference (frozen oracle)

Verbatim copies of the original mL1-ACE losses. Every optimized implementation is
tested against these files.

| File | Upstream path | SHA-256 |
|---|---|---|
| `hardl1ace.py` | `src/losses/hardl1ace.py` | `fa8513bd18c0dcc8e17e64318b4ccd85b72405a745bcfe7332466a9d7fab6da6` |
| `softl1ace.py` | `src/losses/softl1ace.py` | `815d2acd230fe0503f8be0ad45b16c445200df9f7c98aa1549c538392d86eb8e` |

- Repo: https://github.com/cai4cai/Average-Calibration-Losses
- Commit: `faa93c7ca55daf092d1dee130c35da0fa0de2ff1` (2025-05-12)
- License: the upstream repo ships no LICENSE file. Ask the authors before
  redistributing these files outside this project.
- Upstream env: torch 2.5.0, monai 1.4.0. This project: see `uv.lock`.

Known quirks (kept on purpose, they define the reference behaviour):
- Internally casts input/target to fp32 (`.float()`), so fp64 `gradcheck` is not
  possible on the reference.
- Python loop over batch x channel; one `bucketize` + 3 scatters per (b, c).
- `HardL1ACELoss.forward` overwrites `self.class_weight` when a scalar weight is given.
- Paper configs (`bundle_templates/configs/loss/*l1ace*.yaml`): `num_bins=20`,
  `softmax=True`, one-hot target, `include_background=True`,
  `ignore_empty_classes=True`, `reduction="mean"`, soft `empty_weight=0.01`.
- `amos22.yaml` comment: patch size reduced "for softl1ace to prevent CUDA OOM".
