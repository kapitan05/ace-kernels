"""Frozen copy of the original mL1-ACE losses (correctness oracle).

Source: https://github.com/cai4cai/Average-Calibration-Losses
Commit: faa93c7ca55daf092d1dee130c35da0fa0de2ff1, files ``src/losses/hardl1ace.py``
and ``src/losses/softl1ace.py``. Do not edit the vendored files; their hashes are
checked by ``tests/test_reference_frozen.py``.
"""

from ace_reference.hardl1ace import HardL1ACELoss, hard_binned_calibration
from ace_reference.softl1ace import SoftL1ACELoss, soft_binned_calibration

__all__ = [
    "HardL1ACELoss",
    "SoftL1ACELoss",
    "hard_binned_calibration",
    "soft_binned_calibration",
]
