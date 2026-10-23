"""Generate tests/golden/reference_cases.pt from the frozen reference.

Stores, per case: input, target, config, loss and d(loss.sum())/d(input), plus the
environment. Rerun only on purpose (e.g. after a deliberate torch/monai upgrade)
and review the diff of test results first.

    uv run python scripts/make_golden.py
"""

import platform
import warnings
from pathlib import Path

import monai
import torch

from ace_kernels import ace_loss
from ace_kernels.cases import CASES, make_inputs

OUT = Path(__file__).resolve().parents[1] / "tests" / "golden" / "reference_cases.pt"


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    records = {}
    for case in CASES:
        x, y = make_inputs(case.spec, case.config)
        x.requires_grad_(True)
        loss = ace_loss(x, y, case.config, impl="reference")
        torch.autograd.backward(loss.sum())
        assert x.grad is not None
        records[case.name] = {
            "config": case.config.model_dump(),
            "input": x.detach().clone(),
            "target": y.clone(),
            "loss": loss.detach().clone(),
            "grad": x.grad.clone(),
        }
    env = {
        "torch": str(torch.__version__),
        "monai": str(monai.__version__),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"env": env, "cases": records}, OUT)
    print(f"wrote {len(records)} cases to {OUT} ({OUT.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
