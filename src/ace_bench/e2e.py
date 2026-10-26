"""End-to-end training check: the loss inside the paper's training step.

Mirrors the upstream bundle (``bundle_templates/configs``): SegResNetDS
(init_filters 32, blocks_down [1, 2, 2, 4, 4], deep supervision depth 4),
``DeepSupervisionLoss`` around 0.33 * (Dice + CE + mL1-ACE), AdamW (lr 1e-4,
wd 1e-5), AMP on CUDA. Data are synthetic but learnable volumes (smooth random
label maps + noisy intensity), generated per step from the seed, so every
implementation trains on identical batches.

Two experiments:
- curves: N training steps per (impl, binning): loss, ACE term, step time, peak
  memory per step -> overlaid curves and per-step costs.
- max patch: largest cubic patch whose training step fits in GPU memory.

    uv run python -m ace_bench.e2e bench/configs/sprint8_e2e.yaml [--wandb]
"""

import argparse
import itertools
import json
import subprocess
import sys
import time
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import polars as pl
import torch
import torch.nn.functional as F
import yaml
from pydantic import BaseModel, ConfigDict, Field
from torch.nn.modules.loss import _Loss

from ace_bench.env import REPO_ROOT, collect_env
from ace_kernels import ACEConfig, ace_loss
from ace_kernels.config import Binning

MiB = 2**20


class E2EConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    group: str
    hypothesis: str
    device: Literal["cpu", "cuda"] = "cuda"
    impls: list[str]
    binnings: list[Binning] = ["hard", "soft"]
    classes: int = Field(default=4, ge=2)
    batch: int = Field(default=2, ge=1)
    patch: list[int] = [96, 96, 96]
    steps: int = Field(default=60, ge=1)
    skip_steps: int = Field(default=5, ge=0)  # cudnn autotune / Triton autotune
    init_filters: int = 32
    amp: bool = True
    lr: float = 1e-4
    seed: int = 0
    max_patch_candidates: list[int] = []  # cubic sizes to probe (empty: skip)

    @classmethod
    def from_yaml(cls, path: Path) -> "E2EConfig":
        return cls.model_validate(yaml.safe_load(path.read_text()))


class StepRow(BaseModel):
    impl: str
    binning: str
    step: int
    loss: float
    ace: float
    step_ms: float | None
    peak_mib: float | None


class ProbeRow(BaseModel):
    impl: str
    binning: str
    patch: int
    status: Literal["ok", "oom", "error"]
    peak_mib: float | None = None
    error: str | None = None


def synthetic_batch(
    step: int, cfg: E2EConfig, patch: list[int], device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Smooth random label map + noisy intensity image; deterministic per step."""
    gen = torch.Generator().manual_seed(cfg.seed * 1_000_003 + step)
    coarse = torch.randn(
        cfg.batch, cfg.classes, *(max(1, s // 8) for s in patch), generator=gen
    )
    labels = F.interpolate(coarse, size=patch, mode="trilinear").argmax(1, keepdim=True)
    noise = torch.randn(cfg.batch, 1, *patch, generator=gen)
    image = labels.float() / (cfg.classes - 1) + 0.3 * noise
    # float labels, as loaded upstream: DeepSupervisionLoss resizes the target
    # with nearest interpolation, which has no integer kernel
    return image.to(device), labels.float().to(device)


class PaperLoss(_Loss):
    """0.33 * (Dice + CE + mL1-ACE), as ``*l1ace_dice_ce.yaml`` upstream."""

    def __init__(self, impl: str, binning: Binning) -> None:
        super().__init__()
        from monai.losses.dice import DiceLoss

        self.impl = impl
        self.ace_config = ACEConfig(binning=binning, to_onehot_y=True)
        self.dice = DiceLoss(
            to_onehot_y=True,
            softmax=True,
            squared_pred=True,
            smooth_nr=0,
            smooth_dr=1e-5,
        )
        self.ace_sum = torch.zeros(())  # GPU tensor: no host sync inside the step

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits.float(), labels[:, 0].long())
        ace = ace_loss(logits, labels, self.ace_config, impl=self.impl)
        # summed over deep-supervision levels; read after the timed step
        self.ace_sum = self.ace_sum.to(ace.device) + ace.detach().float()
        loss: torch.Tensor = 0.33 * (self.dice(logits, labels) + ce + ace)
        return loss


def _build(cfg: E2EConfig, impl: str, binning: Binning) -> tuple[Any, Any, Any, Any]:
    from monai.losses.ds_loss import DeepSupervisionLoss
    from monai.networks.nets.segresnet_ds import SegResNetDS

    torch.manual_seed(cfg.seed)
    model = SegResNetDS(
        spatial_dims=3,
        init_filters=cfg.init_filters,
        in_channels=1,
        out_channels=cfg.classes,
        blocks_down=(1, 2, 2, 4, 4),
        norm="INSTANCE",
        dsdepth=4,
    ).to(cfg.device)
    paper_loss = PaperLoss(impl, binning)
    loss_fn = DeepSupervisionLoss(paper_loss, weight_mode="exp")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-5)
    return model, paper_loss, loss_fn, opt


def _step(
    cfg: E2EConfig,
    model: Any,
    loss_fn: Any,
    opt: Any,
    scaler: Any,
    image: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    use_amp = cfg.amp and cfg.device == "cuda"
    ctx = torch.autocast("cuda", dtype=torch.float16) if use_amp else nullcontext()
    opt.zero_grad(set_to_none=True)
    with ctx:
        loss: torch.Tensor = loss_fn(model(image), labels)
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    return loss.detach()


def train_curve(cfg: E2EConfig, impl: str, binning: Binning) -> list[StepRow]:
    model, paper_loss, loss_fn, opt = _build(cfg, impl, binning)
    model.train()
    scaler = torch.amp.GradScaler(enabled=cfg.amp and cfg.device == "cuda")
    rows = []
    for step in range(cfg.steps):
        image, labels = synthetic_batch(step, cfg, cfg.patch, cfg.device)
        paper_loss.ace_sum = torch.zeros((), device=cfg.device)
        if cfg.device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record()
            loss = _step(cfg, model, loss_fn, opt, scaler, image, labels)
            end.record()
            torch.cuda.synchronize()
            ms, peak = start.elapsed_time(end), torch.cuda.max_memory_allocated() / MiB
        else:
            t0 = time.perf_counter()
            loss = _step(cfg, model, loss_fn, opt, scaler, image, labels)
            ms, peak = (time.perf_counter() - t0) * 1e3, None
        rows.append(
            StepRow(
                impl=impl,
                binning=binning,
                step=step,
                loss=float(loss),
                ace=float(paper_loss.ace_sum),
                step_ms=ms,
                peak_mib=peak,
            )
        )
        print(
            f"{impl:12s} {binning} step {step:3d} loss {float(loss):.4f} {ms:8.1f} ms",
            file=sys.stderr,
            flush=True,
        )
    return rows


def probe_patch(cfg: E2EConfig, impl: str, binning: Binning, size: int) -> ProbeRow:
    """One training step at a cubic patch: does it fit in memory?"""
    try:
        model, _, loss_fn, opt = _build(cfg, impl, binning)
        scaler = torch.amp.GradScaler(enabled=cfg.amp and cfg.device == "cuda")
        image, labels = synthetic_batch(0, cfg, [size] * 3, cfg.device)
        if cfg.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for _ in range(2):  # second step: optimizer state allocated
            _step(cfg, model, loss_fn, opt, scaler, image, labels)
        peak = torch.cuda.max_memory_allocated() / MiB if cfg.device == "cuda" else None
        return ProbeRow(
            impl=impl, binning=binning, patch=size, status="ok", peak_mib=peak
        )
    except torch.OutOfMemoryError:
        return ProbeRow(impl=impl, binning=binning, patch=size, status="oom")
    except Exception as exc:
        return ProbeRow(
            impl=impl,
            binning=binning,
            patch=size,
            status="error",
            error=f"{type(exc).__name__}: {exc}"[:300],
        )


_MARKER = "ACE_E2E_ROWS:"


def _isolated(task: dict[str, Any]) -> list[dict[str, Any]]:
    """Run one task in a fresh process (clean GPU memory per run / probe)."""
    proc = subprocess.run(
        [sys.executable, "-m", "ace_bench.e2e", "--worker"],
        input=json.dumps(task),
        stdout=subprocess.PIPE,
        text=True,
        check=False,
    )
    for line in proc.stdout.splitlines():
        if line.startswith(_MARKER):
            rows: list[dict[str, Any]] = json.loads(line.removeprefix(_MARKER))
            return rows
    return []


def worker() -> None:
    task = json.loads(sys.stdin.read())
    cfg = E2EConfig.model_validate(task["config"])
    if task["kind"] == "curve":
        rows: list[BaseModel] = list(train_curve(cfg, task["impl"], task["binning"]))
    else:
        rows = [probe_patch(cfg, task["impl"], task["binning"], task["size"])]
    print(_MARKER + json.dumps([r.model_dump() for r in rows]), flush=True)


def run(cfg: E2EConfig, out_root: Path, use_wandb: bool) -> Path:
    from ace_bench.report import write_e2e_summary

    env = collect_env(cfg.device)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = out_root / f"{stamp}-{cfg.name}"
    out.mkdir(parents=True)
    meta = {
        "kind": "e2e",
        "config": cfg.model_dump(),
        "env": env.model_dump(),
        "run_id": out.name,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    base = {"config": cfg.model_dump()}
    steps: list[dict[str, Any]] = []
    for binning, impl in itertools.product(cfg.binnings, cfg.impls):
        rows = _isolated({**base, "kind": "curve", "impl": impl, "binning": binning})
        if not rows:
            print(f"curve {impl} {binning}: worker crashed (see stderr)", flush=True)
        steps += rows
    probes: list[dict[str, Any]] = []
    for binning, impl in itertools.product(cfg.binnings, cfg.impls):
        for size in sorted(cfg.max_patch_candidates):
            row = _isolated(
                {
                    **base,
                    "kind": "probe",
                    "impl": impl,
                    "binning": binning,
                    "size": size,
                }
            )
            probes += row
            status = row[0]["status"] if row else "crashed"
            print(f"probe {impl} {binning} {size}^3: {status}", flush=True)
            if status != "ok":
                break  # larger patches will not fit either
    step_schema = {
        "impl": pl.String,
        "binning": pl.String,
        "step": pl.Int64,
        "loss": pl.Float64,
        "ace": pl.Float64,
        "step_ms": pl.Float64,
        "peak_mib": pl.Float64,
    }
    probe_schema = {
        "impl": pl.String,
        "binning": pl.String,
        "patch": pl.Int64,
        "status": pl.String,
        "peak_mib": pl.Float64,
        "error": pl.String,
    }
    pl.DataFrame(steps, schema=step_schema).write_parquet(out / "steps.parquet")
    pl.DataFrame(probes, schema=probe_schema).write_parquet(out / "probes.parquet")
    print(write_e2e_summary(out))
    if use_wandb:
        from ace_bench.tracking import mirror_e2e

        print(f"W&B: {mirror_e2e(out)}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end training check")
    parser.add_argument("config", type=Path, nargs="?")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker()
        return
    if args.config is None:
        parser.error("config is required")
    run(E2EConfig.from_yaml(args.config), args.out, args.wandb)


if __name__ == "__main__":
    main()
