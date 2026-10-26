"""Benchmark experiment configuration (YAML files in ``bench/configs/``)."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from ace_bench.calculator import Mode
from ace_kernels.config import ACEConfig, Binning


class ShapeSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    batch: int = Field(ge=1)
    channels: int = Field(ge=1)
    spatial: tuple[int, ...]

    @property
    def voxels(self) -> int:
        n = 1
        for s in self.spatial:
            n *= s
        return n


class BenchConfig(BaseModel):
    """One experiment: every impl x binning x shape x mode combination."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    group: str
    hypothesis: str
    device: Literal["cpu", "cuda"] = "cuda"
    impls: list[str]
    binnings: list[Binning] = ["hard", "soft"]
    modes: list[Mode] = ["fwd", "fwd_bwd"]
    shapes: list[ShapeSpec]
    cuda_graph: bool = False  # extra rows with the call captured in a CUDA graph
    # run each (shape, binning, impl) group in a fresh process, so a leak or a
    # broken CUDA context in one group cannot affect the others
    isolate: bool = True
    # compare every row's loss and gradient with the reference on the same input
    verify: bool = True
    # storage dtype of the logits fed to the loss (the reference computes in fp32)
    dtype: Literal["float32", "bfloat16", "float16"] = "float32"
    warmup: int = Field(default=3, ge=0)
    reps: int = Field(default=10, ge=1)
    seed: int = 0
    loss: ACEConfig = ACEConfig(to_onehot_y=True)  # paper settings

    @classmethod
    def from_yaml(cls, path: Path) -> "BenchConfig":
        return cls.model_validate(yaml.safe_load(path.read_text()))
