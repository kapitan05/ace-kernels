"""Evidence figures for the study guide, from the saved Kaggle runs (no GPU needed).

    uv run python -m ace_bench.figures      # -> docs/study/assets/figures/

Each figure backs one claim of ``docs/report.md``; the study pages explain them.
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl
from matplotlib.figure import Figure

from ace_bench.ablation import _style, latest_ok, load_rows
from ace_bench.env import REPO_ROOT
from ace_bench.profile import ACTIVE_STEPS
from ace_bench.report import SERIES, SURFACE, TEXT, TEXT_2

KAGGLE = REPO_ROOT / "results" / "kaggle"
OUT = REPO_ROOT / "docs" / "study" / "assets" / "figures"
MiB = 2**20

MAIN = [
    "reference",
    "reference_compile",
    "v1_clean",
    "v1_clean_compile",
    "v2_recompute",
    "v3_triton",
    "v4_triton",
    "v5_triton",
    "v6_triton",
    "v6_precise",
]


def _run(pattern: str) -> pl.DataFrame:
    paths = sorted(KAGGLE.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no run matches {pattern}")
    return pl.read_parquet(paths[-1])


def _save(fig: Figure, out: Path, name: str) -> Path:
    fig.tight_layout()
    path = out / name
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def _paper_rows(shape: str) -> pl.DataFrame:
    best = latest_ok(load_rows(KAGGLE))
    return best.filter(
        (pl.col("mode") == "fwd_bwd") & (pl.col("shape") == shape)
    ).filter(pl.col("impl").is_in(MAIN))


def fig_distance(out: Path) -> Path:
    """Measured time / bandwidth floor: how far each impl is from the hardware."""
    rows = _paper_rows("kits23")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=SURFACE)
    for ax, binning in zip(axes, ("hard", "soft"), strict=True):
        sub = rows.filter(pl.col("binning") == binning)
        impls = [i for i in MAIN if sub.filter(pl.col("impl") == i).height]
        dist = [
            sub.filter(pl.col("impl") == i)
            .select(pl.col("p50_ms") / pl.col("bw_floor_ms"))
            .item()
            for i in impls
        ]
        ax.barh(range(len(impls)), dist, color=SERIES[0], height=0.7)
        for y, v in enumerate(dist):
            ax.text(v, y, f" {v:.1f}x", va="center", fontsize=7, color=TEXT_2)
        ax.axvline(1.0, color=SERIES[2], linewidth=1.2)
        ax.text(1.05, -0.75, "hardware limit", color=SERIES[2], fontsize=7)
        ax.set_xscale("log")
        ax.set_yticks(range(len(impls)), impls)
        ax.invert_yaxis()
        _style(ax, f"{binning}: measured time / bandwidth floor (log; 1 = limit)")
    fig.suptitle(
        "Distance from the memory-bandwidth limit, kits23 fwd+bwd, Tesla T4",
        x=0.01,
        ha="left",
        fontsize=10,
        color=TEXT,
    )
    return _save(fig, out, "distance-to-limit.png")


def fig_memory(out: Path) -> Path:
    """Peak extra memory and bytes saved for backward, along the ladder."""
    rows = _paper_rows("kits23")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=SURFACE)
    for ax, binning in zip(axes, ("hard", "soft"), strict=True):
        sub = rows.filter(pl.col("binning") == binning)
        impls = [i for i in MAIN if sub.filter(pl.col("impl") == i).height]
        for k, (col, label) in enumerate(
            (
                ("peak_extra_bytes", "peak extra"),
                ("saved_extra_bytes", "saved for backward"),
            )
        ):
            vals = [(sub.filter(pl.col("impl") == i)[col][0] or 0) / MiB for i in impls]
            ys = [y + (k - 0.5) * 0.38 for y in range(len(impls))]
            ax.barh(ys, vals, height=0.36, color=SERIES[k], label=label)
            for y, v in zip(ys, vals, strict=True):
                ax.text(v, y, f" {v:.0f}", va="center", fontsize=6, color=TEXT_2)
        ax.set_yticks(range(len(impls)), impls)
        ax.invert_yaxis()
        ax.legend(fontsize=7, frameon=False, loc="lower right")
        _style(ax, f"{binning}: MiB beyond the inputs (input = 32 MiB)")
    fig.suptitle(
        "Memory, kits23 fwd+bwd (2 x 4 x 128^3, fp32), Tesla T4",
        x=0.01,
        ha="left",
        fontsize=10,
        color=TEXT,
    )
    return _save(fig, out, "memory-ladder.png")


def fig_accuracy(out: Path) -> Path:
    """Loss error vs float64 truth: noise inputs vs trained-network-like inputs."""
    df = _run("*-run-sprint11-precise/results/*/rows.parquet")
    impls = ["reference", "v5_triton", "v6_triton", "v6_precise"]
    shapes = ["kits23", "brats21", "amos22", "kits23-trained"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), facecolor=SURFACE)
    for ax, binning in zip(axes, ("hard", "soft"), strict=True):
        sub = df.filter(pl.col("binning") == binning)
        width = 0.8 / len(impls)
        for k, impl in enumerate(impls):
            vals = []
            for s in shapes:
                v = sub.filter((pl.col("impl") == impl) & (pl.col("shape") == s))
                vals.append(max(v["loss_err_truth"][0] or 1e-12, 1e-12))
            xs = [x + (k - 1.5) * width for x in range(len(shapes))]
            ax.bar(xs, vals, width=width, color=SERIES[k], label=impl)
        ax.set_yscale("log")
        ax.set_xticks(range(len(shapes)), shapes)
        ax.legend(fontsize=7, frameon=False)
        ax.grid(axis="y", color="#e4e3df", linewidth=0.6)
        _style(ax, f"{binning}: |loss - float64 loss| / |float64 loss| (log)")
    fig.suptitle(
        "Loss accuracy against a float64 computation (lower is better), Tesla T4",
        x=0.01,
        ha="left",
        fontsize=10,
        color=TEXT,
    )
    return _save(fig, out, "accuracy.png")


def _deepsup() -> pl.DataFrame:
    df = _run("*-run-sprint11/results/*sprint11-deepsup/rows.parquet")
    return df.filter(pl.col("status") == "ok").with_columns(
        pl.when(pl.col("cuda_graph"))
        .then(pl.col("impl") + " + CUDA graph")
        .otherwise(pl.col("impl"))
        .alias("label")
    )


def fig_deepsup(out: Path) -> Path:
    """Per deep-supervision level: the small levels are launch-bound for v5."""
    df = _deepsup()
    labels = ["reference", "v5_triton", "v5_triton + CUDA graph", "v6_triton"]
    levels = ["ds0-128", "ds1-64", "ds2-32", "ds3-16"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), facecolor=SURFACE)
    for ax, binning in zip(axes, ("hard", "soft"), strict=True):
        sub = df.filter(pl.col("binning") == binning)
        width = 0.8 / len(labels)
        for k, lab in enumerate(labels):
            vals = []
            for lv in levels:
                v = sub.filter((pl.col("label") == lab) & (pl.col("shape") == lv))
                vals.append(v["p50_ms"][0] if v.height else 0.0)
            xs = [x + (k - 1.5) * width for x in range(len(levels))]
            ax.bar(xs, vals, width=width, color=SERIES[k], label=lab)
        ax.set_yscale("log")
        ax.set_xticks(range(len(levels)), ["128^3", "64^3", "32^3", "16^3"])
        ax.legend(fontsize=7, frameon=False)
        ax.grid(axis="y", color="#e4e3df", linewidth=0.6)
        _style(ax, f"{binning}: ms per loss call (fwd+bwd, log)")
    fig.suptitle(
        "Deep supervision: one loss call per SegResNetDS output level (B=1, C=4), T4",
        x=0.01,
        ha="left",
        fontsize=10,
        color=TEXT,
    )
    return _save(fig, out, "deep-supervision.png")


def fig_scaling(out: Path) -> Path:
    """Time vs voxels (log-log) with each impl's launch floor."""
    df = _deepsup().filter(~pl.col("cuda_graph"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), facecolor=SURFACE)
    for ax, binning in zip(axes, ("hard", "soft"), strict=True):
        sub = df.filter(pl.col("binning") == binning).sort("voxels")
        for k, impl in enumerate(["reference", "v5_triton", "v6_triton"]):
            d = sub.filter(pl.col("impl") == impl)
            ax.plot(d["voxels"], d["p50_ms"], "o-", color=SERIES[k], label=impl)
            ax.plot(
                d["voxels"],
                d["launch_floor_ms"],
                ":",
                color=SERIES[k],
                linewidth=1,
                label=f"{impl} launch floor",
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("voxels per channel", fontsize=8, color=TEXT_2)
        ax.legend(fontsize=6, frameon=False, loc="lower right")
        ax.grid(color="#e4e3df", linewidth=0.6)
        _style(ax, f"{binning}: ms per fwd+bwd vs volume (log-log)")
    fig.suptitle(
        "Scaling 16^3 -> 128^3: flat = launch-bound, rising = GPU-bound, T4",
        x=0.01,
        ha="left",
        fontsize=10,
        color=TEXT,
    )
    return _save(fig, out, "scaling.png")


_GROUPS = [
    ("Triton ACE kernels", ("_hist_kernel", "_bwd_kernel", "_epilogue_kernel")),
    ("softmax", ("softmax",)),
    ("scatter / index (atomics)", ("scatter", "index", "bucketize", "gather")),
    ("reductions", ("reduce_kernel",)),
    ("Inductor (compiled)", ("triton_",)),
]


def _group(name: str) -> str:
    for label, keys in _GROUPS:
        if any(k in name.lower() for k in keys):
            return label
    return "other elementwise"


def fig_breakdown(out: Path) -> Path:
    """Where GPU time goes per implementation (profile traces, kits23 hard)."""
    traces = [
        ("reference", "*-run-sprint2/results/profile/reference-hard-kits23-*"),
        ("v1_clean", "33facc78d8f3-run-sprint3/results/profile/v1_clean-hard-*"),
        (
            "v2_recompute",
            "a2696e81824d-run-sprint4/results/profile/v2_recompute-hard-*",
        ),
        ("v4_triton", "dc029258d7b6-run-sprint56/results/profile/v4_triton-hard-*"),
        ("v5_triton", "*-run-sprint8/results/profile/v5_triton-hard-*"),
        ("v6_triton", "60a3c6df2c3c-run-sprint11/results/profile/v6_triton-hard-*"),
    ]
    data: dict[str, Counter[str]] = {}
    for impl, pattern in traces:
        path = sorted(KAGGLE.glob(pattern))[-1] / "trace.json"
        events = json.loads(path.read_text())["traceEvents"]
        per: Counter[str] = Counter()
        for e in events:
            if e.get("cat") == "kernel":
                per[_group(e["name"])] += e["dur"]
        # profile.py records ACTIVE_STEPS steps; durations in us
        data[impl] = Counter({k: v / ACTIVE_STEPS / 1e3 for k, v in per.items()})
    groups = [g for g, _ in _GROUPS] + ["other elementwise"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), facecolor=SURFACE)
    impls = list(data)
    for ax, log in zip(axes, (False, True), strict=True):
        left = [0.0] * len(impls)
        for k, g in enumerate(groups):
            vals = [data[i].get(g, 0.0) for i in impls]
            ax.barh(range(len(impls)), vals, left=left, color=SERIES[k % 6], label=g)
            left = [a + b for a, b in zip(left, vals, strict=True)]
        for y, v in enumerate(left):
            ax.text(v, y, f" {v:.1f} ms", va="center", fontsize=7, color=TEXT_2)
        ax.set_yticks(range(len(impls)), impls)
        ax.invert_yaxis()
        if log:
            ax.set_xscale("log")
        _style(ax, "GPU ms per fwd+bwd" + (" (log)" if log else ""))
    axes[0].legend(fontsize=7, frameon=False, loc="lower right")
    fig.suptitle(
        "GPU time by kernel type, kits23 hard fwd+bwd (torch.profiler), Tesla T4",
        x=0.01,
        ha="left",
        fontsize=10,
        color=TEXT,
    )
    return _save(fig, out, "time-breakdown.png")


TraceEvent = dict[str, Any]


def _one_step(trace: Path) -> tuple[list[TraceEvent], list[TraceEvent]]:
    """GPU events (kernels, copies) of the middle profiled step and the CPU
    launch calls that issued them (linked by the profiler's correlation id)."""
    events = json.loads(trace.read_text())["traceEvents"]
    gpu = sorted(
        (e for e in events if e.get("cat") in ("kernel", "gpu_memcpy")),
        key=lambda e: e["ts"],
    )
    third = len(gpu) // ACTIVE_STEPS
    step = gpu[third : 2 * third]
    ids = {e.get("args", {}).get("correlation") for e in step}
    cpu = [
        e
        for e in events
        if e.get("cat") == "cuda_runtime"
        and e.get("args", {}).get("correlation") in ids
    ]
    return step, cpu


def fig_timeline(out: Path) -> Path:
    """CPU launches vs GPU execution for one step: GPU-bound looks like slide 41."""
    traces = [
        (
            "original (reference), hard, kits23",
            "fb8ebdca9b33-run-sprint2/results/"
            "profile/reference-hard-kits23-fwd_bwd/trace.json",
        ),
        (
            "ours (v6_triton), hard, kits23",
            "60a3c6df2c3c-run-sprint11/results/"
            "profile/v6_triton-hard-kits23-fwd_bwd/trace.json",
        ),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(12, 4.6), facecolor=SURFACE)
    for ax, (title, rel) in zip(axes, traces, strict=True):
        gpu, cpu = _one_step(KAGGLE / rel)
        t0 = min(e["ts"] for e in cpu + gpu)
        end = max(e["ts"] + e["dur"] for e in gpu)
        busy = sum(e["dur"] for e in gpu) / (end - min(e["ts"] for e in gpu))
        for e in cpu:
            ax.barh(
                1,
                e["dur"] / 1e3,
                left=(e["ts"] - t0) / 1e3,
                height=0.5,
                color=SERIES[3],
                linewidth=0,
            )
        for e in gpu:
            scatter = any(k in e["name"].lower() for k in ("scatter", "index"))
            color = SERIES[1] if scatter else SERIES[0]
            ax.barh(
                0,
                e["dur"] / 1e3,
                left=(e["ts"] - t0) / 1e3,
                height=0.5,
                color=color,
                linewidth=0,
            )
        ax.set_yticks([0, 1], ["GPU kernels", "CPU launches"])
        ax.set_xlabel("ms from the first launch of the step", fontsize=8, color=TEXT_2)
        _style(
            ax,
            f"{title}: {len(gpu)} GPU ops, GPU busy {100 * busy:.0f}% "
            f"of its active span",
        )
    axes[0].text(
        0.99,
        0.95,
        "orange = scatter / index (atomics)",
        ha="right",
        va="top",
        transform=axes[0].transAxes,
        fontsize=7,
        color=SERIES[1],
    )
    fig.suptitle(
        "One fwd+bwd step on the GPU timeline (torch.profiler traces, Tesla T4): "
        "CPU launches finish early, the GPU runs back to back = GPU-bound",
        x=0.01,
        ha="left",
        fontsize=10,
        color=TEXT,
    )
    return _save(fig, out, "timeline.png")


def fig_step(out: Path) -> Path:
    """Training step: eager / torch.compile x reference / ours."""
    steps = _run("60a3c6df2c3c-run-sprint11/results/*sprint11-e2e/steps.parquet")
    timed = steps.filter(pl.col("step") >= 5)
    labels = ["reference", "reference +default", "v6_triton", "v6_triton +default"]
    names = ["original, eager", "original + compile", "ours, eager", "ours + compile"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), facecolor=SURFACE)
    for ax, binning in zip(axes, ("hard", "soft"), strict=True):
        sub = timed.filter(pl.col("binning") == binning)
        vals = [
            float(
                sub.filter(pl.col("impl") == lab)
                .select(pl.col("step_ms").median())
                .item()
            )
            for lab in labels
        ]
        base = vals[0]
        colors = [SERIES[1], SERIES[3], SERIES[0], SERIES[2]]
        ax.barh(range(4), vals, color=colors, height=0.7)
        for y, ms in enumerate(vals):
            ax.text(ms, y, f" {ms:.0f} ms ({base / ms:.2f}x)", va="center", fontsize=7)
        ax.set_yticks(range(4), names)
        ax.invert_yaxis()
        ax.set_xlim(0, base * 1.25)
        _style(ax, f"{binning}: median training-step time (128^3, AMP)")
    fig.suptitle(
        "SegResNetDS training step: our kernels speed up the loss, "
        "torch.compile mainly the network",
        x=0.01,
        ha="left",
        fontsize=10,
        color=TEXT,
    )
    return _save(fig, out, "training-step.png")


FIGURES = [
    fig_distance,
    fig_memory,
    fig_accuracy,
    fig_deepsup,
    fig_scaling,
    fig_breakdown,
    fig_step,
    fig_timeline,
]


def build(out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    return [fig(out) for fig in FIGURES]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evidence figures (saved runs)")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    for path in build(args.out):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
