"""Sprint 9: one ablation table across all sprints' benchmark runs.

    uv run python -m ace_bench.ablation            # -> docs/ablation/

Collects ``rows.parquet`` from every fetched Kaggle run (``results/kaggle``),
drops invalid runs (see ``EXCLUDED_RUNS``), and keeps, per implementation x
binning x shape x mode, the measurement from the most recent valid run (fp32
logits, no CUDA graph). Writes the ablation table (markdown + LaTeX), predicted vs
measured floors, the accuracy table (float64 verification runs) and plots.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

from ace_bench.env import REPO_ROOT
from ace_bench.report import GRID, SERIES, SURFACE, TEXT, TEXT_2, to_latex, to_markdown

# Runs whose numbers must not be used (kept on disk as evidence of the bug).
EXCLUDED_RUNS = {
    "afa97a0e3643": "Sprint 3 run 4: memory leak in the harness (38 OOM rows)",
    "967b720e1e11": "Sprint 5+6 run 1: class-presence scatter and grid-limit bugs, "
    "superseded by dc02925",
}

# The ablation ladder: one change per step, in sprint order.
LADDER = [
    ("reference", "S1", "original upstream loss"),
    ("reference_checkpoint", "S2", "+ torch.utils.checkpoint"),
    ("reference_compile", "S2", "+ torch.compile"),
    ("v1_vectorized", "S3", "B x C loop removed"),
    ("v1_clean", "S3", "+ sums, no one-hot, no host sync"),
    ("v1_clean_compile", "S3", "+ torch.compile"),
    ("v2_recompute", "S4", "handwritten backward (PyTorch)"),
    ("v3_triton", "S5", "Triton private-histogram forward"),
    ("v4_triton", "S6/S7", "+ Triton backward"),
    ("v5_triton", "S8", "+ softmax fused into both kernels"),
    ("v6_index", "S11", "+ O(1) bin index"),
    ("v6_triton", "S11", "+ one-launch epilogue, torch.library op"),
    ("v6_dot", "S11", "side branch: v6_index + tensor-core reduction"),
]
PAPER_SHAPES = ["kits23", "acdc17", "brats21", "amos22"]
MiB = 2**20


def load_rows(root: Path) -> pl.DataFrame:
    frames = []
    for path in sorted(root.glob("*/results/*/rows.parquet")):
        sha = path.parents[2].name.split("-")[0]
        if sha in EXCLUDED_RUNS:
            continue
        df = pl.read_parquet(path)
        for col, dtype in (
            ("dtype", pl.String),
            ("grad_l2_err_truth", pl.Float64),
            ("loss_err_truth", pl.Float64),
        ):
            if col not in df.columns:
                fill = "float32" if col == "dtype" else None
                df = df.with_columns(pl.lit(fill, dtype).alias(col))
        frames.append(
            df.with_columns(
                pl.lit(sha).alias("commit"), pl.lit(path.parent.name).alias("run")
            )
        )
    return pl.concat(frames, how="diagonal_relaxed")


# Accuracy-only runs (few repetitions): used for the accuracy table, not timings.
TIMING_EXCLUDED = ("verify",)


def latest_ok(rows: pl.DataFrame) -> pl.DataFrame:
    """Most recent valid benchmark measurement per impl x binning x shape x mode
    (fp32 logits, no CUDA graph, not from accuracy-only runs)."""
    ok = rows.filter(
        (pl.col("status") == "ok")
        & ~pl.col("cuda_graph")
        & (pl.col("dtype") == "float32")
        & ~pl.col("run").str.contains("|".join(TIMING_EXCLUDED))
    )
    return (
        ok.sort("run")
        .group_by("impl", "binning", "shape", "mode", maintain_order=True)
        .last()
    )


def _fmt(rows: pl.DataFrame, col: str, scale: float, spec: str) -> str:
    """First value of ``col`` / scale, formatted; "" when absent (e.g. CPU runs)."""
    if not rows.height or rows[col][0] is None:
        return ""
    return format(rows[col][0] / scale, spec)


def ablation_table(best: pl.DataFrame, binning: str) -> pl.DataFrame:
    sub = best.filter((pl.col("binning") == binning) & (pl.col("mode") == "fwd_bwd"))
    ref = sub.filter(pl.col("impl") == "reference").select(
        "shape", pl.col("p50_ms").alias("ref_ms")
    )
    sub = sub.join(ref, on="shape", how="left")
    records = []
    for impl, sprint, change in LADDER:
        row: dict[str, object] = {"sprint": sprint, "impl": impl, "change": change}
        found = False
        for shape in PAPER_SHAPES:
            r = sub.filter((pl.col("impl") == impl) & (pl.col("shape") == shape))
            if r.height:
                found = True
                ms, ref_ms = r["p50_ms"][0], r["ref_ms"][0]
                speed = f" ({ref_ms / ms:.1f}x)" if ref_ms else ""
                row[f"{shape} ms"] = f"{ms:.1f}{speed}"
            else:
                row[f"{shape} ms"] = ""
        k = sub.filter((pl.col("impl") == impl) & (pl.col("shape") == "kits23"))
        b = sub.filter((pl.col("impl") == impl) & (pl.col("shape") == "brats21"))
        row["% peak BW (kits23)"] = _fmt(k, "pct_peak_bw", 1.0, ".1f")
        row["peak MiB (brats21)"] = _fmt(b, "peak_extra_bytes", MiB, ".0f")
        row["saved MiB (brats21)"] = _fmt(b, "saved_extra_bytes", MiB, ".1f")
        if found:
            records.append(row)
    return pl.DataFrame(records)


def floors_table(best: pl.DataFrame, binning: str, shape: str) -> pl.DataFrame:
    """Measured time vs the bandwidth floor and the kernel-launch floor."""
    sub = best.filter(
        (pl.col("binning") == binning)
        & (pl.col("mode") == "fwd_bwd")
        & (pl.col("shape") == shape)
    )
    records = []
    for impl, _, _ in LADDER:
        r = sub.filter(pl.col("impl") == impl)
        if not r.height:
            continue
        ms, bw, launch = r["p50_ms"][0], r["bw_floor_ms"][0], r["launch_floor_ms"][0]
        if bw is None:  # CPU run: no device bandwidth, no floor
            continue
        records.append(
            {
                "impl": impl,
                "measured ms": round(ms, 2),
                "bandwidth floor ms": round(bw, 2),
                "x above floor": round(ms / bw, 1),
                "kernels / call": r["kernels_per_call"][0],
                "launch floor ms": round(launch, 2) if launch is not None else None,
            }
        )
    return pl.DataFrame(records)


def accuracy_table(rows: pl.DataFrame) -> pl.DataFrame:
    """Loss / L2-gradient error vs float64 truth, worst over shapes (latest run)."""
    truth = rows.filter(pl.col("loss_err_truth").is_not_null())
    if truth.height == 0:
        return pl.DataFrame()
    latest = truth.filter(pl.col("run") == truth["run"].max())
    agg = [pl.col("loss_err_truth").max().alias("max loss err vs fp64")]
    if latest["grad_l2_err_truth"].drop_nulls().len():
        agg.append(pl.col("grad_l2_err_truth").max().alias("max L2 grad err vs fp64"))
    table = latest.group_by("impl", "binning").agg(agg)
    order = {impl: i for i, (impl, _, _) in enumerate(LADDER)}
    table = (
        table.with_columns(pl.col("impl").replace_strict(order, default=99).alias("_o"))
        .sort("binning", "_o")
        .drop("_o")
    )
    return table.with_columns(
        pl.selectors.float().map_elements(lambda v: f"{v:.1e}", return_dtype=pl.String)
    )


def _style(ax: object, title: str) -> None:
    ax.set_facecolor(SURFACE)  # type: ignore[attr-defined]
    ax.set_title(title, fontsize=9, color=TEXT, loc="left")  # type: ignore[attr-defined]
    ax.grid(axis="x", color=GRID, linewidth=0.6)  # type: ignore[attr-defined]
    ax.set_axisbelow(True)  # type: ignore[attr-defined]
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)  # type: ignore[attr-defined]
    ax.tick_params(colors=TEXT_2, labelsize=8)  # type: ignore[attr-defined]


def _column(
    sub: pl.DataFrame, impls: list[str], name: str, scale: float = 1.0
) -> list[float]:
    """``name`` per impl (first row), scaled; 0 when absent (CPU: no GPU data)."""
    vals = [sub.filter(pl.col("impl") == i)[name][0] for i in impls]
    return [0.0 if v is None else v / scale for v in vals]


def ladder_plots(best: pl.DataFrame, out: Path) -> list[Path]:
    """Per binning: time (log) and % of peak bandwidth along the ladder, kits23."""
    paths = []
    for binning in ("hard", "soft"):
        sub = best.filter(
            (pl.col("binning") == binning)
            & (pl.col("mode") == "fwd_bwd")
            & (pl.col("shape") == "kits23")
        )
        impls = [i for i, _, _ in LADDER if sub.filter(pl.col("impl") == i).height]
        if not impls:
            continue
        ms = [sub.filter(pl.col("impl") == i)["p50_ms"][0] for i in impls]

        bw = _column(sub, impls, "pct_peak_bw")
        mem = _column(sub, impls, "peak_extra_bytes", MiB)
        fig, axes = plt.subplots(
            1, 3, figsize=(13, 0.5 * len(impls) + 1.4), facecolor=SURFACE
        )
        for ax, vals, title, log in (
            (axes[0], ms, "Time per fwd+bwd, ms (log)", True),
            (axes[1], bw, "% of peak DRAM bandwidth (100% = hardware limit)", False),
            (axes[2], mem, "Peak extra memory, MiB", False),
        ):
            ax.barh(range(len(impls)), vals, color=SERIES[0], height=0.7)
            for y, v in enumerate(vals):
                ax.text(v, y, f" {v:.3g}", va="center", fontsize=7, color=TEXT_2)
            ax.set_yticks(range(len(impls)), impls)
            ax.invert_yaxis()
            if log:
                ax.set_xscale("log")
            _style(ax, title)
        for ax in axes[1:]:
            ax.set_yticklabels([])
        fig.suptitle(
            f"Ablation, {binning} binning, kits23 (2 x 4 x 128^3), Tesla T4",
            x=0.01,
            ha="left",
            fontsize=10,
            color=TEXT,
        )
        fig.tight_layout()
        path = out / f"ladder-{binning}.png"
        fig.savefig(path, dpi=150, facecolor=SURFACE)
        plt.close(fig)
        paths.append(path)
    return paths


def build(root: Path, out: Path) -> str:
    out.mkdir(parents=True, exist_ok=True)
    rows = load_rows(root)
    best = latest_ok(rows)
    best.write_parquet(out / "ablation_rows.parquet")
    lines = [
        "# Ablation (generated by `python -m ace_bench.ablation`)",
        "",
        "Tesla T4 (Kaggle), fp32 logits, forward + backward, p50 over 10 calls, "
        "paper training shapes (batch 2): kits23 4 x 128^3, acdc17 4 x 256^2 x 16, "
        "brats21 4 x 224^2 x 144, amos22 16 x 176^2 x 64. Each cell: most recent "
        "valid run of that implementation. Excluded runs: "
        + "; ".join(f"`{k}` ({v})" for k, v in EXCLUDED_RUNS.items())
        + ".",
        "",
    ]
    tex = []
    for binning in ("hard", "soft"):
        table = ablation_table(best, binning)
        lines += [
            f"## {binning} binning: time (speedup vs reference)",
            "",
            to_markdown(table),
            "",
        ]
        tex.append(to_latex(table, f"Ablation, {binning} binning (Tesla T4)"))
    for binning in ("hard", "soft"):
        floors = floors_table(best, binning, "kits23")
        lines += [
            f"## Predicted floors vs measured, {binning}, kits23",
            "",
            to_markdown(floors),
            "",
        ]
    acc = accuracy_table(rows)
    if acc.height:
        lines += [
            "## Accuracy vs float64 truth (latest verification run)",
            "",
            to_markdown(acc),
            "",
        ]
    for p in ladder_plots(best, out):
        lines.append(f"![{p.stem}]({p.name})")
    used = best.select("run", "commit").unique().sort("run")
    lines += ["", "## Runs used", "", to_markdown(used), ""]
    text = "\n".join(lines)
    (out / "ablation.md").write_text(text)
    (out / "ablation.tex").write_text("\n\n".join(tex))
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the cross-sprint ablation")
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "results" / "kaggle")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "ablation")
    args = parser.parse_args()
    print(build(args.root, args.out))


if __name__ == "__main__":
    main()
