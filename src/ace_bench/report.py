"""Tables and plots from a results directory.

    uv run python -m ace_bench.report results/<run_id>

Writes ``summary.md`` (markdown tables), ``summary.tex`` (LaTeX tables for the
write-up) and ``plots/*.png``.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl
from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter

from ace_bench.measure import VERIFY_TOLERANCE

# Validated categorical palette (dataviz skill reference instance), fixed order:
# implementation i in the config always gets slot i.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

MiB = 2**20


def _clean(df: pl.DataFrame) -> pl.DataFrame:
    """Peak memory of a CUDA-graph replay is not meaningful (older runs logged 0)."""
    if "allocated_after_bytes" not in df.columns:  # runs before the leak check
        df = df.with_columns(pl.lit(None, pl.Int64).alias("allocated_after_bytes"))
    for col, dtype in (
        ("loss_rel_err", pl.Float64),
        ("grad_rel_err", pl.Float64),
        ("checked_against", pl.String),
    ):  # runs before the output check
        if col not in df.columns:
            df = df.with_columns(pl.lit(None, dtype).alias(col))
    return df.with_columns(
        pl.when(pl.col("cuda_graph"))
        .then(None)
        .otherwise(pl.col("peak_extra_bytes"))
        .alias("peak_extra_bytes")
    )


def _table(df: pl.DataFrame) -> pl.DataFrame:
    """Human-readable columns, one row per shape x impl x cuda_graph."""
    ref = df.filter((pl.col("impl") == "reference") & ~pl.col("cuda_graph")).select(
        "shape", "binning", "mode", pl.col("p50_ms").alias("_ref_ms")
    )
    out = df.join(ref, on=["shape", "binning", "mode"], how="left")
    impl = pl.when(pl.col("cuda_graph")).then(pl.col("impl") + " +graph")
    return out.select(
        pl.col("shape"),
        impl.otherwise(pl.col("impl")).alias("impl"),
        pl.col("status"),
        pl.col("p50_ms").round(3).alias("p50 ms"),
        (
            pl.format("{}-{}", pl.col("p20_ms").round(3), pl.col("p80_ms").round(3))
        ).alias("p20-p80"),
        (pl.col("_ref_ms") / pl.col("p50_ms")).round(2).alias("speedup"),
        pl.col("pct_peak_bw").round(1).alias("% peak BW"),
        pl.col("bw_floor_ms").round(3).alias("BW floor ms"),
        pl.col("kernels_per_call").alias("kernels"),
        pl.col("launch_floor_ms").round(3).alias("launch floor ms"),
        (pl.col("saved_extra_bytes") / MiB).round(1).alias("saved MiB"),
        (pl.col("peak_extra_bytes") / MiB).round(1).alias("peak MiB"),
        (pl.col("allocated_after_bytes") / MiB).round(1).alias("left allocated MiB"),
        pl.col("loss_rel_err")
        .map_elements(_sci, return_dtype=pl.String)
        .alias("loss err"),
        pl.col("grad_rel_err")
        .map_elements(_sci, return_dtype=pl.String)
        .alias("grad err"),
    )


def _sci(v: float | None) -> str | None:
    return None if v is None else f"{v:.1e}"


def verification_summary(df: pl.DataFrame) -> list[str]:
    """One line on the output check, plus a warning line per row above tolerance."""
    checked = df.filter(pl.col("loss_rel_err").is_not_null())
    if checked.height == 0:
        return ["Output check: not run for this experiment."]
    against = ", ".join(sorted(set(checked["checked_against"].drop_nulls())))
    max_loss = checked.select(pl.col("loss_rel_err").max()).item()
    max_grad = checked.select(pl.col("grad_rel_err").max()).item()
    lines = [
        f"Output check (same input, vs {against}): {checked.height} rows, "
        f"max loss rel err {max_loss:.1e}, max grad rel err {max_grad:.1e} "
        f"(tolerance {VERIFY_TOLERANCE:.0e})."
    ]
    bad = checked.filter(
        (pl.col("loss_rel_err") > VERIFY_TOLERANCE)
        | (pl.col("grad_rel_err") > VERIFY_TOLERANCE)
    )
    for r in bad.iter_rows(named=True):
        lines.append(
            f"**WARNING** {r['shape']} {r['binning']} {r['impl']} {r['mode']}: "
            f"loss err {r['loss_rel_err']:.1e}, grad err {r['grad_rel_err']:.1e} "
            f"vs {r['checked_against']}"
        )
    return lines


def _cell(v: Any) -> str:
    return "" if v is None else str(v)


def to_markdown(df: pl.DataFrame) -> str:
    header = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(_cell(v) for v in row) + " |" for row in df.rows()]
    return "\n".join([header, sep, *body])


def to_latex(df: pl.DataFrame, caption: str) -> str:
    esc = [c.replace("%", r"\%").replace("_", r"\_") for c in df.columns]
    lines = [
        r"\begin{table}[t]",
        r"\centering\small",
        r"\begin{tabular}{" + "l" * 3 + "r" * (len(esc) - 3) + "}",
        r"\toprule",
        " & ".join(esc) + r" \\",
        r"\midrule",
        *(
            " & ".join(_cell(v).replace("_", r"\_") for v in row) + r" \\"
            for row in df.rows()
        ),
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{{caption}}}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def _groups(df: pl.DataFrame) -> list[tuple[str, str, pl.DataFrame]]:
    keys = df.select("binning", "mode").unique(maintain_order=True).rows()
    return [
        (b, m, df.filter((pl.col("binning") == b) & (pl.col("mode") == m)))
        for b, m in keys
    ]


def _bar_panel(
    ax: Any,
    sub: pl.DataFrame,
    impls: list[str],
    colors: dict[str, str],
    column: str,
    label: str,
    log: bool,
) -> None:
    shapes = sub["shape"].unique(maintain_order=True).to_list()
    height = 0.8 / max(1, len(impls))
    for i, impl in enumerate(impls):
        rows = {
            r["shape"]: r[column]
            for r in sub.filter(pl.col("label") == impl).iter_rows(named=True)
        }
        labelled = False
        for s_idx, shape in enumerate(shapes):
            v = rows.get(shape)
            if v is None:
                continue
            y = s_idx + i * height
            ax.barh(
                y,
                v,
                height=height - 0.02,
                color=colors[impl],
                hatch="////" if impl.endswith("+graph") else None,
                edgecolor=SURFACE,
                linewidth=0,
                label=None if labelled else impl,
            )
            labelled = True
            ax.text(v, y, f" {v:.3g}", va="center", fontsize=7, color=TEXT_2)
    ax.set_yticks([s + 0.4 - height / 2 for s in range(len(shapes))], shapes)
    ax.invert_yaxis()
    if log:
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_title(label, fontsize=9, color=TEXT, loc="left")
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=TEXT_2, labelsize=8)


def write_plots(df: pl.DataFrame, impl_order: list[str], out: Path) -> list[Path]:
    """Per binning x mode: time, % peak bandwidth, peak memory, one panel each."""
    plot_dir = out / "plots"
    plot_dir.mkdir(exist_ok=True)
    ok = df.filter(pl.col("status") == "ok").with_columns(
        pl.when(pl.col("cuda_graph"))
        .then(pl.col("impl") + " +graph")
        .otherwise(pl.col("impl"))
        .alias("label"),
        (pl.col("peak_extra_bytes") / MiB).alias("peak_mib"),
    )
    labels = [
        lbl
        for impl in impl_order
        for lbl in (impl, impl + " +graph")
        if lbl in set(ok["label"].to_list())
    ]
    # colour follows the implementation (config order), never its rank in a chart;
    # "+graph" rows share their implementation's hue with hatching
    colors = {}
    for i, impl in enumerate(impl_order):
        colors[impl] = colors[impl + " +graph"] = SERIES[i % len(SERIES)]
    paths = []
    panels = [
        ("p50_ms", "Time per call, ms (p50, log)", True),
        ("pct_peak_bw", "% of peak DRAM bandwidth", False),
        ("peak_mib", "Peak extra memory, MiB", False),
    ]
    for binning, mode, sub in _groups(ok):
        present = [p for p in panels if sub[p[0]].drop_nulls().len() > 0]
        if not present:
            continue
        n_shapes = sub["shape"].n_unique()
        fig, axes = plt.subplots(
            1,
            len(present),
            figsize=(
                4.2 * len(present),
                0.9 + 0.55 * n_shapes * max(1, len(labels)) ** 0.5,
            ),
            facecolor=SURFACE,
            squeeze=False,
        )
        for ax, (col, title, log) in zip(axes[0], present, strict=True):
            ax.set_facecolor(SURFACE)
            _bar_panel(ax, sub, labels, colors, col, title, log)
        handles, names = axes[0][0].get_legend_handles_labels()
        fig.legend(
            handles,
            names,
            loc="lower center",
            ncol=min(4, len(names)),
            frameon=False,
            fontsize=8,
            labelcolor=TEXT,
        )
        fig.suptitle(
            f"{binning} binning, {mode}", x=0.01, ha="left", fontsize=10, color=TEXT
        )
        fig.tight_layout(rect=(0, 0.08, 1, 0.95))
        path = plot_dir / f"{binning}-{mode}.png"
        fig.savefig(path, dpi=150, facecolor=SURFACE)
        plt.close(fig)
        paths.append(path)
    return paths


def write_summary(df: pl.DataFrame, meta: dict[str, Any], out: Path) -> str:
    env, config = meta["env"], meta["config"]
    lines = [
        f"# {config['name']} ({meta['run_id']})",
        "",
        f"Hypothesis: {config['hypothesis']}",
        "",
        f"Device: {env['device_name']} | peak BW {env['peak_bandwidth_gbps']} GB/s | "
        f"logits {config.get('dtype', 'float32')} | "
        f"torch {env['torch_version']} | triton {env['triton_version']} | "
        f"git {env['git_sha']}{' (dirty)' if env['git_dirty'] else ''}",
        "",
    ]
    df = _clean(df)
    lines += [*verification_summary(df), ""]
    tex = []
    for binning, mode, sub in _groups(df):
        table = _table(sub)
        lines += [f"## {binning} binning, {mode}", "", to_markdown(table), ""]
        tex.append(to_latex(table, f"{config['name']}: {binning} binning, {mode}"))
    plots = write_plots(df, config["impls"], out)
    lines += [f"![{p.stem}](plots/{p.name})" for p in plots]
    text = "\n".join(lines)
    (out / "summary.md").write_text(text)
    (out / "summary.tex").write_text("\n\n".join(tex))
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild tables/plots of a run")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    df = pl.read_parquet(args.run_dir / "rows.parquet")
    meta = json.loads((args.run_dir / "meta.json").read_text())
    print(write_summary(df, meta, args.run_dir))


if __name__ == "__main__":
    main()
