# 7. Tooling: benchmark harness, Kaggle, W&B, profiler

The results on the other pages are only as trustworthy as the measurement. This page
explains the machinery that produced every number, and the bugs it had.

## The workflow of one experiment

1. Write a **config** with a hypothesis (YAML, validated by pydantic).
2. Commit; submit a **Kaggle job**: a snapshot of the commit runs on a T4, starting
   with the full test suite (the gate).
3. The **benchmark harness** runs every (shape, binning, implementation) group in a
   fresh process and writes `rows.parquet`, `summary.md`, plots.
4. **Fetch** the results into `results/kaggle/<commit>-<job>/`; mirror to **W&B**.
5. Write the report section; rebuild the **ablation** table; tag the sprint in git.

## Configs: the experiment as data

::walk bench/configs/sprint11_deepsup.yaml:1-19
@ # Sprint 11: the four deep-supervision levels | **Identity**: `name` (output folder and W&B run name) and `group` (W&B grouping per sprint).
@ hypothesis: >- | **The hypothesis is part of the data**, written before the run: it is printed in the summary and stored in W&B, so a result is always read against what was expected.
@ device: cuda | **What to measure**: implementations x binnings x modes (`fwd`, `fwd_bwd`); `cuda_graph: true` adds a second row per case with the call captured in a CUDA graph; warm-up and repetitions per row.
@ shapes: | **Inputs**: batch, classes, spatial size per named shape (optionally `logit_scale` / `target_bias` for realistic peaked inputs). The file is validated by the pydantic `BenchConfig`: a typo is an error, not a silently ignored key.
::end

## The harness

**Isolation.** Each group runs in its own Python process. A CUDA error, a memory leak
or an out-of-memory in one group cannot poison the next (this was added after a
leak produced 38 false out-of-memory rows in Sprint 3):

::walk src/ace_bench/run.py:66-157
@ def run_group(job: Job) -> list[Row]: | **One group** = one (shape, binning, implementation), all its modes and graph variants. First the reference result and float64 truth for this input (page 2), then one measured row per mode.
@ for mode, graphed in itertools.product(config.modes, graph_options): | `measure_case` builds the input, checks the output, counts kernels, measures memory and times the call. After each row, `release_memory` frees everything, and `allocated_after_bytes` records what is still allocated: if it grows from row to row, there is a leak.
@ _ROWS_MARKER = "ACE_BENCH_ROWS:" | **Process isolation.** The parent runs each group as `python -m ace_bench.run --worker` with the job as JSON on stdin; the worker prints its rows after a unique marker line (anything else it prints, e.g. warnings, is ignored).
@ error = f"worker exited with code {proc.returncode}, no results" | A crash (CUDA error, killed process) does not lose the run: the group becomes rows with `status="error"` and the next group starts in a clean process.
@ def worker() -> None: | The worker side: parse the job, run the group, print the rows as JSON.
::end

**What a row records.** Time (median, p20, p80 of 10+ cold-L2 repetitions), effective
bandwidth and % of the GPU's peak, the bandwidth and launch floors, kernels per call,
peak extra memory, bytes saved for backward, and the output check vs the reference
and vs float64 (page 2):

::walk src/ace_bench/measure.py::Row
@ class Row(BaseModel): | **One measurement**, a pydantic model (typed, validated) that becomes one row of `rows.parquet`. Identity: implementation, binning, mode, graph, input dtype, shape.
@ status: Status | `ok`, `oom` (out of memory: a result, e.g. "the reference does not fit at 288^3"), `error`, `capture_failed` (CUDA graph impossible), `skipped`.
@ p50_ms: float | None = None | **Time**: median, 20th / 80th percentile, mean, standard deviation over `reps` cold-L2 repetitions.
@ input_bytes: int | **Roofline columns**: bytes of input and target, ideal bytes moved, achieved bandwidth, % of the GPU's peak, the bandwidth floor and the launch floor (kernels x 5 us).
@ kernels_per_call: int | None = None | Kernels (GPU launches) and ATen operators per call, counted by the profiler.
@ saved_extra_bytes: int | None = None | **Memory**: bytes saved for backward beyond the inputs, peak extra memory during the call, and memory still allocated afterwards (leak detector).
@ checked_against: str | None = None | **Accuracy** (page 2): errors vs the reference and vs float64, max-norm and L2, plus the reference's own errors vs float64 for comparison.
::end

**Floors: predicted vs measured.** The calculator predicts the best possible time
from first principles; the distance to it is the honest measure of "how optimized":

::walk src/ace_bench/calculator.py:24-40
@ def ideal_bytes(input_bytes: int, target_bytes: int, mode: Mode) -> int: | **The minimum traffic** of a perfect implementation: forward reads logits and target once; forward + backward reads them twice (forward, backward) and writes the gradient once (same size as the logits).
@ def bandwidth_floor_ms(nbytes: int, peak_gbps: float) -> float: | Bytes / peak bandwidth (T4: 320 GB/s; RTX PRO 6000: ~1.8 TB/s): no memory-bound implementation can be faster.
@ def launch_floor_ms(kernels: int, launch_us: float = LAUNCH_US) -> float: | Kernels x ~5 us: the CPU cost of launching, the lower bound for tiny inputs.
@ def effective_gbps(nbytes: int, ms: float) -> float: | Ideal bytes / measured time: the bandwidth the run *effectively* achieved; divided by the peak it gives `% of peak` (hardware-independent: comparable between the T4 and the RTX PRO 6000).
::end

## Remote GPU: Kaggle from the command line

No notebook: `ace_bench.remote` uploads a `git archive` of the current commit as a
private Kaggle dataset, pushes a script kernel with a T4, polls its status and
downloads `results.tgz`:

::walk src/ace_bench/remote.py:65-158
@ def upload_snapshot(user: str) -> str: | **Only committed code runs**: uncommitted changes abort the upload. `git archive HEAD` packs exactly the committed tree, plus a `COMMIT` file with the hash.
@ meta = { | The tarball is uploaded as a new version of a **private** Kaggle dataset (`kaggle datasets version`), which the GPU job mounts as its source code.
@ print(f"uploaded {sha[:12]} -> dataset {dataset_id}, waiting until ready") | Kaggle processes uploads asynchronously: wait until the new version is ready, or the job would start with the previous commit's code.
@ def push_kernel(user: str, script: Path) -> str: | **The GPU job**: a tiny `entry.py` (unpacks the source, runs the given `kaggle/run_*.sh`) pushed as a private *script kernel* with a T4 accelerator and internet (to install the locked environment with `uv`).
@ def status(kid: str) -> str: | Polls `kaggle kernels status` (parsing both old and new output formats of the CLI; a format change once broke this).
@ def fetch(kid: str, script: Path) -> Path: | Downloads the job's output, reads its `COMMIT`, and stores everything as `results/kaggle/<commit>-<job>/` with `results.tgz` extracted: every result folder names the exact code that produced it.
::end

Every job script starts with the same setup: GPU check, locked environment
(`uv sync --frozen`), then the correctness gate:

::walk kaggle/_setup.sh:1-28
@ # Shared by kaggle/run_*.sh (sourced): GPU info, locked env, correctness gate. | `set -euo pipefail`: stop at the first failing command, unset variable, or failure inside a pipe (without `pipefail`, `pytest | tail` reports tail's success: a real bug we had).
@ echo "== GPU" | Records which GPU and driver the job got.
@ echo "== Environment (uv, locked deps)" | `uv sync --frozen` installs exactly the versions in `uv.lock` (fails instead of resolving new ones). Then a check that the installed PyTorch has kernels for this GPU's architecture (e.g. sm_75 for the T4).
@ echo "== Correctness gate (CPU + CUDA)" | **The gate**: the full test suite, now including the CUDA/Triton tests that cannot run on a Mac. A failure stops the job: no benchmark of code that fails tests. Dev jobs can narrow the tests and continue anyway to get early timings.
@ WANDB="" | W&B logging is enabled only if an API key is present in the environment (Kaggle secret); it never appears in code or logs.
::end

::run uv run python -m ace_bench.remote submit kaggle/run_sprint11.sh --no-wait | push the job; then `... remote status kaggle/run_sprint11.sh` and `... remote fetch kaggle/run_sprint11.sh`

## Experiment tracking: W&B

Parquet files in git-ignored `results/` are the source of truth; W&B (project
`ace-kernels`) is the shareable mirror: one W&B run per benchmark run, grouped by
sprint, with the config, hypothesis, environment, the row table, plots and profiles.
API keys only come from the environment, never from code:

::walk src/ace_bench/tracking.py::mirror_run
@ def mirror_run(run_dir: Path, project: str = "ace-kernels") -> str | None: | Uploads an already finished, fetched run (on the Mac): W&B is a mirror, the parquet files remain the source of truth.
@ with Tracker( | One W&B run per benchmark run: grouped by sprint, the full config (incl. hypothesis) and environment as run config, implementations and GPU as tags.
@ tracker.log_table("results", df) | The rows as an interactive table, the plots as images, the whole folder as a versioned artifact; returns the run URL.
::end

## Profiler and memory snapshots

`ace_bench.profile` records three steps with `torch.profiler` (kernel names and GPU
time per kernel, `trace.json` for [ui.perfetto.dev](https://ui.perfetto.dev)), the
tensors saved for backward, and a CUDA memory snapshot for
[pytorch.org/memory_viz](https://pytorch.org/memory_viz) (allocation timeline: you see
the original's per-voxel tensors pile up and ours not). `ncu` (hardware counters) is
blocked on Kaggle, so occupancy and DRAM-byte counters wait for the RTX PRO 6000.

::walk src/ace_bench/profile.py::profile_case
@ def profile_case( | Profiles one (implementation, binning, shape, mode) and writes a folder of artifacts.
@ for _ in range(2):  # compile / allocator warmup | Two unprofiled steps first (compilation, autotuning, allocator warm-up would otherwise dominate the trace).
@ if nsys:  # nsys --capture-range=cudaProfilerApi records only this block | With Nsight Systems (where installed), mark exactly the steps to record.
@ activities = [ProfilerActivity.CPU] | **torch.profiler** with a schedule: skip 1 step, warm up 1, record `ACTIVE_STEPS` (3). It records every CPU operator and every GPU kernel with its duration.
@ prof.export_chrome_trace(str(out / "trace.json")) | **Artifacts**: `trace.json` (timeline for ui.perfetto.dev), `ops.txt` (operators sorted by GPU time, kernels per step), the saved-for-backward list.
@ _write_saved(out, x, y, config, impl) | A **CUDA memory snapshot** of one step (`memory_snapshot.pickle`, open at pytorch.org/memory_viz: every allocation over time with its Python stack). For the reference, also `torch._dynamo.explain` (graph breaks).
::end

## The ablation and the report

`ace_bench.ablation` collects every fetched run, drops runs known to be invalid
(documented with the reason), keeps the latest valid measurement per implementation x
shape, and writes `docs/ablation/` (markdown, LaTeX, plots):

::code src/ace_bench/ablation.py:25-49 | `EXCLUDED_RUNS` (with reasons) and the `LADDER`: the order and one-line meaning of every step.

## Version control: one tag per sprint

Every validated sprint is a git tag, so any state can be checked out, rerun and
compared:

::run git tag && git diff --stat sprint-9 sprint-11 | the list of sprint tags; what Sprint 11 changed

## Bugs in the tooling (and what they teach)

| bug | symptom | fix |
|---|---|---|
| saved-tensor pack hook returned the tensor itself | reference cycle, GPU memory leak, 38 false OOM rows | return `t.detach()`; process isolation |
| `pytest ... \| tail -1` | a failing suite looked green (tail's exit code) | `set -o pipefail` everywhere |
| grid axis 1 = voxel blocks | "invalid argument" at 256^3 (limit 65535) | blocks on axis 0 + regression test |
| first shape of a run slower for all Triton kernels | v6_index looked slower than v5 at kits23 | cross-check in a second run; noted in the report |
| float atomics compared bit-exactly in a test | false failures (order of atomic adds varies) | exact counts, sums within rounding |
| reference soft path: host-to-device copy | CUDA graph capture failed | a test asserts it (documented, not "fixed": it is the reference) |

::check Why does every benchmark group run in its own process? | So one group's failure (CUDA error, OOM, leak) cannot change the next group's measurements; and peak-memory numbers start from a clean allocator.
::check Why is the time measured with CUDA events and an L2 flush instead of time.time()? | GPU calls are asynchronous (Python returns early); events time the GPU work itself. The L2 flush makes each repetition read from DRAM, as in training, instead of from a warm cache.
::check Where do you find the exact code that produced a number in the report? | The run folder name starts with the commit hash (`results/kaggle/<commit>-<job>/`), the snapshot has a COMMIT file, and every row has `run_id` and the env (GPU, versions, git sha).
