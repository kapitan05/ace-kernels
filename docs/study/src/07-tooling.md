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

::code bench/configs/sprint11_deepsup.yaml:1-19 | Name, W&B group, the hypothesis (written before the run), implementations, binnings, modes, shapes. `cuda_graph: true` adds a row with the call captured in a CUDA graph.

## The harness

**Isolation.** Each group runs in its own Python process. A CUDA error, a memory leak
or an out-of-memory in one group cannot poison the next (this was added after a
leak produced 38 false out-of-memory rows in Sprint 3):

::code src/ace_bench/run.py::run_isolated | The parent sends the job as JSON on stdin; the worker prints its rows after a marker line; a crash becomes error rows instead of a lost run.

**What a row records.** Time (median, p20, p80 of 10+ cold-L2 repetitions), effective
bandwidth and % of the GPU's peak, the bandwidth and launch floors, kernels per call,
peak extra memory, bytes saved for backward, and the output check vs the reference
and vs float64 (page 2):

::code src/ace_bench/measure.py::Row | One measurement = one row of the parquet file; every table in this guide is a query on these columns.

**Floors: predicted vs measured.** The calculator predicts the best possible time
from first principles; the distance to it is the honest measure of "how optimized":

::code src/ace_bench/calculator.py:24-40 | Ideal bytes (inputs read once in forward, twice + gradient write in backward), bandwidth floor = bytes / peak GB/s, launch floor = kernels x 5 us.

## Remote GPU: Kaggle from the command line

No notebook: `ace_bench.remote` uploads a `git archive` of the current commit as a
private Kaggle dataset, pushes a script kernel with a T4, polls its status and
downloads `results.tgz`:

::code src/ace_bench/remote.py::upload_snapshot | The snapshot contains a COMMIT file: results are always traceable to the exact code (and `fetch` warns if they do not match).

Every job script starts with the same setup: GPU check, locked environment
(`uv sync --frozen`), then the correctness gate:

::code kaggle/_setup.sh:1-28 | `set -euo pipefail`; a failing test aborts the job before any benchmark (dev jobs may continue with `GATE_CONTINUE=1` to also get timings).

::run uv run python -m ace_bench.remote submit kaggle/run_sprint11.sh --no-wait | push the job; then `... remote status kaggle/run_sprint11.sh` and `... remote fetch kaggle/run_sprint11.sh`

## Experiment tracking: W&B

Parquet files in git-ignored `results/` are the source of truth; W&B (project
`ace-kernels`) is the shareable mirror: one W&B run per benchmark run, grouped by
sprint, with the config, hypothesis, environment, the row table, plots and profiles.
API keys only come from the environment, never from code:

::code src/ace_bench/tracking.py::mirror_run | Upload an already-finished result folder (runs on the Mac after `fetch`).

## Profiler and memory snapshots

`ace_bench.profile` records three steps with `torch.profiler` (kernel names and GPU
time per kernel, `trace.json` for [ui.perfetto.dev](https://ui.perfetto.dev)), the
tensors saved for backward, and a CUDA memory snapshot for
[pytorch.org/memory_viz](https://pytorch.org/memory_viz) (allocation timeline: you see
the original's per-voxel tensors pile up and ours not). `ncu` (hardware counters) is
blocked on Kaggle, so occupancy and DRAM-byte counters wait for the RTX PRO 6000.

::code src/ace_bench/profile.py::profile_case | Warm-up step, then 3 profiled steps; NVTX ranges for Nsight Systems when available.

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
