# 6. Training: the loss inside the real step, and torch.compile

A 30-75x faster loss does not make training 30-75x faster. This page measures the loss
inside the paper's actual training step and explains the numbers, including how our
kernels and `torch.compile` combine.

## The end-to-end harness

`ace_bench/e2e.py` rebuilds the paper's training setup: MONAI `SegResNetDS` (32
filters, deep supervision with 4 output levels), loss `0.33 * (Dice + CE + mL1-ACE)`
summed over the levels by `DeepSupervisionLoss`, AdamW, AMP (fp16). Inputs are
synthetic but learnable volumes, identical batches for every implementation.

::code src/ace_bench/e2e.py::PaperLoss | The paper's combined loss; only the `impl` of the ACE term changes between runs. The ACE value is accumulated as a GPU tensor (a Python float would force a host sync every step).

Each training run happens in a fresh process, measures every step with CUDA events,
and records loss, ACE term, step time and peak memory. Separate probes find the
largest cubic patch that still fits in the 15 GB of the T4.

::code src/ace_bench/e2e.py::_forward | Network + deep-supervision loss, optionally under `torch.compile` (the whole step's forward, so the network is compiled too).

## Concept: Amdahl's law

If a part takes a fraction `s` of the step and becomes `x` times faster, the step
becomes `1 / ((1 - s) + s / x)` times faster. With the loss at 10-19% of the original
step and 30x faster: `1 / (0.85 + 0.15/30) = 1.17x`. That is what we measure:

::plot docs/study/assets/figures/training-step.png | Median training-step time: original vs ours, eager vs torch.compile (T4, 128^3, AMP) | our kernels speed up the loss part; compile speeds up mainly the network; for hard binning the two add up (472 ms), for soft compile already fuses the original's loss well.

## torch.compile vs our kernels

All four numbers are relative to the same baseline: the original loss, no compile.
"+ compile" compiles **network and loss together**, so it includes compile's gains on
the network, which our work does not touch:

::table results/kaggle/60a3c6df2c3c-run-sprint11/results/*sprint11-e2e/steps.parquet | df.filter((pl.col("step") >= 5) & ~pl.col("impl").str.contains("rerun")).group_by("binning", "impl").agg(pl.col("step_ms").median().round(1).alias("median step ms"), pl.col("peak_mib").max().round(0).alias("peak MiB")).sort("binning", "median step ms", descending=[False, True]) | Sprint 11 end-to-end run, T4: "+default" = torch.compile (default mode) of the whole forward.

How to read it:

- **Ours alone (eager)** replaces only the loss: 1.10x (hard) / 1.23x (soft), capped by
  the loss's share of the step.
- **Compile alone** speeds up mainly the network (it fuses norm + activation layers),
  plus some of the loss: 1.11x / 1.37x.
- **Hard: the gains add up.** Compile saves ~56 ms (network), our kernels ~52 ms
  (loss); together ~113 ms (1.24x). They act on different parts.
- **Soft: they do not.** Compile already makes the original soft loss much faster
  (Sprint 2: 4.7x in isolation), so after compiling, the soft loss is a small share and
  our kernels add ~1 ms. Open question: in isolation the compiled original soft loss
  is still ~8x slower than ours (39 vs 5 ms), which predicts a ~15 ms gap per step;
  measured ~1 ms. Candidates: fp16 logits under AMP, overlap inside the compiled graph,
  single-run noise. Profiling the loss inside the compiled step would settle it.

What compile does **not** give, and ours does:

| | original + compile | ours |
|---|---|---|
| peak memory of the loss (kits23 soft) | 688 MiB | 64 MiB |
| 16 classes, soft (amos22) | 2694 ms (2x *slower* than eager) | 20 ms |
| warm-up | seconds of compilation, recompiles on new shapes | none |
| users who train eagerly | no gain | full gain |

And the graph breaks (`torch._dynamo.explain`): the 8 breaks per step come from MONAI's
`DiceLoss` (`torch.arange(...).tolist()`, 2 per level), for every ACE implementation.
The original ACE loss itself traces fine; v5 added 6 breaks; v6 (custom ops) adds 0:

::table results/kaggle/60a3c6df2c3c-run-sprint11/results/*sprint11-e2e/compile.parquet | df.select("impl", "binning", "graphs", "graph_breaks", "reasons") | Graph breaks of network + loss under torch.compile.

## Is training unchanged? The control run

The loss curves of ours and the original differ by a few percent after a few dozen
steps. Is that a bug? The **control** answers it: the original trained twice with the
same seed and batches also diverges, because cuDNN convolutions, atomics and AMP are
not bit-deterministic and training amplifies tiny differences:

::plot results/kaggle/60a3c6df2c3c-run-sprint11/results/*sprint11-e2e/plots/e2e-soft.png | Soft binning: loss curves (left) and step times (right); "rerun" = the original again | ours stays within the original's own run-to-run spread (rerun max difference 18%, ours 4.6%); step 0 is identical (8e-8), i.e. the forward is exact.

## Larger patches fit

::table results/kaggle/60a3c6df2c3c-run-sprint11/results/*sprint11-e2e/probes.parquet | df.select("impl", "binning", "patch", "status", pl.col("peak_mib").round(0).alias("peak MiB")) | Largest cubic patch probe (two training steps, 15 GB T4): hard binning, the original stops at 208^3, ours fits 224^3.

## Try it

::run uv run python -m ace_bench.e2e bench/configs/e2e_cpu_smoke.yaml | a tiny CPU version of the whole pipeline (curves, probes, compile, summary) in ~1 minute; not a result, a pipeline check

::check The loss is 30x faster but the step only 1.1-1.2x. Where is the rest of the time? | In the network (SegResNetDS forward + backward, ~80-90% of the original step). Amdahl: 1 / (0.85 + 0.15/30) ~ 1.17.
::check Why is a reference-vs-reference control run needed? | Training on a GPU is not bit-deterministic (cuDNN algorithms, atomics, AMP). Without knowing how much the original differs from itself, a difference between ours and the original cannot be attributed to our code.
::check Should a user compile, or use our kernels? | Both: they speed up different parts. Our kernels also bring memory savings, many-class speed and eager-mode gains that compile does not.
