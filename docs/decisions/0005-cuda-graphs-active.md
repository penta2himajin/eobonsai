# ADR 0005: CUDA graphs are active and worth 5%, correcting an earlier claim

- Status: **accepted, documentation correction** (measured 2026-09-27)
- Date: 2026-09-27
- Context: RTX 3060, `PQ2_0`, decode accounting

## What was claimed, and why it was wrong

`docs/decode-profile.md` argued the following: the GPU is saturated (34.45 ms of kernel
time against 34.22 ms of wall clock), therefore launch enqueue is fully hidden, therefore
"CUDA graphs, launch batching, and kernel-count reduction are not worth pursuing on this
workload".

The first two steps are right. The conclusion does not follow. A saturated GPU with ~1967
kernels per token says nothing about whether those kernels are being launched through a
graph; it only says the GPU has no idle gaps. Measured directly, the claim is false.

## Measurement

`llama-bench` tg128, same build, same commit, `REPS=3`, the only difference being the
fork's own switch (`common.cuh:1299`, `GGML_CUDA_DISABLE_GRAPHS`):

| build | tg128 | tg @ d8192 |
|---|---:|---:|
| graphs enabled (default) | **30.14 ± 0.07** | **28.47 ± 0.07** |
| `GGML_CUDA_DISABLE_GRAPHS=1` | 28.55 ± 0.02 | 27.08 ± 0.02 |
| | **-5.3%** | **-4.9%** |

So CUDA graphs are active on sm_86 during decode and are already carrying about 5% of
decode performance. Note the log line pairs ("CUDA graph warmup complete" / "warmup
reset") that appear in CLI runs with a prefill: those come from the prefill phase, where
graph properties genuinely change between calls. Steady-state decode stabilises, the
`cgraph->uid` fast path in `ggml_cuda_graph_update_required` takes over, and the captured
graph is reused.

## Why the forensic attempt failed first

Before finding the switch, three indirect methods were tried and all were misleading:

1. **`launchType` in the nsys SQLite.** Every one of the 3934 captured kernels reported
   `CUDA_KERNEL_LAUNCH_TYPE_REGULAR`, and this CUPTI build's enum has no graph-launch
   entry at all, so the field cannot distinguish graph nodes.
2. **Runtime API counts.** 7868 `cudaLaunchKernel` calls against 7 `cudaGraphLaunch`
   calls looked conclusive, but the nsys capture is known to be truncated (only 2 of 64
   passes) and the runtime table is not a reliable denominator.
3. **Warmup log lines.** "warmup complete" alternating with "warmup reset" suggested the
   graph was never reused, but those lines come from the prefill.

A single environment switch would have answered all of it. The lesson is the same one as
ADR 0004, in a different form: **when a mechanism can be toggled, toggle it and measure,
instead of inferring it from traces.**

## Consequences

- `docs/decode-profile.md` is corrected: the non-GEMV 17.4% is *not* recoverable by
  introducing CUDA graphs, because they are already in use and already amortising part of
  it. What remains is genuine execution time in small, latency-bound kernels.
- This also removes CUDA graphs from the list of future work, where the earlier text had
  placed it as "worth pursuing" by implication.
- The measured 5% is part of every decode number reported in this repository's documents,
  including the pre-existing baselines. Nothing needs re-measuring; it just needs stating.
- The remaining non-GEMV target is unchanged and still requires graph-level fusion of the
  normalisation/quantisation plumbing (`~2.1 ms/token` across 730 launches), whose ceiling
  is now known to be smaller than the 6 ms raw figure suggests, because part of that 6 ms
  is already graph-amortised launch cost.
